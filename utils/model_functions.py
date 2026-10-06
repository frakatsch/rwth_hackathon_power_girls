import pandas as pd
import numpy as np
import logging
import pickle

import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

from typing import Dict, Tuple, Any

from utils.combined_preprocessing import preprocess_temporal_data

logger = logging.getLogger("Power Girls")
logging.basicConfig(level=logging.INFO)


def tune_and_fit_pipeline(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    model_type: str = "linearregression",
    n_inner_splits: int = 3,
    random_state: int = 42,
    params: Dict = None
) -> Tuple[Pipeline, Dict]:
    """
    Tunes pipeline hyperparameters using temporal inner cross-validation.
    Returns the best estimator retrained on the entire X_train set.
    """

    # 1. Define Pipeline and Hyperparameter Grid
    if model_type == "linearregression":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
        param_grid = {
            "scaler": [StandardScaler(), RobustScaler()],
            "model__fit_intercept": [True, False]
        }
    elif model_type == "ridge":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=random_state))
        ])
        param_grid = {
            "scaler": [StandardScaler(), RobustScaler()],
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "model__solver": ["auto", "saga"]
        }
    elif model_type == "elasticnet":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(random_state=random_state))
        ])
        param_grid = {
            "scaler": [StandardScaler(), RobustScaler()],
            "model__alpha": [0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    elif model_type == "xgboost":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBRegressor(random_state=random_state, objective="reg:squarederror", n_jobs=1))
        ])
        param_grid = {
            "scaler": ["passthrough", StandardScaler()], # Trees don't require scaling
            "model__n_estimators": [100, 200, 500],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        }
    elif model_type == "prophet":
        # Prophet requires a different approach and is not compatible with sklearn pipelines
        raise NotImplementedError("Prophet model type is not yet implemented in this pipeline.")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if params is not None:
        logger.info("Using provided hyperparameters. Skipping tuning.")
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        return pipeline, params
    else:
        inner_cv = TimeSeriesSplit(n_splits=n_inner_splits)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1 if model_type != "prophet" else 1,
            verbose=0,
        )

        # GridSearchCV automatically handles the inner loop splits and refits on the whole X_train
        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_, grid_search.best_params_


def evaluate_temporal_breakdown(
    y_true: pd.Series,
    y_pred: np.ndarray,
    dates: pd.Series
) -> None:
    """Calculates and logs RMSE across distinct temporal granularities."""
    # Ensure datetimelike dtype to avoid .dt accessor errors
    datetime_series = pd.to_datetime(dates.values if isinstance(dates, (pd.Series, pd.Index)) else dates)

    eval_df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "datetime": datetime_series
    })

    periods = {
        "Year": eval_df["datetime"].dt.year,
        "Month": eval_df["datetime"].dt.to_period("M"),
        "Week": eval_df["datetime"].dt.to_period("W"),
        "Day": eval_df["datetime"].dt.date
    }

    for period_name, grouping_col in periods.items():
        # logger.info(f"--- Performance by {period_name} ---")
        grouped = eval_df.groupby(grouping_col)
        for name, group in grouped:
            if len(group) > 0:
                rmse = np.sqrt(mean_squared_error(group["y_true"], group["y_pred"]))
                # logger.info(f"  {period_name} [{name}] (n={len(group)}): RMSE = {rmse:.4f}")


def regression_performance(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dates_test: pd.Series = None
) -> Dict[str, float]:
    """Evaluates global performance and prints temporal breakdowns efficiently."""
    # Single prediction pass per dataset
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Compute metrics from existing predictions
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    logger.info("Model Global Performance:")
    logger.info(f"  Training Set: R2 = {r2_train:.4f}, RMSE = {rmse_train:.4f}")
    logger.info(f"  Testing Set:  R2 = {r2_test:.4f}, RMSE = {rmse_test:.4f}")

    if dates_test is not None:
        evaluate_temporal_breakdown(y_test, y_test_pred, dates_test)

    return {"r2_test": float(r2_test), "rmse_test": float(rmse_test)}


def model_training(inputs_df: pd.DataFrame,
                   time_column: str,
                   target_column: str,
                   model_type: str = "linearregression",
                   n_outer_splits: int = 3,
                   n_inner_splits: int = 2,
                   random_state: int = 42
                ) -> Pipeline:
    """
    Train and evaluate a machine learning model using Nested Time-Series Cross Validation.

    Parameters:
    inputs_df (pd.DataFrame): The input DataFrame containing features and target variable.
    time_column (str): The name of the time column in the DataFrame.
    target_column (str): The name of the target column in the DataFrame.
    model_type (str): The type of model to train. Currently supports "linearregression".
    n_outer_splits (int): The number of outer cross-validation splits.
    n_inner_splits (int): The number of inner cross-validation splits.
    random_state (int): The random seed for reproducibility.

    Returns:
    Pipeline: The trained model pipeline.
    """

    # Additional preprocessing
    X, y, dates = preprocess_temporal_data(inputs_df, target_column, time_column, model_type=model_type)

    # Outer loop for cross-validation
    tscv = TimeSeriesSplit(
        n_splits=n_outer_splits,
    )

    outer_scores = []
    oof_predictions = pd.DataFrame({
        "time": dates,
        "y_true": np.nan,
        "y_pred": np.nan
    }, index=X.index)

    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(tscv.split(X)):
        logger.info(f"\n================ Outer Fold {outer_fold_idx + 1}/{n_outer_splits} ================")
        X_outer_train, X_outer_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_outer_train, y_outer_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]
        dates_outer_test = dates.iloc[outer_test_idx]

        # Inner loop is handled automatically by GridSearchCV inside this function
        outer_pipeline, best_params = tune_and_fit_pipeline(
            X_outer_train, 
            y_outer_train, 
            model_type=model_type, 
            n_inner_splits=n_inner_splits, 
            random_state=random_state,
        )

        logger.info(f"Outer Fold {outer_fold_idx + 1} Final Holdout Evaluation:")
        metrics = regression_performance(
            outer_pipeline, 
            X_outer_train, 
            y_outer_train, 
            X_outer_test, 
            y_outer_test, 
            dates_test=dates_outer_test
        )
        outer_scores.append(metrics["rmse_test"])

        oof_predictions.loc[outer_test_idx, "y_true"] = y_outer_test.values
        oof_predictions.loc[outer_test_idx, "y_pred"] = outer_pipeline.predict(X_outer_test)

    logger.info("\n================ Nested CV Summary ================")
    logger.info(f"Average estimated unseen RMSE: {np.mean(outer_scores):.4f} +/- {np.std(outer_scores):.4f}")

    # 2. FINAL PRODUCTION MODEL
    # Fit on 100% of the available data to create the model you will actually deploy
    logger.info("\n================ Training Production Model ================")
    production_pipeline, prod_params = tune_and_fit_pipeline(
        X, y, 
        model_type=model_type, 
        n_inner_splits=n_inner_splits, 
        random_state=random_state
    )
    
    logger.info("Production pipeline successfully tuned and trained. Moving to save the model and parameters for deployment.")
    
    # Save the production model and parameters for deployment
    with open(f"results/production_model_{model_type}.pkl", "wb") as f:
        pickle.dump(production_pipeline, f)
    with open(f"results/production_params_{model_type}.pkl", "wb") as f:
        pickle.dump(prod_params, f)

    # save the metrics of the production model
    mean_rmse = np.mean(outer_scores)
    std_rmse = np.std(outer_scores)

    logger.info(f"Production model metrics: mean_rmse={mean_rmse:.4f}, std_rmse={std_rmse:.4f}")

    with open(f"results/estimated_production_metrics_{model_type}.pkl", "wb") as f:
        pickle.dump({"mean_rmse": mean_rmse, "std_rmse": std_rmse}, f)

    # save the oof predictions of the production model
    oof_predictions.to_pickle(f"results/estimated_production_predictions_{model_type}.pkl")


    return production_pipeline