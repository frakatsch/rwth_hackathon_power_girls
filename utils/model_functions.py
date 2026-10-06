import pandas as pd
import numpy as np
import logging
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
import xgboost as xgb
from prophet import Prophet
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

from typing import Dict, Tuple, Any

from utils.combined_preprocessing import preprocess_temporal_data
from utils.combined_preprocessing import transform_categorical_features
from utils.combined_preprocessing import transform_skewed_features


logger = logging.getLogger("Power Girls")
logging.basicConfig(level=logging.INFO)

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


class ProphetRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn wrapper for Meta Prophet to enable Pipeline and GridSearchCV support.
    """
    def __init__(
        self,
        date_col: str = None,
        growth: str = "linear",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        seasonality_mode: str = "additive",
        yearly_seasonality: Any = "auto",
        weekly_seasonality: Any = "auto",
        daily_seasonality: Any = "auto"
    ):
        self.date_col = date_col
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality

    def _prepare_df(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, list]:
        X_df = pd.DataFrame(X).copy()

        # Extract date column or use datetime index
        if self.date_col and self.date_col in X_df.columns:
            ds = X_df.pop(self.date_col)
        elif isinstance(X_df.index, pd.DatetimeIndex):
            ds = X_df.index
        else:
            datetime_cols = X_df.select_dtypes(include=["datetime", "datetime64"]).columns
            if len(datetime_cols) > 0:
                ds = X_df.pop(datetime_cols[0])
            else:
                raise ValueError("Prophet requires a DatetimeIndex or a datetime column in X.")

        df = pd.DataFrame({"ds": pd.to_datetime(ds)}, index=X_df.index)

        # Remaining columns are treated as exogenous regressors
        exog_cols = list(X_df.columns)
        for col in exog_cols:
            df[col] = X_df[col]

        if y is not None:
            df["y"] = np.asarray(y)

        return df, exog_cols

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df, exog_cols = self._prepare_df(X, y)

        self.model_ = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )

        for col in exog_cols:
            self.model_.add_regressor(col)

        self.model_.fit(df)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        df, _ = self._prepare_df(X)
        forecast = self.model_.predict(df)
        return forecast["yhat"].values



def get_local_filter(target_dt: pd.Timestamp, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Filters data for similar hours and days relative to a target prediction time."""
    
    t_year = target_dt.year
    t_date = target_dt.date()
    t_doy = target_dt.dayofyear
    t_hour = target_dt.hour
    
    # 1. Hour Conditions
    # +/- 2 hours (using modulo 24 to handle midnight wrap-around)
    hour_min, hour_max = (t_hour - 2) % 24, (t_hour + 2) % 24
    if hour_min < hour_max:
        similar_hours = dates.dt.hour.between(hour_min, hour_max)
    else: 
        similar_hours = (dates.dt.hour >= hour_min) | (dates.dt.hour <= hour_max)
        
    # Exactly -2h for the current day
    current_day_hour = dates.dt.hour == (t_hour - 2) % 24

    # 2. Day Conditions
    # Past years: +/- 10 days (calculating day-of-year distance)
    doy_diff = (dates.dt.dayofyear - t_doy).abs()
    doy_diff = np.minimum(doy_diff, 365 - doy_diff) # Handle Dec/Jan wrap-around
    past_years_mask = (dates.dt.year < t_year) & (doy_diff <= 10) & similar_hours
    
    # Current year: strictly past 10 days
    current_year_mask = (
        (dates.dt.year == t_year) & 
        (dates.dt.date < t_date) & 
        (dates >= target_dt - pd.Timedelta(days=10)) & 
        similar_hours
    )
    
    # Current day: only the -2h window
    current_day_mask = (dates.dt.date == t_date) & current_day_hour

    # 3. Combine and Apply
    final_mask = past_years_mask | current_year_mask | current_day_mask
    
    return final_mask


def tune_and_fit_pipeline(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    model_type: str = "linearregression",
    n_inner_splits: int = 3,
    time_column: str = None,
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
        X_with_date = X_train.copy()
        if time_column and time_column.name not in X_with_date.columns:
            X_with_date[time_column.name] = time_column

        pipeline = Pipeline([
            ("scaler", "passthrough"), # Prophet handles feature scaling internally
            ("model", ProphetRegressor(date_col=time_column.name))
        ])
        param_grid = {
            "scaler": ["passthrough"],
            "model__changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
            "model__seasonality_prior_scale": [0.1, 1.0, 10.0],
            "model__seasonality_mode": ["additive", "multiplicative"]
        }
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
            n_jobs=1 if model_type == "prophet" else -1,
            verbose=0
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
                   id_column: str = "id",
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
    id_column (str): The name of the ID column in the DataFrame.
    model_type (str): The type of model to train. Currently supports "linearregression".
    n_outer_splits (int): The number of outer cross-validation splits.
    n_inner_splits (int): The number of inner cross-validation splits.
    random_state (int): The random seed for reproducibility.

    Returns:
    Pipeline: The trained model pipeline.
    """

    # Additional preprocessing
    X, y, dates = preprocess_temporal_data(inputs_df, target_column, time_column, id_column)
    X = transform_categorical_features(X)  #TODO: OHE
    X = transform_skewed_features(X, skew_threshold=0.75)  # TODO: transform back?

    # Nested CV: outer loop
    tscv = TimeSeriesSplit(
        n_splits=n_outer_splits)

    outer_scores = []

    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(tscv.split(X)):
        logger.info(f"\n================ Outer Fold {outer_fold_idx + 1}/{n_outer_splits} ================")
        
        X_outer_train, X_outer_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_outer_train, y_outer_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]
        dates_outer_train, dates_outer_test = dates.iloc[outer_train_idx], dates.iloc[outer_test_idx]

        # Local filter
        for idx, row in X_outer_train.iterrows():
            target_dt = dates.iloc[idx]

            local_idx = get_local_filter(target_dt, dates_outer_train)
            X_outer_train_local = X_outer_train.loc[local_idx.index]
            y_outer_train_local = y_outer_train.loc[local_idx.index]

            if len(X_outer_train_local) < 10:
                continue
            else:
                logger.info(f"Outer Fold {outer_fold_idx + 1}: Local training data size for target_dt {target_dt} is {len(X_outer_train_local)}")

        # Nested CV: inner loop is handled automatically by GridSearchCV
        outer_pipeline, best_params = tune_and_fit_pipeline(
            X_outer_train_local, 
            y_outer_train_local,
            model_type=model_type, 
            n_inner_splits=n_inner_splits, 
            time_column=dates,
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

    logger.info("\n================ Nested CV Summary ================")
    logger.info(f"Average estimated unseen RMSE: {np.mean(outer_scores):.4f} +/- {np.std(outer_scores):.4f}")

    # 2. FINAL PRODUCTION MODEL
    # Fit on 100% of the available data to create the model you will actually deploy
    logger.info("\n================ Training Production Model ================")
    production_pipeline, prod_params = tune_and_fit_pipeline(
        X, y, 
        model_type=model_type, 
        n_inner_splits=n_inner_splits, 
        time_column=dates,
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

    return production_pipeline