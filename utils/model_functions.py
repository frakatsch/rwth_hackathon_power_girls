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
from sklearn.compose import TransformedTargetRegressor

from typing import Dict, Tuple, Any

from utils.combined_preprocessing import preprocess_temporal_data
from utils.combined_preprocessing import SkewnessTransformer
from utils.combined_preprocessing import build_preprocessor


logger = logging.getLogger("Power Girls")
logging.basicConfig(level=logging.INFO)

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


class ProphetRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn wrapper for Meta Prophet with exogenous feature support."""
    def __init__(
        self,
        date_col: str = "datetime",
        growth: str = "linear",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        seasonality_mode: str = "additive"
    ):
        self.date_col = date_col
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_mode = seasonality_mode

    def _prepare_df(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, list]:
        X_df = pd.DataFrame(X).copy()

        if self.date_col in X_df.columns:
            ds = X_df.pop(self.date_col)
        elif isinstance(X_df.index, pd.DatetimeIndex):
            ds = X_df.index
        else:
            datetime_cols = X_df.select_dtypes(include=["datetime", "datetime64"]).columns
            if len(datetime_cols) > 0:
                ds = X_df.pop(datetime_cols[0])
            else:
                raise ValueError(f"Prophet requires '{self.date_col}' or a datetime column/index in X.")

        df = pd.DataFrame({"ds": pd.to_datetime(ds)}, index=X_df.index)
        exog_cols = list(X_df.columns)
        for col in exog_cols:
            df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0)

        if y is not None:
            df["y"] = np.asarray(y)

        return df, exog_cols

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df, exog_cols = self._prepare_df(X, y)
        self.exog_cols_ = exog_cols

        self.model_ = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            seasonality_mode=self.seasonality_mode
        )

        for col in self.exog_cols_:
            self.model_.add_regressor(col)

        self.model_.fit(df)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        df, _ = self._prepare_df(X)
        forecast = self.model_.predict(df)
        return forecast["yhat"].values


def safe_expm1(y_pred: np.ndarray) -> np.ndarray:
    """Clips log-space predictions before applying expm1 to prevent numerical explosion."""
    clipped_pred = np.clip(y_pred, -10.0, 9.21)
    return np.expm1(clipped_pred)


def get_local_filter(target_dt: pd.Timestamp, dates: pd.Series) -> np.ndarray:
    """Filters historical dates relative to a target prediction timestamp target_dt."""
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)

    t_year = target_dt.year
    t_date = target_dt.date()
    t_doy = target_dt.dayofyear
    t_hour = target_dt.hour
    
    # Hour Conditions (+/- 2 hours)
    hour_min, hour_max = (t_hour - 2) % 24, (t_hour + 2) % 24
    if hour_min < hour_max:
        similar_hours = dates.dt.hour.between(hour_min, hour_max)
    else: 
        similar_hours = (dates.dt.hour >= hour_min) | (dates.dt.hour <= hour_max)
        
    current_day_hour = dates.dt.hour == (t_hour - 2) % 24

    # Day Conditions
    doy_diff = (dates.dt.dayofyear - t_doy).abs()
    doy_diff = np.minimum(doy_diff, 365 - doy_diff)
    
    past_years_mask = (dates.dt.year < t_year) & (doy_diff <= 10) & similar_hours
    
    current_year_mask = (
        (dates.dt.year == t_year) & 
        (dates.dt.date < t_date) & 
        (dates >= target_dt - pd.Timedelta(days=10)) & 
        similar_hours
    )
    
    current_day_mask = (dates.dt.date == t_date) & current_day_hour

    return (past_years_mask | current_year_mask | current_day_mask).values


def tune_and_fit_pipeline(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    model_type: str = "linearregression",
    n_inner_splits: int = 3,
    time_column: pd.Series = None,
    random_state: int = 42,
    params: Dict = None
) -> Tuple[Pipeline, Dict]:

    date_col_name = "datetime"
    X_input = X_train.copy()
    
    if time_column is not None and date_col_name not in X_input.columns:
        X_input[date_col_name] = time_column.loc[X_input.index]

    preprocessor = build_preprocessor(X_train)

    if model_type in ["linearregression", "ridge", "elasticnet"]:
        if model_type == "linearregression":
            base_model = LinearRegression()
            param_grid = {
                "regressor__scaler": [StandardScaler(), RobustScaler()],
                "regressor__model__fit_intercept": [True, False]
            }
        elif model_type == "ridge":
            base_model = Ridge(random_state=random_state)
            param_grid = {
                "regressor__scaler": [StandardScaler(), RobustScaler()],
                "regressor__model__alpha": [0.01, 0.1, 1.0, 10.0]
            }
        else:
            base_model = ElasticNet(random_state=random_state)
            param_grid = {
                "regressor__scaler": [StandardScaler(), RobustScaler()],
                "regressor__model__alpha": [0.01, 0.1, 1.0],
                "regressor__model__l1_ratio": [0.2, 0.5, 0.8]
            }

        inner_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("transformer", SkewnessTransformer(skew_threshold=0.75)),
            ("scaler", StandardScaler()),
            ("model", base_model)
        ])
        
        pipeline = TransformedTargetRegressor(
            regressor=inner_pipeline,
            func=np.log1p,
            inverse_func=safe_expm1
        )

    elif model_type == "xgboost":
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", xgb.XGBRegressor(random_state=random_state, objective="reg:squarederror", n_jobs=-1))
        ])
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7]
        }

    elif model_type == "prophet":
        pipeline = Pipeline([
            ("model", ProphetRegressor(date_col=date_col_name))
        ])
        param_grid = {
            "model__changepoint_prior_scale": [0.01, 0.05, 0.1],
            "model__seasonality_prior_scale": [0.1, 1.0, 10.0],
            "model__seasonality_mode": ["additive", "multiplicative"]
        }

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if params is not None:
        pipeline.set_params(**params)
        pipeline.fit(X_input, y_train)
        return pipeline, params

    inner_cv = TimeSeriesSplit(n_splits=n_inner_splits)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1 if model_type == "prophet" else -1,
        verbose=0
    )

    grid_search.fit(X_input, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def predict_with_local_windows(
    base_pipeline_params: Dict,
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dates_train: pd.Series,
    X_eval: pd.DataFrame,
    dates_eval: pd.Series,
    min_samples: int = 15,
    random_state: int = 42
) -> np.ndarray:
    """
    Predicts X_eval row-by-row. For each observation i, filters X_train using
    dates_eval.iloc[i] to build a timestep-specific historical training set.
    """
    preds = np.zeros(len(X_eval))

    # Reset position-based slicing indices
    X_tr = X_train.reset_index(drop=True)
    y_tr = y_train.reset_index(drop=True)
    d_tr = dates_train.reset_index(drop=True)

    X_ev = X_eval.reset_index(drop=True)
    d_ev = dates_eval.reset_index(drop=True)

    for i in range(len(X_ev)):
        target_dt = d_ev.iloc[i]
        mask = get_local_filter(target_dt, d_tr)

        X_local = X_tr[mask]
        y_local = y_tr[mask]
        d_local = d_tr[mask]

        # Fallback to full historical dataset if local window is too small
        if len(X_local) < min_samples:
            X_local, y_local, d_local = X_tr, y_tr, d_tr

        local_pipeline, _ = tune_and_fit_pipeline(
            X_train=X_local,
            y_train=y_local,
            model_type=model_type,
            params=base_pipeline_params,
            time_column=d_local,
            random_state=random_state
        )

        X_single = X_ev.iloc[[i]]
        preds[i] = local_pipeline.predict(X_single)[0]

    return preds


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
    dates_train: pd.Series = None,
    dates_test: pd.Series = None
) -> Dict[str, float]:

    date_col = "datetime"
    X_tr = X_train.copy()
    X_te = X_test.copy()

    if dates_train is not None and date_col not in X_tr.columns:
        X_tr[date_col] = dates_train.loc[X_tr.index]
    if dates_test is not None and date_col not in X_te.columns:
        X_te[date_col] = dates_test.loc[X_te.index]

    # Target back-transformation happens automatically inside predict() if TransformedTargetRegressor is used
    y_train_pred = np.maximum(0, pipeline.predict(X_tr))
    y_test_pred = np.maximum(0, pipeline.predict(X_te))

    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    logger.info("Model Global Performance:")
    logger.info(f"  Training Set: R2 = {r2_train:.4f}, RMSE = {rmse_train:.4f}")
    logger.info(f"  Testing Set:  R2 = {r2_test:.4f}, RMSE = {rmse_test:.4f}")

    return {"r2_test": float(r2_test), "rmse_test": float(rmse_test)}


def model_training(
    inputs_df: pd.DataFrame,
    time_column: str,
    target_column: str,
    id_column: str = "id",
    model_type: str = "linearregression",
    n_outer_splits: int = 3,
    n_inner_splits: int = 2,
    random_state: int = 42
) -> Pipeline:

    X, y, dates = preprocess_temporal_data(inputs_df, target_column, time_column, id_column)

    tscv = TimeSeriesSplit(n_splits=n_outer_splits)
    outer_scores = []

    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(tscv.split(X)):
        logger.info(f"\n================ Outer Fold {outer_fold_idx + 1}/{n_outer_splits} ================")
        
        X_outer_train, X_outer_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_outer_train, y_outer_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]
        dates_outer_train, dates_outer_test = dates.iloc[outer_train_idx], dates.iloc[outer_test_idx]

        # 1. Tune hyperparameter architecture once for this outer fold
        _, best_params = tune_and_fit_pipeline(
            X_outer_train, 
            y_outer_train,
            model_type=model_type, 
            n_inner_splits=n_inner_splits, 
            time_column=dates_outer_train,
            random_state=random_state,
        )

        # 2. Evaluate holdout test fold using observation-specific local windows
        y_test_pred = predict_with_local_windows(
            base_pipeline_params=best_params,
            model_type=model_type,
            X_train=X_outer_train,
            y_train=y_outer_train,
            dates_train=dates_outer_train,
            X_eval=X_outer_test,
            dates_eval=dates_outer_test,
            random_state=random_state
        )

        # 3. Calculate metrics
        y_test_pred = np.maximum(0, y_test_pred)
        r2_test = r2_score(y_outer_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_outer_test, y_test_pred))

        logger.info(f"Outer Fold {outer_fold_idx + 1} Holdout Evaluation:")
        logger.info(f"  Testing Set: R2 = {r2_test:.4f}, RMSE = {rmse_test:.4f}")
        
        outer_scores.append(rmse_test)

    logger.info("\n================ Nested CV Summary ================")
    logger.info(f"Average estimated unseen RMSE: {np.mean(outer_scores):.4f} +/- {np.std(outer_scores):.4f}")

    # 3. Final Production Model Parameter Saving
    logger.info("\n================ Training Production Architecture ================")
    production_pipeline, prod_params = tune_and_fit_pipeline(
        X, y, 
        model_type=model_type, 
        n_inner_splits=n_inner_splits, 
        time_column=dates,
        random_state=random_state
    )
    
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / f"production_model_{model_type}.pkl", "wb") as f:
        pickle.dump(production_pipeline, f)
    with open(results_dir / f"production_params_{model_type}.pkl", "wb") as f:
        pickle.dump(prod_params, f)

    mean_rmse = float(np.mean(outer_scores))
    std_rmse = float(np.std(outer_scores))

    with open(results_dir / f"estimated_production_metrics_{model_type}.pkl", "wb") as f:
        pickle.dump({"mean_rmse": mean_rmse, "std_rmse": std_rmse}, f)

    return production_pipeline