import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class SkewnessTransformer(BaseEstimator, TransformerMixin):
    """Fits skewness thresholds on training data and transforms features consistently."""
    def __init__(self, skew_threshold: float = 0.75):
        self.skew_threshold = skew_threshold
        self.log_cols_ = []
        self.sqrt_cols_ = []
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)

        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        skewness = X_df[numeric_cols].skew()

        self.log_cols_ = skewness[skewness > self.skew_threshold].index.tolist()
        self.sqrt_cols_ = skewness[skewness < -self.skew_threshold].index.tolist()
        return self

    def transform(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            cols = self.feature_names_in_ if self.feature_names_in_ is not None else None
            X_df = pd.DataFrame(X, columns=cols)

        for col in self.log_cols_:
            if col in X_df.columns:
                X_df[col] = np.log1p(np.maximum(0, X_df[col]))

        for col in self.sqrt_cols_:
            if col in X_df.columns:
                X_df[col] = np.power(np.maximum(0, X_df[col]), 0.5)

        return X_df


def preprocess_temporal_data(
    df: pd.DataFrame,
    target_column: str,
    time_column: str,
    id_column: str,
    horizon: int = 24,
    is_training: bool = True
):
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column)

    # Resample to continuous hourly grid
    df = df.set_index(time_column).resample("1h").asfreq().reset_index()

    # Fill NaNs created in exogenous features by resampling
    exog_cols = [c for c in df.columns if c not in [target_column, time_column, id_column]]
    df[exog_cols] = df[exog_cols].ffill().bfill()

    # Time Features
    df["year"] = df[time_column].dt.year
    df["month"] = df[time_column].dt.month
    df["dayofweek"] = df[time_column].dt.dayofweek
    df["hour"] = df[time_column].dt.hour

    # Target Lags
    df[f"{target_column}_lag_0"] = df[target_column]
    for lag in [1, 2, 24, 48, 168]:
        df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)

    df[f"{target_column}_roll_mean_24"] = df[target_column].rolling(window=24).mean()

    # Target shift
    if is_training:
        df["target_future"] = df[target_column].shift(-horizon)
        df = df.dropna(subset=["target_future", f"{target_column}_lag_168"]).reset_index(drop=True)
        y = df["target_future"]
    else:
        y = None

    dates = df[time_column]
    X = df.drop(columns=[target_column, "target_future", time_column, id_column], errors="ignore")

    # drop nans
    X = X.dropna().reset_index(drop=True)
    y = y.loc[X.index] if y is not None else None
    dates = dates.loc[X.index]

    return X, y, dates


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Builds a leakage-free ColumnTransformer for categorical encoding and feature passthrough."""
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical_cols),
            ("num", "passthrough", numeric_cols)
        ],
        remainder="drop"
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def transform_skewed_features(X: pd.DataFrame, skew_threshold: float = 0.75) -> pd.DataFrame:
    """
    Apply a log transformation to right skewed, and Yeo-Johnson transformation to left skewed features in the DataFrame X.
    Skewness is calculated using the skew() function from pandas, and features with absolute skewness greater than the specified threshold are transformed.
    Parameters:
    - X: pd.DataFrame
        The input DataFrame containing features to be transformed.
    - skew_threshold: float, optional (default=0.75)
        The threshold for absolute skewness above which features will be transformed.

    Returns:
    - pd.DataFrame
        The DataFrame with transformed features.
    """
    transformer = SkewnessTransformer(skew_threshold=0.75)

    # Fit on training data and transform
    X_transformed = transformer.fit_transform(X)

    return X_transformed, transformer