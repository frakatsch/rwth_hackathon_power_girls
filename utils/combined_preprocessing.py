import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder, PowerTransformer


def preprocess_temporal_data(
    df: pd.DataFrame,
    target_column: str,
    time_column: str,
    model_type: str = "linearregression",
    horizon: int = 24 * 4,
    is_training: bool = True,
    
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Sorts the DataFrame by the time column, creates lag features for the target variable, and extracts temporal features from the timestamp.
    Parameters:
    - df: pd.DataFrame
        The input DataFrame containing the time series data.
    - target_column: str
        The name of the target variable column in the DataFrame.
    - time_column: str
        The name of the time column in the DataFrame.
    - model_type: str, optional (default="linearregression")
        The type of model being used. This affects how categorical features are handled.
    - horizon: int, optional (default=24)
        The number of time steps ahead to predict. Used for creating the target variable for training.
    - is_training: bool, optional (default=True)
        Indicates whether the function is being called in a training context (True) or for inference (False). If True, the function will create a target variable for training; if False, it will not.
    
    Returns:
    - X: pd.DataFrame
        The features matrix.
    - y: pd.Series
        The target variable (if is_training is True).
    - dates: pd.Series
        The time stamps.
    """
    df = df.copy()
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame.")
    else:
        df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column).reset_index(drop=True)

    # Feature Engineering from timestamp
    df["year"] = df[time_column].dt.year
    df["month"] = df[time_column].dt.month
    df["week"] = df[time_column].dt.isocalendar().week.astype(int)
    df["day"] = df[time_column].dt.day
    df["dayofweek"] = df[time_column].dt.dayofweek
    df["hour"] = df[time_column].dt.hour

    # Target lags
    lags = [i * 4 for i in [0, 24, 48, 72, 96]]  # data from today, 1, 2, 3, and 4 days in the past
    for lag in lags:
        df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)

    # Target shift -> predicting the future target value
    if is_training:
        df["target_future"] = df[target_column].shift(-horizon)
        df = df.dropna(subset=["target_future"]).reset_index(drop=True)
        y = df["target_future"]
    else:
        y = None

    dates = df[time_column]

    # Drop non-predictive metadata from features matrix
    X = df.drop(columns=[target_column, "target_future"], errors="ignore")

    # drop nans
    X = X.dropna().reset_index(drop=True)
    y = y.loc[X.index] if y is not None else None
    dates = dates.loc[X.index]

    if model_type != "prophet":
        X = X.drop(columns=[time_column], errors="ignore")

    if model_type in ["linearregression", "ridge", "elasticnet", "xgboost"]:
        X = transform_categorical_features(X)
        
    # Skew transformation is generally only needed for linear models
    if model_type in ["linearregression", "ridge", "elasticnet"]:
        X = transform_skewed_features(X, skew_threshold=0.75)

    return X, y, dates


def transform_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes categorical features in the DataFrame X.
    """
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        encoded_features = encoder.fit_transform(X[categorical_cols])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
        X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)
    return X


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
    columns = X.select_dtypes(include=[np.number]).columns
    X = X.copy()
    for col in columns:
        skewness = X[col].skew()
        if skewness > skew_threshold:
            X[col] = np.log1p(X[col])
        elif skewness < -skew_threshold:
            X[col] = np.power(X[col], 0.5)
    return X