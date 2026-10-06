import pandas as pd
import numpy as np
from typing import Tuple


def preprocess_temporal_data(
    df: pd.DataFrame,
    target_column: str,
    time_column: str,
    id_column: str,
    
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Sorts data chronologically, extracts datetime features, and separates features/target.
    """
    df = df.copy()
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame.")
    else:
        df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column).reset_index(drop=True)

    # Date metadata retained for temporal breakdown calculations
    dates = df[time_column]

    # Feature Engineering from timestamp
    df["year"] = df[time_column].dt.year
    df["month"] = df[time_column].dt.month
    df["week"] = df[time_column].dt.isocalendar().week.astype(int)
    df["day"] = df[time_column].dt.day
    df["dayofweek"] = df[time_column].dt.dayofweek
    df["hour"] = df[time_column].dt.hour

    # Drop non-predictive metadata from features matrix
    X = df.drop(columns=[target_column, time_column, id_column], errors="ignore")
    y = df[target_column]

    return X, y, dates


def transform_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes categorical features in the DataFrame X.
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
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