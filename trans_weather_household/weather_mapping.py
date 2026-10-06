"""Map weather data to households via the Weather_ID in households.csv.

Both inputs use UTC timestamps, which are joined as they are (no time shift).
Household rows are matched with the weather row of the same station and time;
if the weather is coarser than the household data (e.g. hourly), each household
row gets the weather row of the period it falls in.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
META_DIR = DATA_DIR / "smart_meter_meta_data"

HOUSEHOLD_COLUMNS = ["Household_ID", "Weather_ID", "Group", "Installation_HasPVSystem"]


def load_households(meta_dir: Path = META_DIR) -> pd.DataFrame:
    """Load households.csv with the columns needed for the mapping."""
    households = pd.read_csv(meta_dir / "households.csv", sep=";")
    return households[HOUSEHOLD_COLUMNS]


def _to_utc(timestamps: pd.Series) -> pd.Series:
    return pd.to_datetime(timestamps, utc=True)


def _resolution(df: pd.DataFrame, id_column: str) -> pd.Timedelta:
    """Most common time step between consecutive rows of the same ID."""
    steps = df.sort_values([id_column, "Timestamp"]).groupby(id_column)["Timestamp"].diff()
    return steps.mode().iloc[0]


def map_weather_to_households(
    household_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    households: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach weather features to every household row.

    household_df: long format with columns Household_ID, Timestamp, <values...>
    weather_df:   long format with columns Weather_ID, Timestamp, <features...>
    """
    if households is None:
        households = load_households()

    hh = household_df.copy()
    hh["Timestamp"] = _to_utc(hh["Timestamp"])

    weather = weather_df.copy()
    weather["Timestamp"] = _to_utc(weather["Timestamp"])
    weather_freq = _resolution(weather, "Weather_ID")
    weather = weather.rename(columns={"Timestamp": "_weather_time"})

    # only add meta columns the household data does not already contain (e.g. Group)
    meta_columns = ["Household_ID"] + [c for c in households.columns if c not in hh.columns]
    hh = hh.merge(households[meta_columns], on="Household_ID", how="left", validate="many_to_one")
    unmapped = hh.loc[hh["Weather_ID"].isna(), "Household_ID"].unique()
    if len(unmapped):
        raise ValueError(f"Household_IDs without Weather_ID: {list(unmapped)[:10]}")

    # 15-min weather: exact match; hourly weather: each 15-min row gets its hour
    hh["_weather_time"] = hh["Timestamp"].dt.floor(weather_freq)
    merged = hh.merge(
        weather,
        on=["Weather_ID", "_weather_time"],
        how="left",
        validate="many_to_one",
    ).drop(columns="_weather_time")

    assert len(merged) == len(hh), "merge changed the number of rows"
    return merged


def missing_weather_report(merged: pd.DataFrame, weather_columns: list[str]) -> pd.DataFrame:
    """Share of missing weather values per station and feature (0 = complete)."""
    return merged.groupby("Weather_ID")[weather_columns].apply(lambda df: df.isna().mean()).round(3)
