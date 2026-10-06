"""Fill missing weather values before the weather is mapped to households.

The 15-minute weather data has three kinds of missing values. Each is handled
by its own step, in this order:

1. Drop columns that are empty by design or not useful
   - *_hourly_observation_15min: the original hourly observation, only filled
     on the full hour (75% empty). The *_15min versions of the same signal are
     complete and are kept.
   - Pressure_*: not measured at HbsbG, ceOxS and sV3mR (and one of them not at
     MqO); air pressure has hardly any influence on heat-pump or PV consumption.

2. Interpolate short gaps over time, per station
   Gaps of up to 2 hours (8 x 15 min) inside a station's series are filled by
   linear interpolation between the values before and after the gap. Longer
   gaps are left for step 3, because interpolating across hours or days would
   invent a smooth curve that never happened.

3. Fill the rest from the most similar station (aggregation across stations)
   The stations lie in the same region, so their weather moves together.
   Station similarity is the correlation of temperature and humidity between
   two stations over the whole period. For each missing value, the value of the
   most similar station that has data at that same timestamp is used.
   This also fills sunshine and precipitation for stations that never measure
   them (HbsbG, ceOxS and sV3mR have no sunshine data at all).

Added flag columns, so models and teammates can tell estimated from measured values:
   - Sunshine_duration_15min_imputed: sunshine was taken from another station
   - weather_filled_from_other_station: any weather value in the row was
     taken from another station (step 3)
"""

import pandas as pd

MAX_INTERPOLATION_STEPS = 8  # 8 x 15 min = 2 hours
SIMILARITY_COLUMNS = ["Temperature_avg_15min", "Humidity_avg_15min"]


def drop_unused_columns(weather: pd.DataFrame) -> pd.DataFrame:
    unused = [c for c in weather.columns if c.endswith("_hourly_observation_15min") or c.startswith("Pressure_")]
    return weather.drop(columns=unused)


def interpolate_short_gaps(weather: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    weather = weather.sort_values(["Weather_ID", "Timestamp"]).reset_index(drop=True)
    weather[value_columns] = weather.groupby("Weather_ID", observed=True)[value_columns].transform(
        lambda s: s.interpolate(limit=MAX_INTERPOLATION_STEPS, limit_area="inside")
    )
    return weather


def station_similarity(weather: pd.DataFrame) -> pd.DataFrame:
    """Mean correlation of temperature and humidity between every pair of stations."""
    correlations = [
        weather.pivot(index="Timestamp", columns="Weather_ID", values=column).corr()
        for column in SIMILARITY_COLUMNS
    ]
    return sum(correlations) / len(correlations)


def fill_from_similar_stations(weather: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    similarity = station_similarity(weather)
    filled_any = pd.Series(False, index=weather.index)

    for column in value_columns:
        wide = weather.pivot(index="Timestamp", columns="Weather_ID", values=column)
        filled = wide.copy()
        for station in wide.columns:
            donors = similarity[station].drop(station).sort_values(ascending=False).index
            for donor in donors:  # most similar first; the next one only where it is still missing
                filled[station] = filled[station].fillna(wide[donor])

        values = filled.stack(future_stack=True).rename(column)
        new = weather[["Timestamp", "Weather_ID"]].join(values, on=["Timestamp", "Weather_ID"])[column]
        was_missing = weather[column].isna() & new.notna()
        filled_any |= was_missing
        if column == "Sunshine_duration_15min":
            weather["Sunshine_duration_15min_imputed"] = was_missing
        weather[column] = new

    weather["weather_filled_from_other_station"] = filled_any
    return weather


def missing_share(weather: pd.DataFrame, value_columns: list[str]) -> pd.Series:
    return weather[value_columns].isna().mean()


def clean_weather(weather_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Run steps 1-3 (see module docstring) and return the cleaned weather table."""
    weather = weather_df.copy()
    weather["Weather_ID"] = weather["Weather_ID"].astype(str)
    weather["Timestamp"] = pd.to_datetime(weather["Timestamp"], utc=True)

    weather = drop_unused_columns(weather)
    value_columns = [c for c in weather.columns if c not in ("Weather_ID", "Timestamp")]
    before = missing_share(weather, value_columns)

    weather = interpolate_short_gaps(weather, value_columns)
    after_interpolation = missing_share(weather, value_columns)

    weather = fill_from_similar_stations(weather, value_columns)
    after_fill = missing_share(weather, value_columns)

    if verbose:
        dropped = sorted(set(weather_df.columns) - set(weather.columns))
        print(f"weather cleaning: dropped {len(dropped)} columns: {', '.join(dropped)}")
        report = pd.DataFrame(
            {"missing before": before, "after interpolation": after_interpolation, "after station fill": after_fill}
        )
        print("share of missing weather values (all stations):")
        print((report * 100).round(2).astype(str).add(" %").to_string())
    return weather
