"""Merge preprocessed household data with weather data.

Steps:
    1. read household and weather data
    2. fill missing weather values (weather_cleaning.py: drop unused columns,
       interpolate gaps up to 2 h, fill the rest from the most similar station)
    3. map the weather to the households via Weather_ID (weather_mapping.py)
    4. save the result
Use --no-weather-cleaning to skip step 2.

Put the preprocessed inputs in data/preprocessed/, as Parquet or CSV:
    household.parquet  or  household.csv  or  a folder household/ with .parquet/.csv files
    weather.parquet    or  weather.csv    or  a folder weather/ with .parquet/.csv files

Run from the repository root:
    python Combination/merge_weather.py
    python Combination/merge_weather.py --household path/to/household --weather path/to/weather --output path/to/out.parquet

The result is saved as data/preprocessed/household_weather.parquet; an --output
ending in .csv writes CSV instead. CSV files are semicolon-separated, like the raw data.
"""

import argparse
from pathlib import Path

import pandas as pd

from weather_cleaning import clean_weather
from weather_mapping import DATA_DIR, map_weather_to_households, missing_weather_report

PREPROCESSED_DIR = DATA_DIR / "preprocessed"
SUFFIXES = (".parquet", ".csv")


def read_table(path: Path) -> pd.DataFrame:
    """Read one Parquet or CSV file."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, sep=";")
    raise ValueError(f"unsupported file type: {path}")


def read_input(path: Path) -> pd.DataFrame:
    """Read one file, or all Parquet/CSV files in a folder, into one long table."""
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.suffix in SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no .parquet or .csv files in {path}")
        return pd.concat([read_table(f) for f in files], ignore_index=True)
    return read_table(path)


def find_input(name: str) -> Path:
    """Find data/preprocessed/<name>.parquet, <name>.csv or the folder <name>/."""
    for candidate in [PREPROCESSED_DIR / f"{name}{s}" for s in SUFFIXES] + [PREPROCESSED_DIR / name]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no {name}.parquet, {name}.csv or {name}/ folder in {PREPROCESSED_DIR}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, sep=";", index=False)
    else:
        df.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--household", type=Path, help="file or folder (default: found in data/preprocessed)")
    parser.add_argument("--weather", type=Path, help="file or folder (default: found in data/preprocessed)")
    parser.add_argument("--output", type=Path, default=PREPROCESSED_DIR / "household_weather.parquet")
    parser.add_argument("--no-weather-cleaning", action="store_true", help="merge the weather as it is")
    args = parser.parse_args()

    household_path = args.household or find_input("household")
    weather_path = args.weather or find_input("weather")
    household_df = read_input(household_path)
    weather_df = read_input(weather_path)
    print(f"household: {len(household_df):,} rows, {household_df['Household_ID'].nunique()} households ({household_path})")
    print(f"weather:   {len(weather_df):,} rows, {weather_df['Weather_ID'].nunique()} stations ({weather_path})")

    if not args.no_weather_cleaning:
        print()
        weather_df = clean_weather(weather_df)

    merged = map_weather_to_households(household_df, weather_df)

    weather_columns = [
        c for c in weather_df.columns
        if c not in ("Weather_ID", "Timestamp") and pd.api.types.is_float_dtype(weather_df[c])
    ]
    print("\nshare of missing weather values per station:")
    print(missing_weather_report(merged, weather_columns).to_string())

    write_table(merged, args.output)
    print(f"\nsaved {len(merged):,} rows to {args.output}")


if __name__ == "__main__":
    main()
