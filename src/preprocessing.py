from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Steel_industry_data.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

def load_data():
    """Load the original Steel Industry Energy Consumption dataset."""

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Raw dataset loaded: {df.shape}")

    return df

def clean_data(df):
    """Perform basic cleaning and timestamp validation."""

    df = df.copy()

    # Convert timestamp
    df["date"] = pd.to_datetime(
        df["date"],
        format="%d/%m/%Y %H:%M"
    )

    # Ensure chronological order
    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    # Data-quality checks
    assert df["date"].duplicated().sum() == 0, \
        "Duplicate timestamps detected."

    assert df.isnull().sum().sum() == 0, \
        "Missing values detected."

    assert (df["Usage_kWh"] < 0).sum() == 0, \
        "Negative energy consumption detected."

    time_diff = df["date"].diff().dropna()

    assert (
        time_diff == pd.Timedelta(minutes=15)
    ).all(), "Irregular time intervals detected."

    print("Data quality checks passed.")

    return df

def add_calendar_features(df):
    """Create calendar-based time features."""

    df = df.copy()

    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df

def add_cyclical_features(df):
    """Create cyclical representations of daily, weekly and monthly time."""

    df = df.copy()

    time_of_day = df["hour"] + df["minute"] / 60

    # Daily cycle
    df["time_sin"] = np.sin(2 * np.pi * time_of_day / 24)
    df["time_cos"] = np.cos(2 * np.pi * time_of_day / 24)

    # Weekly cycle
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # Monthly cycle
    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )
    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )

    return df

def add_lag_features(df):
    """Create historical energy consumption lag features."""

    df = df.copy()

    df["lag_15min"] = df["Usage_kWh"].shift(1)
    df["lag_30min"] = df["Usage_kWh"].shift(2)
    df["lag_1h"] = df["Usage_kWh"].shift(4)
    df["lag_2h"] = df["Usage_kWh"].shift(8)
    df["lag_24h"] = df["Usage_kWh"].shift(96)
    df["lag_7d"] = df["Usage_kWh"].shift(672)

    lag_cols = [
        "lag_15min",
        "lag_30min",
        "lag_1h",
        "lag_2h",
        "lag_24h",
        "lag_7d"
    ]

    before = len(df)

    df = (
        df.dropna(subset=lag_cols)
        .reset_index(drop=True)
    )

    after = len(df)

    print(
        f"Lag features created: "
        f"{before - after} initial rows removed."
    )

    return df

def select_model_features(df):
    """Keep only forecasting-safe model features and target."""

    final_cols = [
        "date",

        # Calendar features
        "hour",
        "minute",
        "day_of_week",
        "month",
        "is_weekend",

        # Cyclical features
        "time_sin",
        "time_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",

        # Historical energy consumption
        "lag_15min",
        "lag_30min",
        "lag_1h",
        "lag_2h",
        "lag_24h",
        "lag_7d",

        # Target
        "Usage_kWh"
    ]

    df = df[final_cols].copy()

    assert df.isnull().sum().sum() == 0, \
        "Missing values remain in final dataset."

    print(f"Final model dataset: {df.shape}")

    return df

def split_data(df):
    """Split dataset chronologically into train, validation and test sets."""

    train = df[
        df["date"] < "2018-10-01"
    ].copy()

    validation = df[
        (df["date"] >= "2018-10-01") &
        (df["date"] < "2018-12-01")
    ].copy()

    test = df[
        df["date"] >= "2018-12-01"
    ].copy()

    # Prevent temporal overlap
    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()

    assert (
        len(train) +
        len(validation) +
        len(test)
        == len(df)
    )

    print(f"Train:      {train.shape}")
    print(f"Validation: {validation.shape}")
    print(f"Test:       {test.shape}")

    return train, validation, test

def save_datasets(train, validation, test):
    """Save processed datasets to disk."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    validation.to_csv(
        PROCESSED_DIR / "validation.csv",
        index=False
    )

    test.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    print(
        f"Processed datasets saved to: "
        f"{PROCESSED_DIR}"
    )

def main():
    print("Starting preprocessing pipeline...\n")

    df = load_data()

    df = clean_data(df)

    df = add_calendar_features(df)

    df = add_cyclical_features(df)

    df = add_lag_features(df)

    df = select_model_features(df)

    train, validation, test = split_data(df)

    save_datasets(
        train,
        validation,
        test
    )

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()



