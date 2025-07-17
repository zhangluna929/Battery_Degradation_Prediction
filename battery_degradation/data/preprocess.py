import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing: handle missing values and sort by cycle."""
    df = df.copy()
    df.dropna(inplace=True)
    if "cycle" in df.columns:
        df.sort_values("cycle", inplace=True)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Generate additional engineered features for training."""
    df = df.copy()
    if "capacity" in df.columns:
        df["capacity_normalized"] = df["capacity"] / df["capacity"].iloc[0]
    return df 