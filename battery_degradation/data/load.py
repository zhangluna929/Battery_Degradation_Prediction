import pandas as pd
from pathlib import Path
from typing import Union


def load_battery_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load battery degradation dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded data frame.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    return df 