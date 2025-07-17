"""CALCE Battery Data Set integration utilities.

The CALCE (Center for Advanced Life Cycle Engineering, University of Maryland)
prognostics battery data set provides aging and charge/discharge cycling data
for several commercial lithium-ion cells.

Dataset page: https://web.calce.umd.edu/battery-data/

The raw data are available as CSV (or MAT) files, organised by cell ID.
This helper downloads an archive (mirrored) on demand and converts selected
columns into a tidy DataFrame similar to NASA parser.

For simplicity, we assume each CSV contains the following columns:
    cycle, capacity, charge_time, discharge_time, voltage_mean, current_mean, temperature

Adjust parsers accordingly if your local copy differs.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas as pd

__all__ = ["download_calce_dataset", "load_calce_dataset"]

# URL to archive; replace if broken
CALCE_URL = "https://raw.githubusercontent.com/calceus/calce-battery-dataset/main/calce_battery.zip"
CALCE_FOLDER = "CALCE_BatteryData"


def _default_dir() -> Path:
    return Path.home() / ".cache" / "calce_battery"


def download_calce_dataset(root: str | os.PathLike | None = None, *, force: bool = False) -> Path:
    root_path = Path(root) if root is not None else _default_dir()
    root_path.mkdir(parents=True, exist_ok=True)
    archive_path = root_path / "calce_battery.zip"
    if force or not archive_path.exists():
        print("Downloading CALCE battery dataset…")
        urlretrieve(CALCE_URL, archive_path)
        print(f"Downloaded archive to {archive_path}")

    extract_path = root_path / CALCE_FOLDER
    if not extract_path.exists():
        print("Extracting CALCE battery dataset…")
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_path)
    return extract_path


def load_calce_dataset(root: str | os.PathLike | None = None) -> pd.DataFrame:
    data_dir = download_calce_dataset(root)
    csv_files = list(data_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError("No CSV files found in CALCE dataset directory.")
    dfs: List[pd.DataFrame] = []
    for f in csv_files:
        df = pd.read_csv(f)
        cell_id = f.stem
        if "battery_id" not in df.columns:
            df.insert(0, "battery_id", cell_id)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) 