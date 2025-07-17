"""NASA Battery Aging Dataset integration utilities.

This module provides helper functions to download, extract and parse the
NASA battery aging dataset (Prognostics Center of Excellence, PCoE).

The raw dataset consists of MATLAB ``.mat`` files for several lithium-ion
cells (e.g., B0005, B0006 …). We convert each cycle into a tidy
``pandas.DataFrame`` with one row per cycle containing aggregated features
(capacity, mean voltage/current, discharge time, temperature, etc.).

Usage example
-------------
>>> from battery_degradation.data import load_nasa_dataset
>>> df = load_nasa_dataset()
>>> df.head()

Notes
-----
Downloading the entire archive (~600 MB) may take several minutes depending
on your network speed and the mirror availability. You can set the
``NASA_BATTERY_DATA`` environment variable to point to a pre-downloaded copy
and skip the download step.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas as pd
import scipy.io as sio  # type: ignore

__all__ = [
    "download_nasa_dataset",
    "load_nasa_dataset",
]

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

NASA_DATA_URL = "https://data.nasa.gov/download/vrks-gjie/application%2Fx-tar"  # NOTE: Update to a valid mirror if this URL changes.

NASA_FOLDER_NAME = "NASA_BatteryData"

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def download_nasa_dataset(root: str | os.PathLike | None = None, *, force: bool = False) -> Path:
    """Download and extract the NASA battery dataset archive.

    Parameters
    ----------
    root : Path-like, optional
        Destination directory. Defaults to ``~/.cache/nasa_battery``.
    force : bool, default False
        If *True*, re-download even if archive already exists.

    Returns
    -------
    Path
        Path to the extracted dataset directory.
    """
    root_path = Path(root) if root is not None else _default_cache_dir()
    root_path.mkdir(parents=True, exist_ok=True)

    archive_path = root_path / "nasa_battery.tar.gz"

    if force or not archive_path.exists():
        print("Downloading NASA battery dataset… (may take a while)")
        urlretrieve(NASA_DATA_URL, archive_path)
        print(f"Downloaded archive to {archive_path}")

    extract_path = root_path / NASA_FOLDER_NAME
    if not extract_path.exists():
        print("Extracting archive…")
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract(tar, path=extract_path)
        print(f"Extracted dataset to {extract_path}")

    return extract_path


def load_nasa_dataset(root: str | os.PathLike | None = None) -> pd.DataFrame:
    """Load NASA battery data into a ``pandas.DataFrame``.

    The function ensures the dataset is present locally by calling
    :func:`download_nasa_dataset` if necessary.

    Parameters
    ----------
    root : Path-like, optional
        Directory containing the extracted dataset. If *None*, the default
        cache directory is used.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame containing all cells.
    """
    data_dir = download_nasa_dataset(root)

    dfs: List[pd.DataFrame] = []
    for cell_dir in sorted(data_dir.glob("B*")):
        mat_file = cell_dir / "matlab" / f"{cell_dir.name}.mat"
        if not mat_file.exists():
            continue
        dfs.append(_parse_single_cell(mat_file))

    if not dfs:
        raise RuntimeError("No .mat files found in NASA dataset directory.")

    return pd.concat(dfs, ignore_index=True)

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _default_cache_dir() -> Path:
    return Path(os.environ.get("NASA_BATTERY_DATA", Path.home() / ".cache" / "nasa_battery"))


def _safe_extract(tar: tarfile.TarFile, *, path: os.PathLike | str) -> None:
    """Safely extract tar files without allowing path traversal."""
    dest = Path(path).resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest)):
            raise RuntimeError("Path traversal detected during tar extract")
    tar.extractall(dest)


def _parse_single_cell(mat_path: Path) -> pd.DataFrame:
    """Parse a single ``.mat`` file for one cell into a tidy DataFrame."""
    mat = sio.loadmat(mat_path.as_posix(), squeeze_me=True, struct_as_record=False)

    battery_struct = mat["B"]
    battery_id: str = Path(mat_path).stem  # e.g. B0005

    records: list[dict[str, float | int | None]] = []

    for cycle in battery_struct.cycle:
        if not hasattr(cycle, "data"):
            continue  # Skip malformed cycles

        capacity = getattr(cycle, "Qd", None)
        if capacity is None:
            continue  # Skip cycles without discharge capacity

        data = cycle.data
        discharge_time = data.t[-1] if hasattr(data, "t") else None
        voltage_mean = data.v.mean() if hasattr(data, "v") else None
        current_mean = data.i.mean() if hasattr(data, "i") else None
        temperature = data.temp.mean() if hasattr(data, "temp") else None

        records.append(
            {
                "battery_id": battery_id,
                "cycle": int(cycle.type),
                "capacity": float(capacity),
                "discharge_time": float(discharge_time) if discharge_time is not None else None,
                "voltage_mean": float(voltage_mean) if voltage_mean is not None else None,
                "current_mean": float(current_mean) if current_mean is not None else None,
                "temperature": float(temperature) if temperature is not None else None,
            }
        )

    return pd.DataFrame.from_records(records) 