"""电化学曲线特征提取

本模块实现经典的电池退化特征：
1. dQ/dV (Incremental Capacity, IC) 曲线
2. dV/dQ (Differential Voltage Analysis, DVA) 曲线

并提供若干统计型特征（峰值、峰位、积分等），可直接合并到
机器学习输入表。

所有函数均假设输入为单次循环的 *升序* 电压 (V) 与对应容量 (Ah)
数值序列。如果提供多循环时，请在调用方进行循环拆分。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter  # type: ignore

__all__ = [
    "incremental_capacity",
    "differential_voltage",
    "ic_statistics",
    "dva_statistics",
]


def _smooth(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """Apply Savitzky–Golay smoothing to reduce noise."""
    if len(y) < window:
        window = len(y) // 2 * 2 + 1  # make window odd
    return savgol_filter(y, window_length=window, polyorder=poly)


def incremental_capacity(voltage: np.ndarray, capacity: np.ndarray, *, smooth: bool = True) -> np.ndarray:
    """Compute dQ/dV curve (Incremental Capacity).

    Parameters
    ----------
    voltage : np.ndarray
        Voltage measurements (V), must be strictly monotonic.
    capacity : np.ndarray
        Corresponding discharged capacity (Ah).
    smooth : bool, default True
        Whether to apply Savitzky–Golay smoothing before differentiation.

    Returns
    -------
    np.ndarray
        Array of dQ/dV values with the same length as input.
    """
    if smooth:
        capacity = _smooth(capacity)
        voltage = _smooth(voltage)
    dq = np.gradient(capacity)
    dv = np.gradient(voltage)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.divide(dq, dv, out=np.full_like(dq, np.nan), where=dv != 0)
    return ic


def differential_voltage(voltage: np.ndarray, capacity: np.ndarray, *, smooth: bool = True) -> np.ndarray:
    """Compute dV/dQ curve (Differential Voltage Analysis)."""
    if smooth:
        capacity = _smooth(capacity)
        voltage = _smooth(voltage)
    dq = np.gradient(capacity)
    dv = np.gradient(voltage)
    with np.errstate(divide="ignore", invalid="ignore"):
        dva = np.divide(dv, dq, out=np.full_like(dv, np.nan), where=dq != 0)
    return dva


def _statistics(arr: np.ndarray) -> dict[str, float]:
    """Return common peak statistics for an array."""
    if np.all(np.isnan(arr)):
        return {"peak": np.nan, "peak_index": -1, "mean": np.nan}
    peak_idx = int(np.nanargmax(np.abs(arr)))
    return {"peak": float(arr[peak_idx]), "peak_index": peak_idx, "mean": float(np.nanmean(arr))}


def ic_statistics(voltage: np.ndarray, capacity: np.ndarray) -> dict[str, float]:
    """Return summary statistics of IC curve."""
    ic = incremental_capacity(voltage, capacity)
    stats = _statistics(ic)
    # map peak index to voltage
    stats["peak_voltage"] = float(voltage[stats["peak_index"]]) if stats["peak_index"] >= 0 else np.nan
    return stats


def dva_statistics(voltage: np.ndarray, capacity: np.ndarray) -> dict[str, float]:
    """Return summary statistics of DVA curve."""
    dva = differential_voltage(voltage, capacity)
    stats = _statistics(dva)
    stats["peak_capacity"] = float(capacity[stats["peak_index"]]) if stats["peak_index"] >= 0 else np.nan
    return stats 