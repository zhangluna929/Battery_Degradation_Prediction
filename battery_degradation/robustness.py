"""鲁棒性实验与异常检测工具。

包含:
1. add_noise: 为数值列添加高斯/泊松/均匀噪声
2. zscore_anomaly: 使用 z-score 进行简单异常检测
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["add_noise", "zscore_anomaly"]


def add_noise(df: pd.DataFrame, columns: list[str], noise_level: float = 0.01, kind: str = "gaussian") -> pd.DataFrame:
    """对指定列添加噪声，返回副本。"""
    df_noisy = df.copy()
    for col in columns:
        if col not in df_noisy.columns:
            continue
        if kind == "gaussian":
            noise = np.random.normal(scale=noise_level * df_noisy[col].std(), size=len(df_noisy))
        elif kind == "uniform":
            noise = np.random.uniform(-1, 1, size=len(df_noisy)) * noise_level * df_noisy[col].std()
        else:
            raise ValueError(f"Unknown noise kind: {kind}")
        df_noisy[col] += noise
    return df_noisy


def zscore_anomaly(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """返回布尔序列，True 表示异常点。"""
    z = (series - series.mean()) / series.std()
    return z.abs() > threshold 