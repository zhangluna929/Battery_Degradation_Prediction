"""电池退化的经验/半物理模型。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit  # type: ignore

__all__ = ["double_exponential_model", "fit_empirical_model"]


def double_exponential_model(
    cycle: np.ndarray, a: float, b: float, c: float, d: float
) -> np.ndarray:
    """
    双指数模型，常用于描述锂离子电池容量衰减。

    Q(n) = a * exp(b * n) + c * exp(d * n)

    Parameters
    ----------
    cycle : np.ndarray
        循环次数数组。
    a, b, c, d : float
        模型参数。

    Returns
    -------
    np.ndarray
        预测的容量。
    """
    return a * np.exp(b * cycle) + c * np.exp(d * cycle)


def fit_empirical_model(
    X: np.ndarray | list[int],
    y: np.ndarray,
    model_func=double_exponential_model,
) -> tuple[np.ndarray, callable]:
    """
    拟合经验模型到数据。

    Parameters
    ----------
    X : np.ndarray
        输入特征，这里通常是循环次数。
    y : np.ndarray
        目标值，即容量。
    model_func : callable
        要拟合的经验模型函数。

    Returns
    -------
    tuple[np.ndarray, callable]
        一个元组，包含:
        - `params`: 拟合得到的模型最优参数。
        - `predict_func`: 一个使用最优参数进行预测的函数。
    """
    # 确保 X 是一个扁平的 cycle number 数组
    if isinstance(X, np.ndarray) and X.ndim > 1:
        cycles = X[:, 0]  # 假设 cycle 在第一列
    else:
        cycles = np.array(X)

    # `curve_fit` 对初始参数敏感，提供一个合理的猜测
    initial_params = (y[0] * 0.8, -1e-3, y[0] * 0.2, -1e-4)

    try:
        params, _ = curve_fit(
            model_func,
            cycles,
            y,
            p0=initial_params,
            maxfev=10000,
        )
    except RuntimeError:
        # 如果拟合失败，返回一个默认的线性衰减
        params = np.array([y[0], -1e-5, 0, 0])

    def predict_func(x_new: np.ndarray) -> np.ndarray:
        if x_new.ndim > 1:
            x_new_cycles = x_new[:, 0]
        else:
            x_new_cycles = x_new
        return model_func(x_new_cycles, *params)

    return params, predict_func 