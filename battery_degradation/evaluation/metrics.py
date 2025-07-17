from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from typing import Tuple


def evaluate_regressor(y_true, y_pred) -> Tuple[float, float, float]:
    """Return RMSE, MAE, and R2 for predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def kfold_cross_validation(model, X, y, n_splits: int = 5):
    """Perform KFold CV and return average RMSE and R2."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []
    r2_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse, _, r2 = evaluate_regressor(y_test, pred)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

    return float(np.mean(rmse_scores)), float(np.mean(r2_scores))


def rmse_at_n(y_true, y_pred, n: int = 50) -> float:
    """计算前 n 个样本的 RMSE，用于早期预测评估。"""
    import numpy as np

    y_t = np.array(y_true[:n])
    y_p = np.array(y_pred[:n])
    return float(np.sqrt(mean_squared_error(y_t, y_p)))


def early_prediction_accuracy(y_true, y_pred, n: int = 50) -> float:
    """早期阶段 (前 n 个样本) 的相对误差准确率 (1 - MAPE)。"""
    import numpy as np

    y_t = np.array(y_true[:n])
    y_p = np.array(y_pred[:n])
    mape = np.mean(np.abs((y_t - y_p) / y_t))
    return float(1 - mape)


def calibration_error(y_true, y_pred_mean, y_pred_std, n_bins: int = 10) -> float:
    """简易校准误差 (Expected Calibration Error，ECE)。

    假设预测为高斯分布 (均值/标准差)。统计分位覆盖率与实际覆盖率间差异。
    """
    import numpy as np

    probs = [0.6827, 0.9545]  # 1σ 和 2σ 区间
    ece = 0.0
    for p in probs:
        z = p  # 直接用覆盖率
        in_interval = (y_true >= y_pred_mean - z * y_pred_std) & (
            y_true <= y_pred_mean + z * y_pred_std
        )
        observed = np.mean(in_interval)
        ece += np.abs(observed - p)
    return float(ece / len(probs)) 