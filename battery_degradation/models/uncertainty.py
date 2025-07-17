"""使用蒙特卡洛 Dropout 进行不确定性量化。"""
from __future__ import annotations

import numpy as np
import tqdm

try:
    from tensorflow.keras import models  # type: ignore
except ImportError:  # pragma: no cover
    models = None  # type: ignore


__all__ = ["predict_with_uncertainty"]


def predict_with_uncertainty(
    model: models.Model,
    X: np.ndarray,
    n_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """通过多次启用 Dropout 的前向传播来评估模型不确定性。

    该方法是蒙特卡洛 Dropout (MC Dropout) 的实现。

    Parameters
    ----------
    model : keras.Model
        一个包含 Dropout 层的 Keras 模型。
    X : np.ndarray
        用于预测的输入数据。
    n_samples : int, default 100
        执行前向传播的次数。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        一个元组，包含:
        - `mean_predictions`: 所有采样预测的均值。
        - `std_predictions`: 预测的标准差，代表模型不确定性。
    """
    if models is None:
        raise ImportError("TensorFlow is required for uncertainty prediction.")

    predictions = []
    for _ in tqdm.tqdm(range(n_samples), desc="MC Dropout Samples"):
        # 关键: 在推理时强制启用 training=True
        pred = model(X, training=True)
        predictions.append(pred)

    predictions_arr = np.stack(predictions)
    mean_predictions = np.mean(predictions_arr, axis=0)
    std_predictions = np.std(predictions_arr, axis=0)

    # 结果的 shape 可能是 (num_examples, 1)，展平为 (num_examples,)
    return mean_predictions.flatten(), std_predictions.flatten() 