"""深度学习模型架构

该模块提供多种时序模型构建函数，依赖 TensorFlow/Keras。
若环境中缺少 TensorFlow，则调用相应函数会抛出 ImportError。
"""
from __future__ import annotations

from typing import Tuple

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models  # type: ignore
except ImportError:  # pragma: no cover
    tf = None  # type: ignore
    layers = models = None  # type: ignore

__all__ = [
    "build_gru",
    "build_tcn",
    "build_transformer",
]


def _check_tf():  # noqa: D401
    """Raise ImportError if TensorFlow is not available."""
    if tf is None:
        raise ImportError("TensorFlow is not installed. Install it with `pip install tensorflow`. ")


# -----------------------------------------------------------------------------
# GRU
# -----------------------------------------------------------------------------

def build_gru(input_shape: Tuple[int, int], units: int = 64, dropout: float = 0.2):
    """构建简单 GRU 回归网络。"""
    _check_tf()
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(units, dropout=dropout),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# -----------------------------------------------------------------------------
# Temporal Convolutional Network (简化版)
# -----------------------------------------------------------------------------

def build_tcn(input_shape: Tuple[int, int], filters: int = 64, kernel_size: int = 3, dilations: Tuple[int, ...] | None = None):
    """简易 TCN 实现，无依赖外部库。"""
    _check_tf()
    if dilations is None:
        dilations = (1, 2, 4, 8)

    inputs = layers.Input(shape=input_shape)
    x = inputs
    for d in dilations:
        x_prev = x
        x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=d, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        # Residual
        if x_prev.shape[-1] != x.shape[-1]:
            x_prev = layers.Conv1D(filters, 1, padding="same")(x_prev)
        x = layers.Add()([x, x_prev])
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# -----------------------------------------------------------------------------
# Transformer (简化版)
# -----------------------------------------------------------------------------

def _positional_encoding(length: int, d_model: int):
    import numpy as np

    pos = np.arange(length)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads, tf.float32)


def build_transformer(input_shape: Tuple[int, int], d_model: int = 64, num_heads: int = 4, ff_dim: int = 128, num_layers: int = 2):
    """简化版 Transformer 编码器 + MLP 头。"""
    _check_tf()
    seq_len, features = input_shape
    inputs = layers.Input(shape=input_shape)

    # 线性投影到 d_model
    x = layers.Dense(d_model)(inputs)
    x += _positional_encoding(seq_len, d_model)

    for _ in range(num_layers):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)
        # Feed-forward
        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dense(d_model)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model 