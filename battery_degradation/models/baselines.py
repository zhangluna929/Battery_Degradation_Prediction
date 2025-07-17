from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Any, Dict

try:
    from xgboost import XGBRegressor  # type: ignore
except ImportError:
    XGBRegressor = None  # pragma: no cover

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models  # type: ignore
except ImportError:
    tf = None  # type: ignore
    layers = models = None


def train_random_forest(X, y, **rf_params: Any) -> RandomForestRegressor:
    """Train a Random Forest regressor on provided data."""
    model = RandomForestRegressor(**rf_params)
    model.fit(X, y)
    return model


def train_xgboost(X, y, **xgb_params: Any):
    """Train an XGBoost regressor if XGBoost is available."""
    if XGBRegressor is None:
        raise ImportError("XGBoost is not installed. Install it with `pip install xgboost`. ")
    model = XGBRegressor(**xgb_params)
    model.fit(X, y)
    return model


def build_lstm(input_shape: tuple[int, int], units: int = 64, dropout: float = 0.2):
    """Build a simple LSTM network for sequence regression."""
    if tf is None:
        raise ImportError("TensorFlow is not installed. Install it with `pip install tensorflow`. ")
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, dropout=dropout),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm(X_train, y_train, X_val, y_val, epochs: int = 10, batch_size: int = 32, **kwargs):
    """Train LSTM model and return the trained model."""
    model = build_lstm((X_train.shape[1], X_train.shape[2]), **kwargs)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return model


def train_ensemble_model(X, y):
    """Placeholder for ensemble (e.g., Voting Regressor)."""
    from sklearn.ensemble import VotingRegressor

    # Simple ensemble of RF and, if available, XGB.
    estimators: list[tuple[str, Any]] = [("rf", RandomForestRegressor(n_estimators=200))]
    if XGBRegressor is not None:
        estimators.append(("xgb", XGBRegressor()))
    model = VotingRegressor(estimators)
    model.fit(X, y)
    return model 