from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .data.load import load_battery_data
from .data.preprocess import preprocess_data, feature_engineering
from .models.baselines import train_random_forest, train_xgboost, train_ensemble_model
from .evaluation import evaluate_regressor, kfold_cross_validation


DEFAULT_DATA_PATH = Path("data/battery_degradation_data.csv")


def run_baseline(
    data_path: Path | str = DEFAULT_DATA_PATH,
    target_column: str = "capacity",
):
    """End-to-end baseline: load data, train models, and print metrics."""
    # 1. Load & preprocess
    df = load_battery_data(data_path)
    df = preprocess_data(df)
    df = feature_engineering(df)

    # 2. Feature/target split
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train models
    rf_model = train_random_forest(X_train, y_train, n_estimators=200)
    xgb_model = None
    try:
        xgb_model = train_xgboost(X_train, y_train, n_estimators=300)
    except ImportError:
        print("[WARN] XGBoost not installed; skipping XGB model.")

    ensemble_model = train_ensemble_model(X_train, y_train)

    # 5. Evaluate
    for name, model in ("RF", rf_model), ("Ensemble", ensemble_model):
        y_pred = model.predict(X_test)
        rmse, mae, r2 = evaluate_regressor(y_test, y_pred)
        print(f"{name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

        cv_rmse, cv_r2 = kfold_cross_validation(model, X, y)
        print(f"   CV: RMSE={cv_rmse:.3f}, R2={cv_r2:.3f}")

    if xgb_model is not None:
        y_pred = xgb_model.predict(X_test)
        rmse, mae, r2 = evaluate_regressor(y_test, y_pred)
        print(f"XGB: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")


if __name__ == "__main__":
    run_baseline() 