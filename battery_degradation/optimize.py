"""使用 Optuna 进行超参数优化。

示例命令:
    python -m battery_degradation.optimize --config configs/optuna_rf.yaml --n-trials 100
"""
from __future__ import annotations

import argparse
from typing import Any, Dict

import optuna
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

from battery_degradation.cli import get_model_builder, load_config
from battery_degradation.data.load import load_battery_data
from battery_degradation.data.preprocess import preprocess_data
from battery_degradation.evaluation.metrics import evaluate_regressor


def suggest_params(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """根据搜索空间为 Optuna trial 建议参数。"""
    params = {}
    for name, definition in search_space.items():
        param_type = definition["type"]
        if param_type == "int":
            params[name] = trial.suggest_int(name, **definition["args"])
        elif param_type == "float":
            params[name] = trial.suggest_float(name, **definition["args"])
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, **definition["args"])
        else:
            raise ValueError(f"不支持的参数类型: {param_type}")
    return params


def objective(trial: optuna.Trial, cfg: Dict[str, Any]) -> float:
    """Optuna 的目标函数。"""
    # 1. 生成此 trial 的参数
    model_params = suggest_params(trial, cfg["search_space"])
    
    # 2. 加载和预处理数据
    df = load_battery_data(cfg["data_path"])
    df = preprocess_data(df)
    target_col = cfg.get("target_column", "capacity")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(
        np.array(X), np.array(y), test_size=0.25, random_state=42
    )

    # 3. 训练模型
    # 这里简化为只支持 Scikit-learn 模型
    builder = get_model_builder(cfg["model_name"])
    model = builder(**model_params)
    model.fit(X_train, y_train)

    # 4. 评估并返回指标
    y_pred = model.predict(X_val)
    rmse, _, _ = evaluate_regressor(y_val, y_pred)
    
    return rmse # Optuna 默认最小化目标


def main():
    parser = argparse.ArgumentParser(description="超参数优化脚本")
    parser.add_argument("--config", required=True, help="指向优化的 YAML 配置文件")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna 试验次数")
    args = parser.parse_args()

    cfg = load_config(args.config)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, cfg), n_trials=args.n_trials)

    print("\n优化完成!")
    print(f"最佳试验 Trial: {study.best_trial.number}")
    print(f"最佳分数 (RMSE): {study.best_value:.4f}")
    print("最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 