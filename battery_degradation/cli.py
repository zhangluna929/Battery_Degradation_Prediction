"""命令行接口：基于 YAML 配置运行完整流水线。

示例命令：

    python -m battery_degradation.cli --config configs/exp_rf.yaml
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore

try:
    import mlflow  # type: ignore
except ImportError:  # pragma: no cover
    mlflow = None

from .data.load import load_battery_data
from .data.preprocess import preprocess_data, feature_engineering
from .models import baselines, deep, empirical
from .models.hybrid import HybridModel
from .models.stacking import StackingModel
from .models.uncertainty import predict_with_uncertainty
from .evaluation.metrics import evaluate_regressor, rmse_at_n, calibration_error


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_builder(model_name: str):
    """
    从模块返回模型构建器(类或函数)。
    支持 'module.submodule.ClassName' 格式的动态导入。
    """
    if "." in model_name:
        try:
            module_path, class_name = model_name.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"无法从 '{model_name}' 动态导入模型: {e}")

    if hasattr(baselines, model_name):
        return getattr(baselines, model_name)
    if hasattr(deep, model_name):
        # 兼容旧的 baselines.py 中的 build_lstm
        if model_name == "build_lstm" and hasattr(baselines, model_name):
             return getattr(baselines, model_name)
        return getattr(deep, model_name)
    if hasattr(empirical, model_name):
        return getattr(empirical, model_name)
    raise ValueError(f"未知模型 '{model_name}'.")


def main():
    parser = argparse.ArgumentParser(description="电池退化预测流水线")
    parser.add_argument("--config", type=str, required=True, help="指向 YAML 配置文件")
    parser.add_argument("--mc-dropout", action="store_true", help="启用 MC Dropout 进行不确定性评估")
    parser.add_argument("--mc-samples", type=int, default=100, help="MC Dropout 采样次数")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_type = cfg.get("model_type", "standard")
    is_hybrid_model = model_type == "hybrid"
    is_stacking_model = model_type == "stacking"
    is_deep_model = not (is_hybrid_model or is_stacking_model) and cfg["model_fn"].startswith("build_")

    # ------------------------------------------------------------------
    # 数据加载 / 预处理
    # ------------------------------------------------------------------
    df = load_battery_data(cfg["data_path"])
    df = preprocess_data(df)
    df = feature_engineering(df)

    target_col = cfg.get("target_column", "capacity")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ------------------------------------------------------------------
    # 数据分割
    # ------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("test_size", 0.2), random_state=42
    )
    
    # 转换为 numpy 数组以兼容所有模型
    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    # ------------------------------------------------------------------
    # 模型训练
    # ------------------------------------------------------------------
    if mlflow is not None:
        mlflow.set_experiment(cfg.get("experiment", "battery_degradation"))
        mlflow.start_run(run_name=cfg.get("run_name", "cli_run"))
        mlflow.log_params(cfg)

    if is_stacking_model:
        print("--- 运行 Stacking 集成模型 ---")
        stack_cfg = cfg["stacking_params"]
        
        base_learners = []
        for name, params in stack_cfg["base_learners"].items():
            builder = get_model_builder(name)
            base_learners.append(builder(**params))

        meta_builder = get_model_builder(stack_cfg["meta_learner"]["name"])
        meta_learner = meta_builder(**stack_cfg["meta_learner"].get("params", {}))

        model = StackingModel(base_learners=base_learners, meta_learner=meta_learner)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif is_hybrid_model:
        print("--- 运行混合模型 ---")
        hybrid_cfg = cfg["hybrid_params"]
        
        emp_model_builder = get_model_builder(hybrid_cfg["empirical_model_fn"])
        res_learner_builder = get_model_builder(hybrid_cfg["residual_learner_fn"])
        
        model = HybridModel(
            empirical_model_func=emp_model_builder,
            residual_learner_builder=res_learner_builder,
            residual_learner_params=hybrid_cfg.get("residual_learner_params", {})
        )
        model.fit(X_train, y_train, **hybrid_cfg.get("fit_params", {}))
        y_pred = model.predict(X_test)

    elif is_deep_model:
        # Keras/TensorFlow 模型
        input_shape = (X_train.shape[1], 1) if len(X_train.shape) == 2 else (X_train.shape[1], X_train.shape[2])
        model = get_model_builder(cfg["model_fn"])(input_shape=input_shape, **cfg.get("model_params", {}))
        
        train_params = cfg.get("train_params", {})
        model.fit(X_train, y_train, validation_data=(X_test, y_test), **train_params)

        if args.mc_dropout:
            print(f"执行 MC Dropout (N={args.mc_samples})...")
            y_pred_mean, y_pred_std = predict_with_uncertainty(model, X_test, n_samples=args.mc_samples)
            y_pred = y_pred_mean
            cal_err = calibration_error(y_test, y_pred_mean, y_pred_std)
            print(f"校准误差 (ECE): {cal_err:.4f}")
            if mlflow:
                mlflow.log_metric("calibration_error", cal_err)
        else:
            y_pred = model.predict(X_test).flatten()

    else:
        # Scikit-learn 模型
        model_builder = get_model_builder(cfg["model_fn"])
        model_params = cfg.get("model_params", {})
        model = model_builder(X_train, y_train, **model_params)
        y_pred = model.predict(X_test)

    # ------------------------------------------------------------------
    # 评估与日志
    # ------------------------------------------------------------------
    rmse, mae, r2 = evaluate_regressor(y_test, y_pred)
    rmse50 = rmse_at_n(y_test, y_pred, n=50)

    print(f"\n评估结果:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")
    print(f"  RMSE@50: {rmse50:.4f}")

    if mlflow:
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2, "rmse@50": rmse50})
        mlflow.end_run()


if __name__ == "__main__":
    main() 