"""混合模型：物理/经验模型 + 机器学习模型。"""
from __future__ import annotations

import numpy as np

from .empirical import fit_empirical_model
from .deep import build_gru  # 默认的残差学习器

__all__ = ["HybridModel"]


class HybridModel:
    """
    一个混合模型，结合了经验模型和深度学习模型。

    工作流程:
    1. 使用经验模型 (如双指数模型) 拟合数据的整体趋势。
    2. 计算真实值与趋势预测之间的残差。
    3. 训练一个深度学习模型来预测这些残差。
    4. 最终预测 = 经验模型预测 + 残差模型预测。
    """

    def __init__(self, empirical_model_func, residual_learner_builder=build_gru, residual_learner_params=None):
        self.empirical_model_func = empirical_model_func
        self.empirical_predict_func = None
        
        self.residual_learner_builder = residual_learner_builder
        self.residual_learner_params = residual_learner_params if residual_learner_params is not None else {}
        self.residual_learner = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **fit_kwargs):
        """
        训练混合模型。

        Parameters
        ----------
        X_train : np.ndarray
            训练输入特征。
        y_train : np.ndarray
            训练目标值。
        **fit_kwargs :
            传递给残差学习器 `fit` 方法的额外参数 (如 epochs, batch_size)。
        """
        # 1. 拟合经验模型
        print("Fitting empirical model to capture trend...")
        _, self.empirical_predict_func = fit_empirical_model(
            X_train, y_train, model_func=self.empirical_model_func
        )
        y_trend_train = self.empirical_predict_func(X_train)

        # 2. 计算残差
        y_residual_train = y_train - y_trend_train
        print(f"Residuals computed. Mean: {y_residual_train.mean():.4f}, Std: {y_residual_train.std():.4f}")

        # 3. 训练残差学习器
        print("Training residual learner...")
        input_shape = (X_train.shape[1], 1) if X_train.ndim == 2 else (X_train.shape[1], X_train.shape[2])
        self.residual_learner = self.residual_learner_builder(
            input_shape=input_shape, **self.residual_learner_params
        )
        self.residual_learner.fit(X_train, y_residual_train, **fit_kwargs)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用混合模型进行预测。

        Parameters
        ----------
        X : np.ndarray
            用于预测的输入数据。

        Returns
        -------
        np.ndarray
            最终的预测结果。
        """
        if self.empirical_predict_func is None or self.residual_learner is None:
            raise RuntimeError("Model must be fitted before prediction.")

        # 预测趋势
        y_trend_pred = self.empirical_predict_func(X)

        # 预测残差
        y_residual_pred = self.residual_learner.predict(X).flatten()

        # 合并结果
        return y_trend_pred + y_residual_pred 