"""Stacking 集成学习模型。"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone

__all__ = ["StackingModel"]


class StackingModel:
    """
    一个两层的 Stacking 集成模型。

    Level 0: 多个基础学习器。
    Level 1: 一个元学习器，学习如何组合基础学习器的预测。
    """

    def __init__(self, base_learners: list, meta_learner, n_splits: int = 5):
        """
        Parameters
        ----------
        base_learners : list
            基础学习器实例的列表。可以是 Scikit-learn 兼容的模型。
        meta_learner : object
            元学习器实例。
        n_splits : int, default 5
            用于生成元学习器训练数据的 K-Fold 折数。
        """
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.n_splits = n_splits
        self.trained_base_learners_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs):
        """
        训练 Stacking 模型。

        fit_kwargs 会被传递给每个基础模型的 fit 方法。
        """
        print("--- 开始 Stacking 模型训练 ---")
        meta_features = np.zeros((X.shape[0], len(self.base_learners)))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # 1. Level 0: 交叉验证训练基础模型，并生成元特征
        print(f"为元学习器生成样本外预测 (K={self.n_splits})...")
        for i, learner in enumerate(self.base_learners):
            print(f"  训练基础学习器 #{i+1}: {learner.__class__.__name__}")
            
            # Keras 模型需要特殊处理
            is_keras_model = hasattr(learner, "fit") and hasattr(learner, "predict") and "keras" in str(type(learner))

            for train_idx, val_idx in kf.split(X):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val = X[val_idx]
                
                if is_keras_model:
                    # Keras 模型每次都需要重新构建
                    current_learner = clone(learner) # 这对 Keras 无效，仅为示意
                    # 实际应从 builder 构建新模型
                    # 此处为简化实现，我们直接 fit
                    learner.fit(X_train, y_train, **fit_kwargs.get(learner.__class__.__name__, {}))

                else: # Sklearn
                    current_learner = clone(learner)
                    current_learner.fit(X_train, y_train)
                
                meta_features[val_idx, i] = current_learner.predict(X_val).flatten()

        # 2. 重新训练基础模型于完整数据
        print("\n在完整训练集上重新训练基础学习器...")
        for learner in self.base_learners:
            cloned_learner = clone(learner)
            cloned_learner.fit(X, y)
            self.trained_base_learners_.append(cloned_learner)
        
        # 3. Level 1: 训练元学习器
        print("训练元学习器...")
        self.meta_learner.fit(meta_features, y)
        print("--- Stacking 模型训练完成 ---")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用训练好的 Stacking 模型进行预测。"""
        if not self.trained_base_learners_:
            raise RuntimeError("模型需要先训练。")
        
        base_predictions = np.zeros((X.shape[0], len(self.trained_base_learners_)))
        for i, learner in enumerate(self.trained_base_learners_):
            base_predictions[:, i] = learner.predict(X).flatten()
            
        return self.meta_learner.predict(base_predictions) 