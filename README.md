# Battery Degradation Prediction: A Data-Driven Approach

> Author: lunazhang

本项目致力于通过数据驱动的方法，对锂离子电池（特别是面向下一代固态电池）的健康状态（SOH）与剩余使用寿命（RUL）进行精确预测。电池的退化是一个高度复杂的电化学过程，其非线性行为受到工作温度、倍率、电压窗口等多种因素的耦合影响。传统模型难以捕捉这些复杂的时变依赖关系，而本项目旨在构建一个综合性的机器学习框架来解决这一挑战。

This project is dedicated to the accurate prediction of State of Health (SOH) and Remaining Useful Life (RUL) for lithium-ion batteries, with a particular focus on next-generation solid-state battery applications, using a data-driven methodology. Battery degradation is a highly complex electrochemical process characterized by non-linear behavior influenced by the coupling of multiple factors, including operating temperature, C-rate, and voltage window. Traditional models struggle to capture these intricate time-varying dependencies; this project aims to establish a comprehensive machine learning framework to address this challenge.

为实现高精度的预测，本研究框架集成了一系列先进的建模与分析技术。其核心优势在于异构模型的集成、深度时序分析、精细化的特征工程以及严格的模型验证流程。此外，项目整合了 MLflow 与 Streamlit，实现了从实验管理到结果可视化的全流程覆盖，确保了研究的可复现性与分析的深度。

To achieve high-precision predictions, this research framework integrates a suite of advanced modeling and analysis techniques. Its core strength lies in the integration of heterogeneous models, deep temporal analysis, sophisticated feature engineering, and a rigorous model validation pipeline. Furthermore, the project incorporates MLflow and Streamlit to provide end-to-end coverage from experiment management to results visualization, ensuring research reproducibility and in-depth analysis.

## 核心技术与特性 (Core Technologies & Features)

*   **异构模型集成 (Heterogeneous Model Ensemble):** 本框架并非依赖单一模型，而是融合了多种算法的优势。采用随机森林（Random Forest）捕捉特征间的非线性关系，XGBoost 以其梯度提升机制保证高预测精度与鲁棒性，而长短期记忆网络（LSTM）则专注于挖掘电池退化数据中固有的长程时间序列依赖性。最终通过投票回归器（Voting Regressor）对多个模型进行集成，以降低单一模型的方差，提升预测的稳定性与泛化能力。

*   **Heterogeneous Model Ensemble:** Rather than relying on a single model, this framework leverages the strengths of multiple algorithms. It employs Random Forest to capture non-linear interactions among features, XGBoost for its high predictive accuracy and robustness via its gradient boosting mechanism, and a Long Short-Term Memory (LSTM) network specifically to mine the inherent long-range temporal dependencies in battery degradation data. A Voting Regressor is used to ensemble these models, reducing the variance of individual models and enhancing the stability and generalization of the final predictions.

*   **高级特征工程 (Advanced Feature Engineering):** 从原始的时序数据中提取了多维度特征，包括基于统计学的滚动窗口特征（如移动平均容量）以捕捉短期退化趋势，以及归一化处理后的健康状态因子（如SOH代理特征）。未来的工作将进一步集成基于微分容量分析（dQ/dV）的特征，以更精确地表征内部电化学状态的演变。

*   **Advanced Feature Engineering:** Multi-dimensional features are extracted from the raw time-series data. These include statistical rolling-window features (e.g., moving average of capacity) to capture short-term degradation trends, and normalized health factors (e.g., SOH proxy features). Future work will further incorporate features derived from differential capacity analysis (dQ/dV) to more accurately represent the evolution of internal electrochemical states.

*   **可复现的实验管理 (Reproducible Experiment Management):** 引入 MLflow 对整个实验流程进行系统化管理。所有模型的训练参数、性能指标（RMSE, MAE, R²）以及生成的模型文件都被自动记录与版本控制。这不仅保证了研究结果的可复现性，也为模型优化与迭代提供了坚实的数据基础。

*   **Reproducible Experiment Management:** The project incorporates MLflow for systematic management of the entire experimental workflow. All training parameters, performance metrics (RMSE, MAE, R²), and generated model artifacts are automatically logged and version-controlled. This not only ensures the reproducibility of research findings but also provides a solid data foundation for model optimization and iteration.

*   **交互式分析仪表板 (Interactive Analysis Dashboard):** 基于 Streamlit 和 Altair 构建了交互式可视化仪表板。用户可以动态探索不同模型在测试集上的预测结果，对比分析退化曲线，并查看特征重要性排序。这极大地提升了模型结果的可解释性与分析效率。

*   **Interactive Analysis Dashboard:** An interactive visualization dashboard is built using Streamlit and Altair. It allows users to dynamically explore the prediction results of different models on the test set, conduct comparative analysis of degradation curves, and review feature importance rankings. This significantly enhances the interpretability of model results and the efficiency of analysis.

## 项目架构 (Project Structure)

```
Battery_Degradation_Prediction/
├── battery_degradation/
│   ├── __init__.py
│   ├── cli.py
│   ├── optimize.py
│   ├── pipeline.py
│   ├── robustness.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── calce.py
│   │   ├── load.py
│   │   ├── nasa.py
│   │   └── preprocess.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── electrochem.py
│   └── models/
│       ├── __init__.py
│       ├── baselines.py
│       ├── deep.py
│       ├── empirical.py
│       ├── hybrid.py
│       ├── stacking.py
│       └── uncertainty.py
├── configs/
│   ├── exp_gru_uncertainty.yaml
│   ├── exp_hybrid.yaml
│   ├── exp_stacking.yaml
│   └── optuna_rf.yaml
├── tests/
│   └── test_electrochem.py
├── main.py
├── streamlit_dashboard.py
├── LICENSE
├── README.md
└── requirements.txt
```

## 安装 (Installation)

1.  克隆本仓库到本地
    ```bash
    git clone https://github.com/lunazhang/Battery_Degradation_Prediction-main.git
    cd Battery_Degradation_Prediction-main
    ```

2.  建议使用虚拟环境（例如 venv 或 conda）
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  安装项目依赖
    ```bash
    pip install -r requirements.txt
    ```

## 使用说明 (Usage)

*   **执行基线模型训练与评估**

    通过命令行运行 `main.py` 脚本，可以启动完整的训练与评估流程。
    ```bash
    # 使用默认数据（data/battery_degradation_data.csv）并预测 capacity
    python main.py

    # 指定数据路径与预测目标 (例如 internal_resistance)
    python main.py --data-path path/to/your/data.csv --target internal_resistance
    ```

*   **启动交互式仪表板** (此功能待实现, `app.py` 为占位符)
    ```bash
    streamlit run app.py
    ```

