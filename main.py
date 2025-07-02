# battery_degradation_model

#KFold交叉验证(fighting!)
from sklearn.model_selection import KFold


def kfold_cross_validation(model, X, y, n_splits=5):
    """
    使用KFold交叉验证来评估模型性能。

    :param model: 训练好的回归模型
    :param X: 特征
    :param y: 目标变量（容量或内阻）
    :param n_splits: 交叉验证的折数
    :return: 平均RMSE和R²
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

    avg_rmse = np.mean(rmse_scores)
    avg_r2 = np.mean(r2_scores)

    print(f"Average RMSE: {avg_rmse}")
    print(f"Average R²: {avg_r2}")

    return avg_rmse, avg_r2


# 使用 GridSearchCV 调整模型超参数
from sklearn.model_selection import GridSearchCV


def tune_model_with_gridsearch(model, X, y, param_grid):
    """
    使用GridSearchCV调整模型的超参数。

    :param model: 初始回归模型
    :param X: 特征
    :param y: 目标变量（容量或内阻）
    :param param_grid: 超参数搜索范围
    :return: 最优的模型
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")

    return best_model


# 增加噪声模拟，增强模型鲁棒性
def add_noise_to_data(data, noise_factor=0.01):
    """
    为数据添加噪声，模拟现实世界中的测量误差。

    :param data: 原始数据
    :param noise_factor: 噪声比例
    :return: 含噪声的数据
    """
    noisy_data = data + noise_factor * np.random.normal(size=data.shape)
    return noisy_data


# 预测电池不同退化阶段的性能
def predict_degradation_stages(model, X_test):
    """
    预测电池在不同退化阶段的性能。

    :param model: 训练好的回归模型
    :param X_test: 测试数据
    :return: 不同退化阶段的预测结果
    """
    predictions = model.predict(X_test)

    # 假设退化分为几个阶段：初期、中期和末期
    stages = ['Early Stage', 'Mid Stage', 'Late Stage']
    capacity_thresholds = [0.8, 0.5, 0.2]  # 设定不同阶段的容量阈值

    degradation_stage = []
    for pred in predictions:
        if pred >= capacity_thresholds[0]:
            degradation_stage.append(stages[0])
        elif pred >= capacity_thresholds[1]:
            degradation_stage.append(stages[1])
        else:
            degradation_stage.append(stages[2])

    return degradation_stage


# 预测电池性能在不同温度下的变化
def predict_temperature_effect_on_degradation(model, X_test, temperature_variation):
    """
    预测电池在不同温度下的退化性能。

    :param model: 训练好的回归模型
    :param X_test: 测试数据
    :param temperature_variation: 温度变化范围
    :return: 温度变化下的退化预测
    """
    predictions = []

    for temp in temperature_variation:
        X_test['temperature'] = temp  # 假设温度是测试数据的一部分
        prediction = model.predict(X_test)
        predictions.append(prediction)

    # 可视化温度与退化的关系
    plt.plot(temperature_variation, predictions, label='Capacity Degradation', color='orange')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Predicted Capacity')
    plt.title('Battery Degradation under Different Temperature Conditions')
    plt.legend()
    plt.show()

    return predictions


# 更新主函数，包含新的功能和优化
def main():
    # 加载数据
    data = load_battery_data('battery_degradation_data.csv')

    # 数据预处理和特征工程
    processed_data = preprocess_data(data)
    engineered_data = feature_engineering(processed_data)

    # 准备特征和目标变量
    X = engineered_data.drop(columns=['capacity', 'internal_resistance'])
    y_capacity = engineered_data['capacity']
    y_resistance = engineered_data['internal_resistance']

    # 训练集成模型（Voting Regressor）
    print("Training ensemble model (Voting Regressor)...")
    ensemble_model = train_ensemble_model(X, y_capacity)

    # 评估集成模型
    print("Evaluating ensemble model...")
    evaluate_ensemble_model(ensemble_model, X, y_capacity)

    # 交叉验证集成模型
    print("KFold cross-validation...")
    kfold_cross_validation(ensemble_model, X, y_capacity)

    # 超参数调优
    param_grid = {'n_estimators': [100, 150], 'max_depth': [5, 10, 15]}
    print("Tuning RandomForest model with GridSearchCV...")
    best_rf_model = tune_model_with_gridsearch(RandomForestRegressor(), X, y_capacity, param_grid)

    # 训练LSTM模型（用于容量预测）
    X_train, X_test, y_train_capacity, y_test_capacity = train_test_split(X, y_capacity, test_size=0.2, random_state=42)
    X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))
    print("Training improved LSTM capacity model with dropout...")
    lstm_model = train_lstm_model_with_dropout(X_train_lstm, y_train_capacity, X_test_lstm, y_test_capacity)

    # 训练XGBoost模型（用于内阻预测）
    X_train_xgb, X_test_xgb, y_train_resistance, y_test_resistance = train_test_split(X, y_resistance, test_size=0.2,
                                                                                      random_state=42)
    print("Training XGBoost internal resistance model...")
    xgb_model = train_xgboost_model(X_train_xgb, y_train_resistance)

    # 评估模型性能
    print("Evaluating Random Forest models...")
    evaluate_model(best_rf_model, X_test, y_test_capacity)
    evaluate_model(rf_model_resistance, X_test, y_test_resistance)

    print("Evaluating LSTM model...")
    evaluate_lstm_model(lstm_model, X_test_lstm, y_test_capacity)

    print("Evaluating XGBoost model...")
    evaluate_xgboost_model(xgb_model, X_test_xgb, y_test_resistance)

    # 绘制退化过程图
    plot_degradation_details(data, 'capacity', 'Capacity Degradation Over Time')
    plot_degradation_details(data, 'internal_resistance', 'Internal Resistance Change Over Time')

    # 绘制容量和内阻的退化曲线
    plot_capacity_and_resistance_degradation(data)

    # 特征重要性分析
    feature_importance_analysis(rf_model_capacity, X.columns)

    # 预测剩余使用寿命（RUL）
    print("Predicting remaining useful life (RUL)...")
    predict_remaining_useful_life(lstm_model, X_test_lstm)

    # 动态预测电池在不同充电电压条件下的退化
    voltage_variation = np.linspace(3.0, 4.2, 10)  # 电压从3.0V到4.2V变化
    print("Predicting degradation under varying voltage conditions...")
    predict_degradation_under_varying_conditions(ensemble_model, X_test, voltage_variation)

    # 模拟不同温度下的电池退化
    temperature_variation = np.linspace(20, 50, 10)  # 温度从20°C到50°C变化
    print("Predicting degradation under varying temperature conditions...")
    predict_temperature_effect_on_degradation(ensemble_model, X_test, temperature_variation)


if __name__ == "__main__":
    main()
