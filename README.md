# Battery Degradation Prediction

This repository contains code for predicting and analyzing battery degradation using data-driven models. The project aims to predict battery capacity and internal resistance degradation over time using machine learning techniques, including Random Forest, XGBoost, LSTM, and an ensemble model.

## Features

- **Battery Degradation Models**: The models include regression techniques like Random Forest, XGBoost, LSTM, and a Voting Regressor ensemble approach.
- **Data Preprocessing**: Includes normalization, handling missing values, and feature engineering.
- **Model Evaluation**: The models are evaluated using RMSE, R², and cross-validation techniques to assess their performance.
- **Degradation Stages Prediction**: The battery degradation process is categorized into different stages based on capacity thresholds.
- **Temperature and Voltage Effects**: Predicts how battery degradation changes under varying temperature and voltage conditions.
- **Noise Simulation**: Simulates real-world measurement noise to test model robustness.

