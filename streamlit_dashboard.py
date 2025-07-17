"""电池退化预测可视化看板 (Streamlit)。

在终端运行:
    streamlit run streamlit_dashboard.py
"""
import streamlit as st
import pandas as pd
import altair as alt

from battery_degradation.data.load import load_battery_data
from battery_degradation.data.preprocess import preprocess_data, feature_engineering
from battery_degradation.models.baselines import train_random_forest
from battery_degradation.evaluation.metrics import evaluate_regressor

st.title("🔋 Battery Degradation Dashboard")

uploaded_file = st.file_uploader("上传电池数据 CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)
    df = feature_engineering(df)

    st.subheader("数据预览")
    st.dataframe(df.head())

    target_col = st.selectbox("目标列", [c for c in df.columns if c != "cycle"], index=df.columns.get_loc("capacity") if "capacity" in df.columns else 0)

    if st.button("训练随机森林模型"):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        model = train_random_forest(X, y, n_estimators=200)
        y_pred = model.predict(X)
        rmse, mae, r2 = evaluate_regressor(y, y_pred)
        st.write(f"RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")

        chart_df = pd.DataFrame({"True": y, "Pred": y_pred, "cycle": df["cycle"] if "cycle" in df.columns else range(len(y))})
        chart = (
            alt.Chart(chart_df)
            .transform_fold(["True", "Pred"], as_["Type", "Capacity"])  # type: ignore
            .mark_line()
            .encode(x="cycle", y="Capacity", color="Type")
        )
        st.altair_chart(chart, use_container_width=True) 