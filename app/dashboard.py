import streamlit as st
import pandas as pd
import json
import plotly.express as px
import requests
import time
import os

from api import predict_engine
from utils import create_health_gauge, create_shap_bar_chart

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PredictX AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✈️ PredictX AI: Engine Maintenance Dashboard")
st.caption("Powered by XGBoost + SHAP Explainability + FastAPI")
st.markdown("---")

# ==========================================
# SIDEBAR & SYSTEM STATUS
# ==========================================
st.sidebar.header("System Status")

# They share localhost inside the unified container!
API_URL = "http://127.0.0.1:8000"

st.sidebar.success("🟢 System Online & Ready")
st.sidebar.caption("API and UI running securely in unified container.")

st.sidebar.markdown("---")
st.sidebar.header("Data Input")

# 1. 1-Click Demo Button for Interviewers
use_demo = st.sidebar.button("🚀 Load Demo Data (For Testing)", use_container_width=True)

# 2. Standard Upload Option
uploaded_file = st.sidebar.file_uploader("Or Upload Engine History", type=["csv", "txt"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write("**Model:** XGBoost")
st.sidebar.write("**Dataset:** NASA CMAPSS FD001")
st.sidebar.write("**Explainability:** SHAP")

# ==========================================
# MAIN DASHBOARD LOGIC
# ==========================================
df = None

# 1. Determine which data to load
if use_demo:
    try:
        df = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None)
        st.sidebar.success("Demo dataset loaded successfully!")
    except FileNotFoundError:
        st.sidebar.error("Demo file not found. Please ensure 'data/test_FD001.txt' exists.")
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)

# 2. If data is loaded (either demo or upload), show the dashboard
if df is not None:
    # Assign column names (assuming raw CMAPSS format)
    if len(df.columns) == 26:
        columns = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
        df.columns = columns
    else:
        st.error("⚠️ Invalid file format. Please upload a valid NASA CMAPSS dataset (26 columns).")
        st.stop() # Halts execution gracefully
        
    # Select Engine
    engine_ids = sorted(df["engine_id"].unique())
    selected_engine = st.sidebar.selectbox("Select Engine to Analyze", engine_ids)
    engine_df = df[df["engine_id"] == selected_engine]
    
    st.sidebar.info(f"Engine {selected_engine} has {len(engine_df)} cycles of history.")

    # Predict Button
    if st.sidebar.button("Predict Remaining Useful Life", use_container_width=True):
        with st.spinner("Running prediction pipeline..."):
            try:
                # Call the FastAPI backend
                result = predict_engine(engine_df)
                
                # ==========================================
                # TOP METRICS
                # ==========================================
                c1, c2, c3 = st.columns(3)
                c1.metric("Remaining Useful Life", f"{round(result['predicted_rul'])} Cycles")
                c2.metric("Engine Health", f"{result['health_score']}/100")
                c3.metric("Risk Level", result['risk_level'])
                
                # Recommendation Banner
                if result["risk_level"] == "High":
                    st.error(f"🚨 **Action Required:** {result['recommendation']}")
                elif result["risk_level"] == "Medium":
                    st.warning(f"⚠️ **Warning:** {result['recommendation']}")
                else:
                    st.success(f"✅ **Healthy:** {result['recommendation']}")
                
                st.markdown("---")
                
                # ==========================================
                # VISUALIZATIONS
                # ==========================================
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Engine Health Score")
                    st.plotly_chart(create_health_gauge(result["health_score"]), use_container_width=True)
                    
                    # Download Button
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=json.dumps(result, indent=4),
                        file_name=f"engine_{selected_engine}_report.json",
                        mime="application/json",
                        use_container_width=True
                    )

                with col2:
                    exp_df = pd.DataFrame(result["explanation"]["top_features"])
                    st.plotly_chart(create_shap_bar_chart(exp_df), use_container_width=True)
                    
                    st.dataframe(
                        exp_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.subheader("Model Explanation")
                    top = result["explanation"]["top_features"]
                    if len(top) >= 3:
                        st.info(
                            f"""
                            The prediction is mainly influenced by **{top[0]['Feature']}**, 
                            **{top[1]['Feature']}**, and **{top[2]['Feature']}**.
                            
                            These sensors contributed the most toward the predicted Remaining Useful Life.
                            """
                        )
                    
                st.markdown("---")
                
                # ==========================================
                # HISTORICAL SENSOR TREND
                # ==========================================
                st.subheader("Historical Sensor Analysis")
                DROP_SENSORS = [
                    "sensor_1", "sensor_5", "sensor_6", 
                    "sensor_10", "sensor_16", "sensor_18", "sensor_19"
                ]
                sensor_cols = [
                    c for c in engine_df.columns 
                    if c.startswith("sensor_") and c not in DROP_SENSORS
                ]
                selected_sensor = st.selectbox("Select Sensor to Plot", sensor_cols)
                
                trend_fig = px.line(
                    engine_df, 
                    x="cycle", 
                    y=selected_sensor, 
                    markers=True,
                    title=f"{selected_sensor} Trend Across Engine Cycles"
                )
                trend_fig.update_layout(
                    xaxis_title="Cycle",
                    yaxis_title=selected_sensor,
                    template="plotly_white"
                )
                st.plotly_chart(trend_fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

# 3. If NO data is loaded, show the professional splash screen
else:
    # ==========================================
    # PROFESSIONAL SPLASH SCREEN FOR INTERVIEWERS
    # ==========================================
    st.markdown("### ✈️ Welcome to the PredictX AI Dashboard")
    st.write("This application predicts the **Remaining Useful Life (RUL)** of turbofan engines to prevent catastrophic failures and optimize maintenance schedules.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🛠️ System Architecture")
        st.markdown("""
        * **Deployment:** Monolithic Docker container hosted on Render.
        * **Backend:** FastAPI REST API handling model inference.
        * **Frontend:** Streamlit interactive UI.
        * **Network:** Zero-latency internal communication via `localhost`.
        """)
        
    with col2:
        st.markdown("#### 🧠 Machine Learning")
        st.markdown("""
        * **Model:** XGBoost Regressor trained on NASA CMAPSS sensor data.
        * **Explainability:** Real-time SHAP (SHapley Additive exPlanations) values to identify failing components.
        * **Risk Assessment:** Automated categorization into Healthy, Medium, or High Risk based on engine cycles.
        """)

    st.markdown("---")
    st.info("👈 **To test the application:** Click the **'🚀 Load Demo Data'** button in the sidebar to instantly load a test dataset, or upload your own raw CMAPSS file.")

st.markdown("---")
st.caption("PredictX AI • Built using Python • XGBoost • SHAP • FastAPI • Streamlit")