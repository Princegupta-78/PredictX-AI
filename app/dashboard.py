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

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
HEALTH_URL = f"{API_URL.rstrip('/')}/"

if "backend_awake" not in st.session_state:
    st.session_state.backend_awake = False

def check_backend():
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False

if not st.session_state.backend_awake:
    st.session_state.backend_awake = check_backend()

if st.session_state.backend_awake:
    st.sidebar.success("🟢 Backend API Connected")
else:
    st.sidebar.warning(
        "🟡 Backend is waking up...\n\n"
        "This app runs on free-tier hosting, which spins down the ML backend "
        "(XGBoost + SHAP) after inactivity. First load can take **30-60 seconds** "
        "to wake up — click below to check again."
    )
    if st.sidebar.button("🔄 Check again"):
        st.session_state.backend_awake = check_backend()
        st.rerun()


st.sidebar.markdown("---")
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Engine History", type=["csv", "txt"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write("**Model:** XGBoost")
st.sidebar.write("**Dataset:** NASA CMAPSS FD001")
st.sidebar.write("**Explainability:** SHAP")

# ==========================================
# MAIN DASHBOARD LOGIC
# ==========================================
if uploaded_file is not None:
    # 1. Load Data
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    
    # Assign column names (assuming raw CMAPSS format)
    if len(df.columns) == 26:
        columns = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
        df.columns = columns
    else:
        st.error("⚠️ Invalid file format. Please upload a valid NASA CMAPSS dataset (26 columns).")
        st.stop() # Halts execution gracefully
        
    # 2. Select Engine
    engine_ids = sorted(df["engine_id"].unique())
    selected_engine = st.sidebar.selectbox("Select Engine to Analyze", engine_ids)
    engine_df = df[df["engine_id"] == selected_engine]
    
    st.sidebar.info(f"Engine {selected_engine} has {len(engine_df)} cycles of history.")

    # 3. Predict Button
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
else:
    # Splash Screen when no data is uploaded
    st.info("👈 Upload a raw CMAPSS `test_FD001.txt` CSV file in the sidebar to begin.")

st.markdown("---")
st.caption("PredictX AI • Built using Python • XGBoost • SHAP • FastAPI • Streamlit")