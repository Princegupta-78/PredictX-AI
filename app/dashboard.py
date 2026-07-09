import streamlit as st
import pandas as pd
import json
import plotly.express as px

from api import check_api_status, predict_engine
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
# SIDEBAR
# ==========================================
st.sidebar.header("System Status")
if check_api_status():
    st.sidebar.success("🟢 Backend API Connected")
else:
    st.sidebar.error("🔴 Backend API Offline")
    st.stop() # Stop execution if backend is down

st.sidebar.markdown("---")
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Engine History (CSV)", type="csv")

# ==========================================
# MAIN DASHBOARD LOGIC
# ==========================================
if uploaded_file is not None:
    # 1. Load Data
    df = pd.read_csv(uploaded_file, sep="\s+", header=None)
    
    # Assign column names (assuming raw CMAPSS format)
    columns = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
    if len(df.columns) == 26:
        df.columns = columns
        
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
                c1.metric("Remaining Useful Life", f"{result['predicted_rul']} Flights")
                c2.metric("Health Score", f"{result['health_score']} %")
                c3.metric("Risk Level", result['risk_level'])
                
                # Recommendation Banner
                if result['risk_level'] == "High":
                    st.error(f"⚠️ **Action Required:** {result['recommendation']}")
                elif result['risk_level'] == "Medium":
                    st.warning(f"⚡ **Warning:** {result['recommendation']}")
                else:
                    st.success(f"✅ **Healthy:** {result['recommendation']}")
                
                st.markdown("---")
                
                # ==========================================
                # VISUALIZATIONS
                # ==========================================
                col1, col2 = st.columns([1, 2])
                
                with col1:
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
                    
                st.markdown("---")
                
                # ==========================================
                # HISTORICAL SENSOR TREND
                # ==========================================
                st.subheader("Historical Sensor Analysis")
                sensor_cols = [c for c in engine_df.columns if "sensor" in c]
                selected_sensor = st.selectbox("Select Sensor to Plot", sensor_cols)
                
                trend_fig = px.line(
                    engine_df, 
                    x="cycle", 
                    y=selected_sensor, 
                    title=f"{selected_sensor} History for Engine {selected_engine}",
                    markers=True
                )
                st.plotly_chart(trend_fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
else:
    # Splash Screen when no data is uploaded
    st.info("👈 Upload a raw CMAPSS `test_FD001.txt` CSV file in the sidebar to begin.")