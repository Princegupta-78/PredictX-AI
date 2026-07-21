#!/bin/bash
set -e

# 1. Start FastAPI backend internally (correct module)
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

# 2. Give it time to load XGBoost model
sleep 8

# 3. Start Streamlit on Render's public port
streamlit run app/dashboard.py --server.port $PORT --server.address 0.0.0.0