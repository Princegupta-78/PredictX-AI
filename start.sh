#!/bin/bash

# 1. Start the FastAPI backend in the background on port 8000
# (Targets app/api.py)
uvicorn app.api:predict_api --host 0.0.0.0 --port 8000 &

# 2. Give the backend 5 seconds to load the ML model into memory
sleep 5

# 3. Start the Streamlit frontend in the foreground on Render's assigned port
# (Targets app/dashboard.py)
streamlit run app/dashboard.py --server.port $PORT --server.address 0.0.0.0