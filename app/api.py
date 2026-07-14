import os
import requests

# Dynamically fetch the API URL. Defaults to localhost for local testing.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def check_api_status():
    """Pings the FastAPI backend to ensure it's online."""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
        

def predict_engine(engine_df):
    """Sends engine data to the FastAPI backend and returns the JSON prediction."""
    payload = {
        "readings": engine_df.to_dict(orient="records")
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=20
    )
    
    response.raise_for_status()
    return response.json()