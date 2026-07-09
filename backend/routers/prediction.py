from fastapi import APIRouter, HTTPException
import pandas as pd

from backend.schemas import EngineHistoryRequest, PredictionResponse
from backend.dependencies import pipeline
from src.preprocessing import preprocess_training_data

router = APIRouter()

@router.get("/")
def health_check():
    """Basic health check endpoint."""
    return {
        "status": "Running",
        "model": "XGBoost + SHAP",
        "version": "1.0.0"
    }

@router.post("/predict", response_model=PredictionResponse)
def predict_rul(request: EngineHistoryRequest):
    """Takes raw engine history, preprocesses it, and returns the RUL prediction."""
    try:
        # 1. Convert incoming JSON to Pandas DataFrame
        df = pd.DataFrame([r.model_dump() for r in request.readings])
        
        # 2. Preprocess (Drop sensors, cap RUL, add rolling features)
        processed_df = preprocess_training_data(df)
        
        # 3. Predict and Explain
        result = pipeline.predict(processed_df)
        
        return result

    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing required column: {e}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Data validation error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")