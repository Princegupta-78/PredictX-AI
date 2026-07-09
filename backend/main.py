from fastapi import FastAPI
from backend.routers.prediction import router

app = FastAPI(
    title="PredictX AI",
    version="1.0.0",
    description="Predictive Maintenance API for Aircraft Engines using XGBoost and SHAP.",
    contact={
        "name": "Prince Gupta"
    }
)

# Register the prediction endpoints
app.include_router(router)