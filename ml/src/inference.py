import joblib
import pandas as pd
from pathlib import Path

# Connect to sibling modules
from src.explain import ModelExplainer
from src.config import MODEL_DIR

class RULInferencePipeline:
    # Configurable Business Logic Thresholds
    HIGH_RISK = 20
    MEDIUM_RISK = 50

    def __init__(self):
        """Initializes the model, feature list, and SHAP explainer."""
        # Using the Day 8 fix: load raw model and extract features dynamically
        model_path = MODEL_DIR / "xgb_rul_model.pkl"
        self.model = joblib.load(model_path)
        self.feature_cols = self.model.feature_names_in_.tolist()
        
        # Initialize Explainer
        self.explainer = ModelExplainer(model_path)

    def get_risk_level(self, rul):
        """Classifies risk based on remaining flights."""
        if rul <= self.HIGH_RISK:
            return "High"
        elif rul <= self.MEDIUM_RISK:
            return "Medium"
        return "Low"

    def get_health_score(self, rul):
        """Converts RUL to a 0-100% health metric."""
        score = min(rul / 125, 1)
        return round(score * 100, 1)

    def get_recommendation(self, risk):
        """Provides actionable maintenance advice."""
        recommendations = {
            "High": "Immediate inspection recommended. Engine nearing failure.",
            "Medium": "Schedule maintenance within the next inspection cycle.",
            "Low": "Engine operating normally. Continue monitoring."
        }
        return recommendations[risk]

    def predict(self, engine_df):
        """
        The main pipeline execution function.
        Takes a processed engine DataFrame and returns a comprehensive JSON response.
        """
        # Grab the absolute latest cycle for this engine
        latest = engine_df.sort_values("cycle").iloc[[-1]]
        
        # Filter down to just the features the model needs
        X = latest[self.feature_cols]

        # 1. Get Prediction
        prediction = float(self.model.predict(X)[0])
        
        # 2. Get SHAP Explanation
        explanation = self.explainer.explain_json(X, 0)

        # 3. Assemble Final JSON Payload
        return {
            "predicted_rul": round(prediction, 2),
            "health_score": self.get_health_score(prediction),
            "risk_level": self.get_risk_level(prediction),
            "recommendation": self.get_recommendation(self.get_risk_level(prediction)),
            "explanation": explanation
        }