import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ModelExplainer:
    def __init__(self, model_path):
        """Loads the saved artifact and initializes the SHAP TreeExplainer."""
        # 1. Load the raw model directly
        self.model = joblib.load(model_path)
        
        # 2. Extract feature names directly from the model
        self.feature_cols = self.model.feature_names_in_.tolist()
        
        # 3. Initialize SHAP
        self.explainer = shap.TreeExplainer(self.model)
        
        # 4. Setup paths for saving plots
        self.docs_path = Path(__file__).resolve().parents[2] / "docs"
        self.docs_path.mkdir(parents=True, exist_ok=True)
    
    def get_shap_values(self, X):
        """Computes SHAP values for a given dataset."""
        return self.explainer(X)

    def summary_plot(self, X):
        """Generates and saves the Global Feature Importance plot."""
        shap_values = self.get_shap_values(X)
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(self.docs_path / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

    def waterfall_plot(self, X, index):
        """Generates and saves a Waterfall plot for a specific prediction."""
        shap_values = self.get_shap_values(X)
        shap.plots.waterfall(shap_values[index], show=False)
        plt.tight_layout()
        plt.savefig(self.docs_path / "shap_waterfall.png", dpi=300, bbox_inches="tight")
        plt.close()

    def force_plot(self, X, index):
        """Generates and saves a Force plot for a specific prediction."""
        shap_values = self.get_shap_values(X)
        shap.force_plot(
            self.explainer.expected_value,
            shap_values.values[index],
            X.iloc[index],
            matplotlib=True,
            show=False
        )
        plt.savefig(self.docs_path / "shap_force.png", dpi=300, bbox_inches="tight")
        plt.close()

    def explain_prediction(self, X, index):
        """Returns a DataFrame detailing how each feature pushed the prediction."""
        shap_values = self.get_shap_values(X)
        prediction = self.model.predict(X.iloc[[index]])[0]

        contributions = pd.DataFrame({
            "Feature": self.feature_cols,
            "Feature Value": X.iloc[index].values,
            "SHAP Value": shap_values.values[index]
        })
        
        # Sort by absolute impact (highest positive OR negative push)
        contributions = contributions.sort_values("SHAP Value", key=abs, ascending=False)
        return prediction, contributions

    def explain_json(self, X, index):
        """Returns a JSON-serializable dictionary for FastAPI integration."""
        prediction, table = self.explain_prediction(X, index)
        return {
            "prediction": float(prediction),
            "top_features": table.head(10).to_dict(orient="records")
        }