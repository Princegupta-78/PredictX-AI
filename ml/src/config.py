from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================
# ROOT_DIR points to PredictX-AI/ml/
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Preprocessing Constants
# ==========================================================
RUL_CAP = 125
WINDOW = 5
DROP_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6",
    "sensor_10", "sensor_16", "sensor_18", "sensor_19"
]

# ==========================================================
# Model Constants
# ==========================================================
RANDOM_STATE = 42

# The optimized parameters from your Day 5/6 experiments
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}