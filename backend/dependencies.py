import sys
from pathlib import Path

# Connect the backend to the ml directory
ml_root = Path(__file__).resolve().parents[1] / "ml"
sys.path.append(str(ml_root))

from src.inference import RULInferencePipeline

# Initialize the pipeline once globally
pipeline = RULInferencePipeline()