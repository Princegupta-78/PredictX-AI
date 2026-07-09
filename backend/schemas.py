from pydantic import BaseModel, Field
from typing import List, Dict, Any

class SensorReading(BaseModel):
    engine_id: int
    cycle: int
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float

class EngineHistoryRequest(BaseModel):
    # min_length=5 ensures we have enough cycles to calculate our rolling window features!
    readings: List[SensorReading] = Field(
        ...,
        min_length=5,
        description="Complete engine history (minimum 5 cycles required for rolling stats)"
    )

class PredictionResponse(BaseModel):
    predicted_rul: float
    health_score: float
    risk_level: str
    recommendation: str
    explanation: Dict[str, Any]