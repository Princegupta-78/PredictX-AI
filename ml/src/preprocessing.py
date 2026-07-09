import pandas as pd
from .config import RUL_CAP, WINDOW, DROP_SENSORS

def add_rolling_features(df, window=WINDOW):
    """Adds rolling mean and standard deviation for sensor columns."""
    df = df.sort_values(["engine_id", "cycle"]).copy()
    
    # Identify sensor columns dynamically
    sensor_cols = [c for c in df.columns if c.startswith("sensor_") and c not in DROP_SENSORS]
    
    for col in sensor_cols:
        df[f"{col}_rollmean"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        df[f"{col}_rollstd"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )
    return df

def preprocess_training_data(df):
    """Full preprocessing pipeline for training data."""
    # 1. Drop useless sensors
    df_clean = df.drop(columns=DROP_SENSORS, errors='ignore')
    
    # 2. Calculate Piecewise RUL
    if "RUL" not in df_clean.columns:
        max_cycle = df_clean.groupby("engine_id")["cycle"].transform("max")
        df_clean["RUL"] = max_cycle - df_clean["cycle"]
        
    # Apply capping vectorization
    df_clean["RUL"] = df_clean["RUL"].clip(upper=RUL_CAP)
    
    # 3. Add Rolling Features
    df_features = add_rolling_features(df_clean)
    df_features = df_features.fillna(0)
    
    return df_features