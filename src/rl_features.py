import pandas as pd
import numpy as np
from pathlib import Path

from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_FILE = BASE_DIR / "data" / "features" / "Time series features_ml.csv"

_FEATURE_DF = None

@lru_cache(maxsize=1024)
def get_latest_features_for_sku(sku: str):
    global _FEATURE_DF
    if not FEATURE_FILE.exists():
        # absolute fallback
        return np.array([0] * 27, dtype=np.float32), 0.0

    if _FEATURE_DF is None:
        _FEATURE_DF = pd.read_csv(FEATURE_FILE)
    df = _FEATURE_DF

    sku_df = df[df["sku"] == sku]

    # 🚨 CRITICAL SAFETY CHECK
    if sku_df.empty or len(sku_df) < 2:
        # Not enough history → fallback
        return np.array([0, 0, 0, 0]), 0.0

    recent = sku_df.tail(14).copy()

    # Safe extraction with defaults
    # Feature list matching train_xgb.py (27 features)
    features_list = [
        "sku_id", "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
        "roll_3_mean", "roll_7_mean", "roll_14_mean", "roll_28_mean",
        "roll_3_std", "roll_7_std", "roll_14_std",
        "ema_7", "ema_14", "ema_28",
        "revenue_lag_1", "price_ratio", "vol_ratio",
        "trend_ratio_7_28", "diff_7_14", "diff_1",
        "dow_sin", "dow_cos", "month", "day", "weekend"
    ]

    # Fill missing columns with 0
    for col in features_list:
        if col not in recent.columns:
            recent[col] = 0

    # Extract last row as vector
    last_row = recent.iloc[-1]
    feat_vec = last_row[features_list].values.astype(np.float32)

    demand_std = float(recent["qty"].std()) if "qty" in recent else 0.0

    return feat_vec, demand_std
