import pandas as pd
import numpy as np
import time
from pathlib import Path
from src.rl_features import get_latest_features_for_sku
from src.xgb_forecast import cached_predict_demand

FEATURE_FILE = Path("data/features/Time series features_ml.csv")

def run_logic():
    t0 = time.time()
    df = pd.read_csv(FEATURE_FILE)
    print(f"Read CSV: {time.time()-t0:.2f}s")
    
    features_list = [
        "sku_id", "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
        "roll_3_mean", "roll_7_mean", "roll_14_mean", "roll_28_mean",
        "roll_3_std", "roll_7_std", "roll_14_std",
        "ema_7", "ema_14", "ema_28",
        "revenue_lag_1", "price_ratio", "vol_ratio",
        "trend_ratio_7_28", "diff_7_14", "diff_1",
        "dow_sin", "dow_cos", "month", "day", "weekend"
    ]
    
    count = 0
    t_loop = time.time()
    for sku, sku_df in df.groupby("sku"):
        count += 1
        if len(sku_df) >= 2:
            recent = sku_df.tail(14).copy()
            for col in features_list:
                if col not in recent.columns:
                    recent[col] = 0
            
            last_row = recent.iloc[-1]
            feat_vec = last_row[features_list].values.astype(np.float32)
            
            # Predict
            cached_predict_demand(tuple(feat_vec))
            
    print(f"Processed {count} SKUs in {time.time()-t_loop:.2f}s")

if __name__ == "__main__":
    run_logic()
