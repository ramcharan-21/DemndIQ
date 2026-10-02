import pandas as pd
import numpy as np
import time
from pathlib import Path
from src.xgb_forecast import cached_predict_demand

FEATURE_FILE = Path("data/features/Time series features_ml.csv")

t0 = time.time()
df = pd.read_csv(FEATURE_FILE)
print(f"Read CSV in {time.time() - t0:.4f}s")

skus = df["sku"].unique()
print(f"Total Unique SKUs: {len(skus)}")

# Measure one prediction
t1 = time.time()
feat_vec = np.zeros(27, dtype=np.float32)
cached_predict_demand(tuple(feat_vec))
print(f"One prediction (cold): {time.time() - t1:.4f}s")

t2 = time.time()
for _ in range(10):
    cached_predict_demand(tuple(feat_vec))
print(f"10 predictions (warm): {time.time() - t2:.4f}s")
