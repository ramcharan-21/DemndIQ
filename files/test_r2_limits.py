import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import xgboost as xgb

df = pd.read_csv("data/features/Time series features_ml.csv")
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

df = df[df["target_7"] > 12].reset_index(drop=True)
p99 = df["target_7"].quantile(0.99)
df["target_7"] = np.where(df["target_7"] > p99, p99, df["target_7"])
df["target_smooth"] = df["target_7"].rolling(window=7, min_periods=1).mean()
df["target_log"] = np.log1p(df["target_smooth"])

y_true_log = df["target_log"].values
y_true = np.expm1(y_true_log)

# Test 1: Perfect prediction
print("1. Perfect prediction R2:", r2_score(y_true, y_true))

# Test 2: Perfect prediction but rounded (as in our script)
y_pred_rounded = np.rint(y_true)
print("2. Perfect but rounded to int R2:", r2_score(y_true, y_pred_rounded))

# Test 3: What does generate_metrics_table.py do?
features = [
    "sku_id", "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
    "roll_3_mean", "roll_7_mean", "roll_14_mean", "roll_28_mean",
    "roll_3_std", "roll_7_std", "roll_14_std",
    "ema_7", "ema_14", "ema_28",
    "revenue_lag_1", "price_ratio", "vol_ratio",
    "trend_ratio_7_28", "diff_7_14", "diff_1",
    "dow_sin", "dow_cos", "month", "day", "weekend"
]
df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
df["sku_id"] = df["sku"].astype("category").cat.codes

df_clean = df.dropna(subset=features)
y_true_clean_log = df_clean["target_log"].values
y_true_clean = np.expm1(y_true_clean_log)
print("3. Perfect but on dropna dataset R2:", r2_score(y_true_clean, y_true_clean))
