import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/features/Time series features_ml.csv")
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

df = df[df["target_7"] > 35].reset_index(drop=True)
p99 = df["target_7"].quantile(0.99)
df["target_7"] = np.where(df["target_7"] > p99, p99, df["target_7"])

df["target_smooth"] = df["target_7"].rolling(window=7, min_periods=1).mean()
df["target_log"] = np.log1p(df["target_smooth"])

df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
df["sku_id"] = df["sku"].astype("category").cat.codes
df["lag_30"] = df["target_7"].shift(30)
df["lag_60"] = df["target_7"].shift(60)
df["roll_30_mean"] = df["target_7"].rolling(window=30, min_periods=1).mean()
df["roll_30_std"] = df["target_7"].rolling(window=30, min_periods=1).std()

features = [
    "sku_id", "qty", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7",
    "lag_14", "lag_21", "lag_28", "lag_30", "lag_60",
    "roll_3_mean", "roll_7_mean", "roll_14_mean", "roll_28_mean", "roll_30_mean",
    "roll_7_max", "roll_7_min", "roll_3_std", "roll_7_std", "roll_14_std", "roll_30_std",
    "ema_7", "ema_28", "avg_unit_price", "online_ratio", "revenue_lag_1", "revenue_roll_7_std",
    "price_ratio", "price_change_7", "vol_ratio", "trend_ratio_7_28", "diff_7_14", "diff_1",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "week_sin", "week_cos", "weekend",
    "month", "day", "week_of_year" # add raw time features
]

# Ensure we drop NaNs so features shape matches
df = df.dropna(subset=features + ["target_log"])

X = df[features].values
y = df["target_log"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 20, 
    "learning_rate": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "gamma": 0.0,
    "seed": 42
}

print("Training high-capacity model...")
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtest, "eval")],
    early_stopping_rounds=50,
    verbose_eval=100,
)

preds = bst.predict(dtest)
y_pred_log = preds
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

r2 = r2_score(y_true, y_pred)
print(f"R2 on actuals: {r2:.4f}")

preds_train = bst.predict(dtrain)
r2_train = r2_score(np.expm1(y_train), np.expm1(preds_train))
print(f"R2 on training set: {r2_train:.4f}")
