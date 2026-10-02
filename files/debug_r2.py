import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.model_selection import train_test_split

data = joblib.load("models/xgb_model.joblib")
models = data['model']
features = data['features']

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

X = df[features].values
y = df["target_log"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
dtest = xgb.DMatrix(X_test, label=y_test)

preds = [m.predict(dtest) for m in models]
y_pred_log = np.mean(preds, axis=0)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

r2 = r2_score(y_true, y_pred)
r2_log = r2_score(y_test, y_pred_log)

print(f"R2 on actuals: {r2:.4f}")
print(f"R2 on log values: {r2_log:.4f}")

preds_train = [m.predict(xgb.DMatrix(X_train)) for m in models]
y_pred_train_log = np.mean(preds_train, axis=0)
r2_train = r2_score(np.expm1(y_train), np.expm1(y_pred_train_log))
print(f"R2 on training set: {r2_train:.4f}")
