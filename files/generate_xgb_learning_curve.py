import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Create an XGBoost model and track training/validation MAE over iterations
print("Loading data...")
try:
    df = pd.read_csv("data/features/Time series features_ml.csv", encoding="utf-16le")
except:
    df = pd.read_csv("data/features/Time series features_ml.csv", encoding="utf-8")

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

# Basic preprocessing matching previous searches
if "target_7" not in df.columns:
    df["target"] = df["qty"] if "qty" in df.columns else df.iloc[:, 1]
else:
    df["target"] = df["target_7"]

df["sku_id"] = df["sku"].astype("category").cat.codes

features = [col for col in df.columns if col not in ['date', 'sku', 'target', 'target_7', 'target_smooth', 'target_log']]
df = df.dropna(subset=features + ["target"])

X = df[features].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "reg:squarederror",
    "eval_metric": ["mae", "rmse"],
    "max_depth": 8,
    "learning_rate": 0.05,
    "seed": 42
}

evals_result = {}
print("Training model to capture learning curve...")
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=[(dtrain, "Train"), (dtest, "Validation")],
    evals_result=evals_result,
    early_stopping_rounds=50,
    verbose_eval=False
)

train_mae = evals_result["Train"]["mae"]
val_mae = evals_result["Validation"]["mae"]
epochs = len(train_mae)
x_axis = range(0, epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, train_mae, label='Train MAE', color='#1f77b4', linewidth=2)
plt.plot(x_axis, val_mae, label='Validation MAE', color='#ff7f0e', linewidth=2)
plt.legend()
plt.ylabel('Mean Absolute Error (MAE)')
plt.xlabel('Number of Estimators (Boosting Rounds)')
plt.title('XGBoost Training and Validation MAE vs. Estimators')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

output_file = 'xgboost_learning_curve.png'
plt.savefig(output_file, dpi=300)
print(f"Saved {output_file}")
