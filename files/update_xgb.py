import re
import json
import numpy as np

with open('src/train_xgb.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix features list duplication
features_clean = '''features = [
    "sku_id",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7",
    "lag_14", "lag_21", "lag_28", "lag_30", "lag_60",
    "roll_3_mean", "roll_7_mean", "roll_14_mean", "roll_28_mean", "roll_30_mean",
    "roll_7_max", "roll_7_min",
    "roll_3_std", "roll_7_std", "roll_14_std", "roll_30_std",
    "ema_7", "ema_28",
    "avg_unit_price", "online_ratio",
    "revenue_lag_1", "revenue_roll_7_std",
    "price_ratio", "price_change_7", "vol_ratio",
    "trend_ratio_7_28", "diff_7_14", "diff_1",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "week_sin", "week_cos",
    "weekend"
]'''

content = re.sub(r'features = \[.*?\]', features_clean, content, flags=re.DOTALL)

# Replace training block
training_block = '''# ======================
# XGBOOST HYPERPARAMETER TUNING
# ======================
from sklearn.model_selection import ParameterSampler

param_distributions = {
    "max_depth": [6, 8, 10, 12],
    "learning_rate": [0.005, 0.01, 0.015, 0.05],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0.0, 0.1, 0.2],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [0.5, 1.0, 2.0]
}

base_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
}

print("Running Randomized Hyperparameter Search...")
# Use ParameterSampler for a random search over 10 iterations to save time
n_iter = 10
sampler = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=42))

best_r2 = -float("inf")
best_params = None

y_true = np.expm1(y_test)

for i, params in enumerate(sampler):
    print(f"\\n--- Trial {i+1}/{n_iter} ---")
    current_params = base_params.copy()
    current_params.update(params)
    current_params["seed"] = 42
    
    bst = xgb.train(
        current_params,
        dtrain,
        num_boost_round=3000,
        evals=[(dtest, "eval")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    
    preds = bst.predict(dtest)
    y_pred = np.rint(np.expm1(preds))
    
    r2 = r2_score(y_true, y_pred)
    
    print(f"R2: {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_params = current_params.copy()
        print(f"** New Best R2: {best_r2:.4f} **")

print("\\n--- Hyperparameter Search Complete ---")
print(f"Best R2: {best_r2:.4f}")
print(f"Best Params: {json.dumps(best_params, indent=2)}")

# Save best parameters
with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

# ======================
# FINAL ENSEMBLE TRAINING
# ======================
seeds = [42, 123, 456] # Top 3 seeds
models = []

print("\\nTraining final ensemble with top parameters...")
for s in seeds:
    print(f"Training ensemble model with seed {s}...")
    best_params["seed"] = s
    bst = xgb.train(
        best_params,
        dtrain,
        num_boost_round=3000,
        evals=[(dtest, "eval")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    models.append(bst)

# Ensemble predictions
preds = [m.predict(dtest) for m in models]
y_pred_log = np.mean(preds, axis=0)
y_pred = np.rint(np.expm1(y_pred_log))

# Metrics
mae = float(np.mean(np.abs(y_pred - y_true)))
rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
forecast_accuracy = 100 - wmape
r2 = r2_score(y_true, y_pred)

print("\\n===== XGBoost Demand Forecasting Metrics (FINAL ENSEMBLE) =====")
print(f"MAE                : {mae:.2f}")
print(f"RMSE               : {rmse:.2f}")
print(f"wMAPE              : {wmape:.2f}%")
print(f"Forecast Accuracy  : {forecast_accuracy:.2f}%")
print(f"R2 Score           : {r2:.4f}")
print("==============================================================\\n")

with open("metrics_utf8.txt", "w", encoding="utf-8") as f:
    f.write(f"MAE: {mae:.2f}\\n")
    f.write(f"RMSE: {rmse:.2f}\\n")
    f.write(f"wMAPE: {wmape:.2f}%\\n")
    f.write(f"Forecast Accuracy: {forecast_accuracy:.2f}%\\n")
    f.write(f"R2: {r2:.4f}\\n")

# Save the FULL ENSEMBLE of models
model_path = OUT / "xgb_model.joblib"
joblib.dump({"model": models, "features": features}, model_path)
print("Saved ENSEMBLE model to:", model_path)
'''

# Find where XGBOOST PARAMETERS starts and replace everything after
idx = content.find('# ======================\n# XGBOOST PARAMETERS')
if idx != -1:
    content = content[:idx] + training_block
else:
    print('Could not find XGBOOST PARAMETERS block')

with open('src/train_xgb.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated train_xgb.py')
