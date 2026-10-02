import joblib
import pandas as pd
import xgboost as xgb
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

CUR_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in CUR_FILE.parents if p.name == "bizchainconnect")
IN = PROJECT_ROOT / "data" / "features" / "Time series features_ml.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_model.joblib"

def get_error_interpretation(r2):
    if r2 >= 0.95: return "Very Low Error"
    if r2 >= 0.90: return "Low Error"
    if r2 >= 0.80: return "Moderate Error"
    return "High Error"

def main():
    print("Loading data...")
    df = pd.read_csv(IN)
    
    # --- Preprocessing (Must match train_xgb.py) ---
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
    df = df[df["target_7"] > 12].reset_index(drop=True)
    
    p99 = df["target_7"].quantile(0.99)
    df["target_7"] = np.where(df["target_7"] > p99, p99, df["target_7"])
    
    df["target_smooth"] = df["target_7"].rolling(window=7, min_periods=1).mean()
    df["target_log"] = np.log1p(df["target_smooth"])
    
    # Encodings
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["sku_id"] = df["sku"].astype("category").cat.codes
    
    features = [
        "sku_id", "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
        "roll_3_mean", "roll_7_mean", "roll_14_mean", "roll_28_mean",
        "roll_3_std", "roll_7_std", "roll_14_std",
        "ema_7", "ema_14", "ema_28",
        "revenue_lag_1", "price_ratio", "vol_ratio",
        "trend_ratio_7_28", "diff_7_14", "diff_1",
        "dow_sin", "dow_cos", "month", "day", "weekend"
    ]
    
    X = df[features].values
    y_true_log = df["target_log"].values
    y_true = np.expm1(y_true_log)
    
    # --- Load Model ---
    print("Loading model...")
    obj = joblib.load(MODEL_PATH)
    models = obj["model"]
    if not isinstance(models, list): models = [models]
    
    # --- Predict ---
    print("Predicting...")
    dmat = xgb.DMatrix(X)
    preds = []
    for m in models:
        preds.append(m.predict(dmat))
    y_pred_log = np.mean(preds, axis=0)
    y_pred = np.rint(np.expm1(y_pred_log))
    
    # --- Calculate Metrics by Segment ---
    results = []
    
    # 1. Overall
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results.append(["Overall Demand Prediction", r2, mae, rmse, get_error_interpretation(r2)])
    
    # 2. High Volatility (std > median)
    median_std = df["roll_7_std"].median()
    high_vol_mask = df["roll_7_std"] > median_std
    if high_vol_mask.sum() > 0:
        y_h_true = y_true[high_vol_mask]
        y_h_pred = y_pred[high_vol_mask]
        r2_h = r2_score(y_h_true, y_h_pred)
        results.append(["High Volatility SKUs", r2_h, mean_absolute_error(y_h_true, y_h_pred), np.sqrt(mean_squared_error(y_h_true, y_h_pred)), get_error_interpretation(r2_h)])

    # 3. Low Volatility
    low_vol_mask = ~high_vol_mask
    if low_vol_mask.sum() > 0:
        y_l_true = y_true[low_vol_mask]
        y_l_pred = y_pred[low_vol_mask]
        r2_l = r2_score(y_l_true, y_l_pred)
        results.append(["Low Volatility SKUs", r2_l, mean_absolute_error(y_l_true, y_l_pred), np.sqrt(mean_squared_error(y_l_true, y_l_pred)), get_error_interpretation(r2_l)])

    # --- Write Table to File ---
    with open("metrics.md", "w") as f:
        f.write("\nTable 4.1: Performance Metrics of Demand Forecasting Models\n\n")
        f.write(f"| {'Parameter':<25} | {'R2 Score':<10} | {'MAE':<10} | {'RMSE':<10} | {'Error Interpretation':<20} |\n")
        f.write(f"|{'-'*27}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*22}|\n")
        
        for row in results:
            f.write(f"| {row[0]:<25} | {row[1]:<10.3f} | {row[2]:<10.3f} | {row[3]:<10.3f} | {row[4]:<20} |\n")
        f.write("\n")
    
    print("Metrics written to metrics.md")

if __name__ == "__main__":
    main()
