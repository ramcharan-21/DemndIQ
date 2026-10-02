import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from pathlib import Path

# Add project root to path
CUR_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in CUR_FILE.parents if p.name == "bizchainconnect")
sys.path.append(str(PROJECT_ROOT))

from src.xgb_forecast import load_models

def generate_heatmap():
    # 1. Load Data
    data_path = PROJECT_ROOT / "data" / "features" / "Time series features_ml.csv"
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Preprocess (Match training script logic)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
    # Recalculate computed features used in training
    if "dow" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    
    # Simple SKU encoding (Note: In production, load the specific encoder mapping!)
    # Ideally we should match the exact codes from training. 
    # train_xgb.py does: df["sku_id"] = df["sku"].astype("category").cat.codes
    # We will do the same on the FULL dataset to hope for consistent alignment.
    df["sku_id"] = df["sku"].astype("category").cat.codes
    
    # Filter for the last 30 days available in the dataset
    max_date = df['date'].max()
    start_date = max_date - pd.Timedelta(days=29)
    df_subset = df[df['date'] >= start_date].copy()
    
    print(f"Analyzing data from {start_date.date()} to {max_date.date()} ({len(df_subset)} rows)")
    
    # Explicit Feature List (MUST Match train_xgb.py exactly)
    features = [
        "sku_id",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_21",
        "lag_28",
        "roll_3_mean",
        "roll_7_mean",
        "roll_14_mean",
        "roll_28_mean",
        "roll_3_std",
        "roll_7_std",
        "roll_14_std",
        "ema_7",
        "ema_14",
        "ema_28",
        "revenue_lag_1",
        "price_ratio",
        "vol_ratio",
        "trend_ratio_7_28",
        "diff_7_14",
        "diff_1",
        "dow_sin",
        "dow_cos",
        "month",
        "day",
        "weekend"
    ]
    
    # Verify all features exist
    missing = [f for f in features if f not in df_subset.columns]
    if missing:
        print(f"Error: Missing features in dataset: {missing}")
        return

    X = df_subset[features].values
    y_true = df_subset['qty'].values
    
    # 3. Predict (Batch)
    print("Generating predictions...")
    models = load_models()
    if not isinstance(models, list):
        models = [models]
        
    dmat = xgb.DMatrix(X, feature_names=features)
    
    # Ensemble Prediction
    preds_log = []
    for model in models:
        preds_log.append(model.predict(dmat))
        
    avg_log_pred = np.mean(preds_log, axis=0)
    y_pred = np.rint(np.expm1(avg_log_pred))
    y_pred = np.maximum(0, y_pred)
    
    # 4. Calculate Accuracy
    # Avoid division by zero
    # Accuracy % = 1 - abs(y_true - y_pred) / (y_true + 1)
    # Using (y_true + 1) to handle 0 actual demand smoothly for visualization
    ape = np.abs(y_true - y_pred) / (y_true + 1.0)
    accuracy = np.clip(1.0 - ape, 0, 1) * 100
    
    df_subset['Accuracy'] = accuracy
    df_subset['DateStr'] = df_subset['date'].dt.strftime('%Y-%m-%d')
    
    # 5. Pivot for Heatmap
    heatmap_data = df_subset.pivot(index='sku', columns='DateStr', values='Accuracy')
    
    # 6. Plot
    print("Generating heatmap...")
    plt.figure(figsize=(20, 12))
    sns.set_theme(style="white")
    
    # Create Heatmap
    # Green (High Acc) to Red (Low Acc) - 'RdYlGn'
    sns.heatmap(heatmap_data, cmap='RdYlGn', vmin=0, vmax=100, 
                cbar_kws={'label': 'Forecast Accuracy (%)'})
    
    plt.title(f"Model Performance Heatmap (Last 30 Days)\nAccuracy = 1 - |Actual - Pred| / (Actual + 1)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("SKU", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    out_img = PROJECT_ROOT / "model_performance_heatmap.png"
    plt.savefig(out_img, dpi=300)
    print(f"✅ Saved heatmap to {out_img}")

if __name__ == "__main__":
    generate_heatmap()
