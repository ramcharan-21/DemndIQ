import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path to enable imports
CUR_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in CUR_FILE.parents if p.name == "bizchainconnect")
sys.path.append(str(PROJECT_ROOT))

# Import project modules
from src.rl_features import get_latest_features_for_sku
from src.xgb_forecast import cached_predict_demand

def get_sku_list():
    """Load unique SKUs from the features file."""
    data_path = PROJECT_ROOT / "data" / "features" / "Time series features_ml.csv"
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return []
    
    # Read just the 'sku' column to be fast
    df = pd.read_csv(data_path, usecols=["sku"])
    return sorted(df["sku"].unique())

def calculate_sku_metrics(sku):
    """Calculate forecast and reorder logic for a single SKU."""
    # 1. Get Inputs
    feat, std = get_latest_features_for_sku(sku)
    
    # 2. Forecast
    raw_forecast = cached_predict_demand(tuple(feat))
    
    # Fallback Logic (mirroring api.py)
    # roll_7_mean is at index 7
    roll_7 = float(feat[7])
    
    if raw_forecast < 0.5 * roll_7:
        forecast = roll_7
    else:
        forecast = raw_forecast
        
    # 3. Reorder Logic
    lead_time = 15
    dynamic_safety_stock = max(50, forecast * 7)
    reorder_point = (forecast * lead_time) + dynamic_safety_stock
    
    return {
        "SKU": sku,
        "Forecasted_Demand": round(forecast, 2),
        "Safety_Stock": round(dynamic_safety_stock, 2),
        "Reorder_Point": round(reorder_point, 2),
        "Volatility_Std": round(std, 2)
    }

def generate_report():
    print("Loading SKU list...")
    skus = get_sku_list()
    print(f"Found {len(skus)} SKUs. Calculating metrics...")
    
    results = []
    for sku in skus:
        metrics = calculate_sku_metrics(sku)
        results.append(metrics)
        
    df = pd.DataFrame(results)
    
    # --- 1. Save Table ---
    out_csv = PROJECT_ROOT / "sku_reorder_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved summary table to {out_csv}")
    
    # --- 2. Generate Graph ---
    print("Generating summary graph...")
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    # We will plot Forecast vs Reorder Point
    # Melt for easier plotting with seaborn
    df_melt = df.melt(id_vars=["SKU"], value_vars=["Forecasted_Demand", "Reorder_Point"], 
                      var_name="Metric", value_name="Units")
    
    # Create Bar Chart
    chart = sns.barplot(data=df_melt, x="SKU", y="Units", hue="Metric", palette="viridis")
    
    plt.title("Forecast vs Reorder Point by SKU", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    out_img = PROJECT_ROOT / "all_products_summary.png"
    plt.savefig(out_img)
    print(f"✅ Saved summary graph to {out_img}")
    
    # --- 3. Print Top 5 Table for Log ---
    print("\n--- Top 5 SKUs by Forecast ---")
    print(df.sort_values("Forecasted_Demand", ascending=False).head(5).to_markdown(index=False))

if __name__ == "__main__":
    generate_report()
