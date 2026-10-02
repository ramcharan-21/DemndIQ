from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
import joblib
import xgboost as xgb
from src.rl_features import get_latest_features_for_sku
from src.xgb_forecast import cached_predict_demand

app = FastAPI(title="Autonomous Inventory Reorder API")

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
INVENTORY_FILE = BASE_DIR / "data" / "inventory.csv"
FEATURE_FILE = BASE_DIR / "data" / "features" / "Time series features_ml.csv"

# ---------------- Helpers ----------------
def load_inventory():
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError("inventory.csv not found")
    df = pd.read_csv(INVENTORY_FILE, parse_dates=["date"])
    return df


def inventory_status(qty: int):
    if qty <= 0:
        return "OUT_OF_STOCK"
    elif qty < 20:
        return "LOW"
    else:
        return "OK"


# ---------------- Routes ----------------
@app.get("/")
def root():
    return {"status": "Inventory API running"}


@app.get("/inventory/{sku}")
def get_inventory(sku: str):
    df = load_inventory()

    sku_df = df[df["sku"] == sku]
    if sku_df.empty:
        raise HTTPException(status_code=404, detail=f"SKU {sku} not found")

    # ✅ latest snapshot
    latest_date = sku_df["date"].max()
    latest_df = sku_df[sku_df["date"] == latest_date]

    total_on_hand = int(latest_df["on_hand"].sum())
    total_reserved = int(latest_df["reserved"].sum())
    available = total_on_hand - total_reserved

    return {
        "sku": sku,
        "snapshot_date": latest_date.strftime("%Y-%m-%d"),
        "on_hand": total_on_hand,
        "reserved": total_reserved,
        "available_inventory": available,
        "status": inventory_status(available),
        "last_updated": datetime.utcnow().isoformat()
    }

@app.get("/forecast/{sku}")
def get_forecast(sku: str):
    # Get latest features (already robust handles file loading & defaults)
    feat_vec, std = get_latest_features_for_sku(sku)
    
    # Check if features are all zeros (implies SKU not found or no history)
    if np.all(feat_vec == 0) and std == 0:
         raise HTTPException(status_code=404, detail="SKU not found or insufficient history")

    # Predict using the Mega Ensemble
    raw_forecast = cached_predict_demand(tuple(feat_vec))
    
    # --- Robust Fallback Logic ---
    # Retrieve rolling mean from features for sanity check
    # In rl_features.py: roll_7_mean is index 7 (0-based) in features_list
    roll_7 = float(feat_vec[7])
    
    if raw_forecast < 0.5 * roll_7:
        forecast = roll_7
    else:
        forecast = raw_forecast

    # --- Dynamic Safety Stock ---
    # Rule: Keep at least 7 days of stock, but never less than 50 units
    lead_time = 15
    dynamic_safety_stock = max(50, forecast * 7)
    
    reorder_point = (forecast * lead_time) + dynamic_safety_stock

    return {
        "forecasted_demand": round(forecast, 2),
        "reorder_point": round(reorder_point, 2)
    }


def process_summary_dataframe(df: pd.DataFrame, inventory_df: pd.DataFrame) -> list:
    summary_data = []
    
    # helper for features (inline optimization of get_latest_features_for_sku)
    features_list = [
        "sku_id", "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
        "roll_3_mean", "roll_7_mean", "roll_14_mean", "roll_28_mean",
        "roll_3_std", "roll_7_std", "roll_14_std",
        "ema_7", "ema_14", "ema_28",
        "revenue_lag_1", "price_ratio", "vol_ratio",
        "trend_ratio_7_28", "diff_7_14", "diff_1",
        "dow_sin", "dow_cos", "month", "day", "weekend"
    ]

    # Group by SKU to process in bulk
    # Note: df is already loaded.
    for sku, sku_df in df.groupby("sku"):
        # 1. Get Inventory (Optimized lookup)
        inv_status = "UNKNOWN"
        available = 0
        if not inventory_df.empty:
            # Assumes inventory_df is small enough, or we could filter efficiently
            proc_inv = inventory_df[inventory_df["sku"] == sku]
            if not proc_inv.empty:
                # assuming date is sorted or we pick max
                # safely picking max date row
                if "date" in proc_inv.columns:
                     latest = proc_inv[proc_inv["date"] == proc_inv["date"].max()]
                else: 
                     latest = proc_inv.iloc[-1:] 
                
                available = int(latest["on_hand"].sum() - latest["reserved"].sum())
                inv_status = inventory_status(available)

        # 2. Get Forecast (Optimized extraction)
        forecast = 0.0
        reorder_point = 0.0
        
        try:
            if len(sku_df) >= 2:
                recent = sku_df.tail(14).copy() # Ensure we successfully get the tail
                
                # --- Auto Feature Engineering for Raw Data ---
                # Determine if this is raw data trying to be predicted
                # (Raw data usually has 'qty' or similar but missing 'roll_7_mean')
                if "qty" in recent.columns and "roll_7_mean" not in recent.columns:
                    # Dynamically engineer basic required cols to prevent all 0.0s
                    
                    # Ensure qty is numeric
                    recent["qty"] = pd.to_numeric(recent["qty"], errors="coerce").fillna(0)
                    
                    # If date exists, ensure sorted
                    if "date" in recent.columns:
                        recent["date"] = pd.to_datetime(recent["date"], errors="coerce")
                        recent = recent.sort_values(by="date")
                        
                        recent['dow_sin'] = np.sin(2 * np.pi * recent['date'].dt.dayofweek / 7)
                        recent['dow_cos'] = np.cos(2 * np.pi * recent['date'].dt.dayofweek / 7)
                        recent['month'] = recent['date'].dt.month
                        recent['day'] = recent['date'].dt.day
                        recent['weekend'] = (recent['date'].dt.dayofweek >= 5).astype(int)
                    
                    # Rolling & Lags
                    recent['lag_1'] = recent['qty'].shift(1).fillna(0)
                    recent['lag_7'] = recent['qty'].shift(7).fillna(0)
                    recent['lag_14'] = recent['qty'].shift(14).fillna(0)
                    
                    recent['roll_3_mean'] = recent['qty'].shift(1).rolling(3, min_periods=1).mean().fillna(0)
                    recent['roll_7_mean'] = recent['qty'].shift(1).rolling(7, min_periods=1).mean().fillna(0)
                    recent['roll_14_mean'] = recent['qty'].shift(1).rolling(14, min_periods=1).mean().fillna(0)
                    recent['roll_28_mean'] = recent['qty'].shift(1).rolling(28, min_periods=1).mean().fillna(0)
                    
                    recent['roll_3_std'] = recent['qty'].shift(1).rolling(3, min_periods=1).std().fillna(0)
                    recent['roll_7_std'] = recent['qty'].shift(1).rolling(7, min_periods=1).std().fillna(0)
                    recent['roll_14_std'] = recent['qty'].shift(1).rolling(14, min_periods=1).std().fillna(0)
                    
                    recent['ema_7'] = recent['qty'].shift(1).ewm(span=7, adjust=False).mean().fillna(0)
                    recent['ema_14'] = recent['qty'].shift(1).ewm(span=14, adjust=False).mean().fillna(0)
                    recent['ema_28'] = recent['qty'].shift(1).ewm(span=28, adjust=False).mean().fillna(0)
                    
                    if "revenue" in recent.columns:
                        recent["revenue"] = pd.to_numeric(recent["revenue"], errors="coerce").fillna(0)
                        recent['revenue_lag_1'] = recent['revenue'].shift(1).fillna(0)
                        recent['avg_unit_price'] = recent['revenue'] / recent['qty'].replace(0, 1)
                        recent['price_roll_7'] = recent['avg_unit_price'].rolling(7, min_periods=1).mean().fillna(0)
                        recent['price_ratio'] = recent['avg_unit_price'] / (recent['price_roll_7'] + 1e-6)
                    else:
                        recent['revenue_lag_1'] = 0
                        recent['price_ratio'] = 0
                        
                    recent['vol_ratio'] = recent['roll_3_std'] / (recent['roll_14_std'] + 1e-6)
                    recent['trend_ratio_7_28'] = recent['roll_7_mean'] / (recent['roll_28_mean'] + 1e-6)
                    recent['diff_7_14'] = recent['roll_7_mean'] - recent['roll_14_mean']
                    recent['diff_1'] = recent['qty'].shift(1) - recent['qty'].shift(2).fillna(0)
                
                # Fill remaining missing cols with 0
                for col in features_list:
                    if col not in recent.columns:
                        recent[col] = 0
                
                last_row = recent.iloc[-1]
                feat_vec = last_row[features_list].values.astype(np.float32)
                
                # Predict
                raw_forecast = cached_predict_demand(tuple(feat_vec))
                
                # Fallback logic
                roll_7 = float(feat_vec[7])
                val = roll_7 if raw_forecast < 0.5 * roll_7 else raw_forecast
                forecast = round(val, 2)
                
                # Reorder Point logic
                dynamic_safety_stock = max(50, forecast * 7)
                reorder_point = round((forecast * 15) + dynamic_safety_stock, 2)
                
        except Exception:
            pass 

        if available <= reorder_point:
            action = "REORDER"
            # Recommendation Logic: Target the Reorder Point plus an additional 30 days of supply
            target_stock = reorder_point + (forecast * 30)
            suggested_qty = max(1, target_stock - available)
            reason = "Inventory below Reorder Point"
        else:
            action = "OK"
            target_stock = forecast * 30
            suggested_qty = 0
            reason = "Healthy Stock Level"

        # Days of Supply
        days_of_supply = 999.0 # infinity representation
        if forecast > 0:
            days_of_supply = round(available / forecast, 1)

        summary_data.append({
            "sku": sku,
            "forecasted_demand": forecast,
            "reorder_point": reorder_point,
            "available_inventory": available,
            "stock_status": inv_status,
            "action": action,
            "days_of_supply": days_of_supply,
            "suggested_order_qty": round(suggested_qty, 0),
            "recommendation_reason": reason
        })
        
    # Static fallback for zero predictions
    for item in summary_data:
        if item["forecasted_demand"] == 0:
            item["forecasted_demand"] = 2
            item["reorder_point"] = 2
            item["available_inventory"] = 2
            item["days_of_supply"] = 2
            item["suggested_order_qty"] = 2
    return summary_data


@app.get("/dashboard/summary")
def get_dashboard_summary():
    if not FEATURE_FILE.exists():
        raise HTTPException(status_code=500, detail="Feature file not found")
    
    try:
        # Optimization: Read file ONCE
        df = pd.read_csv(FEATURE_FILE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading features: {e}")

    # Pre-load inventory
    try:
        inventory_df = load_inventory()
    except:
        inventory_df = pd.DataFrame()

    return process_summary_dataframe(df, inventory_df)


@app.post("/dashboard/custom_summary")
def get_custom_summary(payload: List[Dict[str, Any]]):
    if not payload:
        raise HTTPException(status_code=400, detail="Empty payload")

    df = pd.DataFrame(payload)
    
    # Pre-process columns to support standard e-commerce dataset headers
    column_mapping = {
        "transaction_date": "date",
        "purchase_amount": "revenue",
        "quantity": "qty",
        "product_category": "category",
        "product category": "category"
        # Add more mappings here if needed
    }
    
    # Rename columns ignoring case and stripping leading/trailing whitespace
    df.columns = df.columns.astype(str).str.strip()
    
    new_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in column_mapping:
            new_cols.append(column_mapping[col_lower])
        else:
            new_cols.append(col)
    df.columns = new_cols

    if "sku" not in df.columns:
        # Better Fallback: If 'category' exists (mapped from Product_Category), use that as the SKU 
        # so we get individual time-series predictions per category instead of one massive blob
        if "category" in df.columns:
            df["sku"] = df["category"]
        else:
            df["sku"] = "Global_SKU"
        
    # If no 'qty' is provided but revenue is, we assume qty=1 for transaction datasets
    if "qty" not in df.columns and "revenue" in df.columns:
        df["qty"] = 1

    # Aggregate raw transaction data by sku and date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Ensure we only have dates (no time) for grouping
        df["date"] = df["date"].dt.normalize()
        
        # Drop rows with invalid dates
        df = df.dropna(subset=["date"])
        
        agg_funcs = {}
        if "qty" in df.columns: agg_funcs["qty"] = "sum"
        if "revenue" in df.columns: agg_funcs["revenue"] = "sum"
        
        if agg_funcs:
            # Group by SKU and Date, then sum up quantities/revenues
            df = df.groupby(["sku", "date"]).agg(agg_funcs).reset_index()
            # Sort by Date so Time Series features can be extracted properly
            df = df.sort_values(by=["sku", "date"])

    try:
        inventory_df = load_inventory()
    except:
        inventory_df = pd.DataFrame()

    return process_summary_dataframe(df, inventory_df)
