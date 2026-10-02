
# src/reorder.py  -- fixed clean version
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
import joblib
import xgboost as xgb
import numpy as np

# SQLite DB path
DB_FILE = Path('data/bizchain.db')
DB_URL = f"sqlite:///{DB_FILE.as_posix()}"
engine = create_engine(DB_URL, echo=False)

from pathlib import Path

CUR_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in CUR_FILE.parents if p.name == "bizchainconnect")

MODEL_PATH = PROJECT_ROOT / "models" / "xgb_model.joblib"
xgb_obj = joblib.load(MODEL_PATH)
xgb_model = xgb_obj['model']
features = xgb_obj['features']

def compute_reorder(sku: str, safety_days=7, lead_days=7):
    # Inventory
    inv = pd.read_sql(
        'SELECT date, on_hand FROM inventory WHERE sku = ? ORDER BY date DESC LIMIT 1',
        con=engine, params=(sku,)
    )
    if inv.empty:
        return {"error": "SKU not found in inventory", "sku": sku}

    on_hand = int(inv.iloc[0]['on_hand'])

    # Sales
    sales = pd.read_sql(
        'SELECT date, qty FROM sales WHERE sku = ? ORDER BY date',
        con=engine, params=(sku,)
    )
    sales['date'] = pd.to_datetime(sales['date'])
    sales = sales.set_index('date').resample('D').sum().fillna(0).reset_index()

    # Feature engineering
    sales['lag_1'] = sales['qty'].shift(1).fillna(0)
    sales['lag_7'] = sales['qty'].shift(7).fillna(0)
    sales['roll_7_mean'] = sales['qty'].rolling(7, min_periods=1).mean()
    sales['roll_14_mean'] = sales['qty'].rolling(14, min_periods=1).mean()
    sales['dow'] = sales['date'].dt.weekday

    last = sales.iloc[-1]
    feat_vector = [last[c] for c in features]

    # Predict demand
    d = xgb.DMatrix(np.array([feat_vector]), feature_names=None)
    pred = float(xgb_model.predict(d)[0])

    # Baseline demand stats
    avg_daily = float(sales['qty'].tail(30).mean())

    # Simple reorder logic
    action = np.argmax(q_table[current_inventory, predicted_demand])
    order_qty = action * 10
    print("RL decided order quantity:", order_qty)


    return {
        "sku": sku,
        "on_hand": on_hand,
        "forecast_next_7_days": pred,
        "avg_daily_sales": avg_daily,
        "safety_stock": safety_stock,
        "suggested_order_quantity": suggested_qty
    }

if __name__ == "__main__":
    df = pd.read_sql("SELECT sku FROM product_master LIMIT 5", con=engine)
    for s in df['sku'].tolist():
        print(compute_reorder(s))
