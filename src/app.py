

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import xgboost as xgb
from datetime import datetime

# SQLite DB connection
DB_FILE = Path('data/bizchain.db')
DB_URL = f"sqlite:///{DB_FILE.as_posix()}"
engine = create_engine(DB_URL, echo=False)

# Load model
MODEL_PATH = Path('models/xgb_model.joblib')
obj = joblib.load(MODEL_PATH)
xgb_model = obj['model']
features = obj['features']

app = FastAPI(title="BizChainConnect API")

class Order(BaseModel):
    order_id: str
    date: str
    sku: str
    qty: int
    channel: str = "online"

@app.post("/orders")
def ingest_order(o: Order):
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO sales (date, sku, qty, price, promo_flag, channel, order_id) "
                "VALUES (:date, :sku, :qty, 0, 0, :channel, :order_id)"
            ),
            {
                "date": o.date,
                "sku": o.sku,
                "qty": o.qty,
                "channel": o.channel,
                "order_id": o.order_id
            }
        )
        res = conn.execute(
            text("SELECT on_hand FROM inventory WHERE sku=:sku ORDER BY date DESC LIMIT 1"),
            {"sku": o.sku}
        ).fetchone()
        last_on_hand = res[0] if res else 0
        new_on_hand = max(0, last_on_hand - o.qty)
        conn.execute(
            text(
                "INSERT INTO inventory (date, sku, warehouse, on_hand, reserved) "
                "VALUES (:date, :sku, 'WH1', :on_hand, 0)"
            ),
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "sku": o.sku,
                "on_hand": new_on_hand
            }
        )
    return {"status": "ok", "updated_on_hand": new_on_hand}

@app.get("/forecast/{sku}")
def forecast(sku: str, horizon: int = 7):
    df = pd.read_sql(
        "SELECT date, qty FROM sales WHERE sku = ? ORDER BY date",
        con=engine,
        params=(sku,)
    )
    if df.empty:
        raise HTTPException(status_code=404, detail="SKU not found")
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').resample('D').sum().fillna(0).reset_index()
    df['lag_1'] = df['qty'].shift(1).fillna(0)
    df['lag_7'] = df['qty'].shift(7).fillna(0)
    df['roll_7_mean'] = df['qty'].rolling(7, min_periods=1).mean()
    df['roll_14_mean'] = df['qty'].rolling(14, min_periods=1).mean()
    df['dow'] = df['date'].dt.weekday
    last = df.iloc[-1]
    feat = np.array([[last[c] for c in features]])
    pred = float(xgb_model.predict(xgb.DMatrix(feat))[0])
    return {"sku": sku, "horizon": horizon, "predicted_qty": pred}

@app.get("/reorder/{sku}")
def reorder_api(sku: str):
    # import reorder with package-aware fallback so this works whether running as `uvicorn src.app:app`
    try:
        # when running as package (recommended)
        from src.reorder import compute_reorder
    except Exception:
        # fallback if running as plain script
        from reorder import compute_reorder
    return compute_reorder(sku)

# --- lightweight HTML dashboard served by FastAPI (no extra installs) ---
from fastapi.responses import HTMLResponse

@app.get("/dashboard", response_class=HTMLResponse)
def simple_dashboard():
    # simple single-page app that fetches forecast & reorder from your API
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>BizChainConnect — Mini Dashboard</title>
      <style>
        body{font-family: Arial; padding:20px; max-width:900px}
        .card{border:1px solid #ddd;padding:12px;border-radius:8px;margin-bottom:12px}
        .row{display:flex;gap:12px;align-items:center}
      </style>
    </head>
    <body>
      <h2>BizChainConnect — Mini Dashboard</h2>
      <div>
        <label>SKU: <input id="sku" value="SKU_000" /></label>
        <button onclick="load()">Get Forecast & Reorder</button>
      </div>
      <div id="out"></div>
      <script>
        async function load(){
          const sku = document.getElementById('sku').value;
          const out = document.getElementById('out');
          out.innerHTML = '<div class="card">Loading...</div>';
          try{
            const f = await fetch(`/forecast/${sku}?horizon=7`);
            const fore = await f.json();
            const r = await fetch(`/reorder/${sku}`);
            const ro = await r.json();
            out.innerHTML = `
              <div class="card"><strong>Forecast (7 days)</strong><pre>${JSON.stringify(fore,null,2)}</pre></div>
              <div class="card"><strong>Reorder Suggestion</strong><pre>${JSON.stringify(ro,null,2)}</pre></div>
            `;
          }catch(e){
            out.innerHTML = '<div class="card" style="color:red">Error: '+e.toString()+'</div>';
          }
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

