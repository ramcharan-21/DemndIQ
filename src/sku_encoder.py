import pandas as pd
from pathlib import Path

CUR_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in CUR_FILE.parents if p.name == "bizchainconnect")
DATA_PATH = PROJECT_ROOT / "data" / "features" / "Time series features_ml.csv"

def load_sku_map():
    df = pd.read_csv(DATA_PATH)
    skus = sorted(df["sku"].unique())
    sku_to_id = {sku: i for i, sku in enumerate(skus)}
    id_to_sku = {i: sku for sku, i in sku_to_id.items()}
    max_id = max(sku_to_id.values())
    return sku_to_id, id_to_sku, max_id
