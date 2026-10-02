# src/db.py  -- load Olist dataset into your BizChain DB structure
from pathlib import Path
from sqlalchemy import create_engine, text
import pandas as pd

DB_FILE = Path('data/bizchain.db')
DB_FILE.parent.mkdir(parents=True, exist_ok=True)  # ensure /data exists
DB_URL = f"sqlite:///{DB_FILE.as_posix()}"

def init_db_and_load():
    engine = create_engine(DB_URL, echo=False)

    # ---------- 1) CREATE INTERNAL TABLES ----------
    create_statements = [
        '''CREATE TABLE IF NOT EXISTS product_master (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT UNIQUE,
            category TEXT,
            lead_time_days INTEGER,
            supplier_id TEXT
        );''',

        '''CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            sku TEXT,
            qty INTEGER,
            price REAL,
            promo_flag INTEGER,
            channel TEXT,
            order_id TEXT
        );''',

        '''CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            sku TEXT,
            warehouse TEXT,
            on_hand INTEGER,
            reserved INTEGER
        );''',

        '''CREATE TABLE IF NOT EXISTS lead_times (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT,
            supplier_id TEXT,
            lead_time_days INTEGER
        );''',

        '''CREATE TABLE IF NOT EXISTS purchase_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT,
            qty INTEGER,
            created_at TEXT,
            status TEXT
        );'''
    ]

    with engine.begin() as conn:
        for stmt in create_statements:
            conn.execute(text(stmt))

    # ---------- 2) LOAD RAW OLIST CSVs ----------
    base = Path('data')
    df_orders = pd.read_csv(base / 'olist_orders_dataset.csv')
    df_orditems = pd.read_csv(base / 'olist_order_items_dataset.csv')
    df_products = pd.read_csv(base / 'olist_products_dataset.csv')

    # Replace raw tables for clean reload
    df_orders.to_sql('orders', engine, if_exists='replace', index=False)
    df_orditems.to_sql('orditems', engine, if_exists='replace', index=False)
    df_products.to_sql('products', engine, if_exists='replace', index=False)

    print("Raw Olist tables loaded.")

    # ---------- 3) BUILD product_master ----------
    df_pm = df_products[['product_id', 'product_category_name']].copy()
    df_pm = df_pm.rename(columns={
        'product_id': 'sku',
        'product_category_name': 'category'
    })
    df_pm['lead_time_days'] = 7     # default placeholder
    df_pm['supplier_id'] = 'SUP001' # placeholder supplier

    df_pm.to_sql('product_master', engine, if_exists='replace', index=False)
    print("product_master populated.")

    # ---------- 4) BUILD sales table ----------
    df_sales = df_orditems.merge(
        df_orders[['order_id', 'order_purchase_timestamp']],
        on='order_id',
        how='left'
    )

    df_sales = df_sales.rename(columns={
        'order_purchase_timestamp': 'date',
        'product_id': 'sku',
        'price': 'price'
    })

    df_sales['qty'] = 1                     # each row = 1 item sold
    df_sales['promo_flag'] = 0              # Olist has no promo data
    df_sales['channel'] = 'online'          # Olist = online marketplace

    df_sales = df_sales[['date', 'sku', 'qty', 'price', 'promo_flag', 'channel', 'order_id']]

    df_sales.to_sql('sales', engine, if_exists='replace', index=False)
    print("sales table populated.")

    print('Database ready:', DB_FILE)


if __name__ == '__main__':
    init_db_and_load()
