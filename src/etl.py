import pandas as pd
from pathlib import Path


def prepare_features(sales_df, out_path, freq='D'):
    """
    sales_df: DataFrame containing at least ['date','sku','qty','price','order_id']
    out_path: Path to write the expanded time-series CSV
    freq: 'D' for daily, 'W' for weekly
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()  # remove old file

    print("ETL started...")

    # 1) Clean & optimize memory
    sales_df = sales_df.copy()
    sales_df['date'] = pd.to_datetime(sales_df['date'], errors='coerce')
    sales_df = sales_df.dropna(subset=['date', 'sku'])

    sales_df['qty'] = pd.to_numeric(
        sales_df.get('qty', 1), downcast='integer'
    ).fillna(0).astype('int32')

    sales_df['price'] = pd.to_numeric(
        sales_df.get('price', 0.0), downcast='float'
    ).fillna(0.0).astype('float32')

    sales_df['sku'] = sales_df['sku'].astype('category')

    # 2) Date range
    min_date = sales_df['date'].min()
    max_date = sales_df['date'].max()
    dates = pd.date_range(start=min_date, end=max_date, freq=freq)

    # 3) Process per SKU (memory safe)
    skus = sales_df['sku'].cat.categories.tolist()
    header_written = False

    for i, sku in enumerate(skus):
        grp = sales_df[sales_df['sku'] == sku].set_index('date')

        # Resample daily totals
        qty_series = grp['qty'].resample(freq).sum().reindex(dates, fill_value=0)
        
        # Correct Revenue Calculation: Sum of price column (assuming price in sales.csv is line total)
        # If price is unit price, we should multiply by qty first. 
        # Checking sales.csv sample: qty=4, price=120.51. Hard to tell if unit or total.
        # However, standard ETL usually implies 'price' in sales data is often transaction amount or needs multiplication.
        # Let's assume 'price' column in `sales.csv` is the total line item amount (revenue) based on the user request context 
        # "revenue calculation... was flawed (squared instead of linear)". 
        # Previous code: price_series = grp['price'].resample(freq).sum() -> then feature eng: qty * price_series.
        # If 'price' in sales.csv is unit price, then `grp['price'].sum()` is sum of unit prices (meaningless).
        # We need to calculate line revenue first before resampling.
        
        # Better approach: Calculate revenue at transaction level first
        grp['line_revenue'] = grp['qty'] * grp['price'] # Assuming price is unit price. 
        # If price was already total, this would be wrong. 
        # But commonly `price` = unit price. Let's calculate typical unit price to check.
        # In sample: qty=4, price=120.51 -> unit=30.12. qty=7, price=161.62 -> unit=23.08. 
        # These look like realistic unit prices, or `price` is total?
        # If price is total: 120.51 / 4 = 30.12. 
        # Let's stick to the plan: "Rename aggregated price to revenue". 
        # If the user says "squared instead of linear", they imply we were doing (Sum of Unit Prices) * (Sum of Qty), which is wrong.
        # We want Sum of (Unit Price * Qty) OR Sum of (Total Line Amount).
        # Let's assume `price` in sales.csv is UNIT PRICE.
        
        # Calculate revenue for each transaction
        # NOTE: Modifying the input DataFrame `sales_df` or `grp` before resampling
        
        revenue_series = (grp['qty'] * grp['price']).resample(freq).sum().reindex(dates, fill_value=0)
        
        # Channel splits
        grp_online = grp[grp['channel'] == 'online']
        grp_retail = grp[grp['channel'] == 'retail'] # explicit check for safety
        
        qty_online = grp_online['qty'].resample(freq).sum().reindex(dates, fill_value=0)
        qty_retail = grp_retail['qty'].resample(freq).sum().reindex(dates, fill_value=0)

        order_count = grp['order_id'].resample(freq).nunique().reindex(dates, fill_value=0)

        df_out = pd.DataFrame({
            'date': dates,
            'sku': sku,
            'qty': qty_series.values.astype('int32'),
            'revenue': revenue_series.values.astype('float32'),
            'qty_online': qty_online.values.astype('int32'),
            'qty_retail': qty_retail.values.astype('int32'),
            'order_count': order_count.values.astype('int32')
        })

        df_out.to_csv(
            out_path,
            mode='w' if not header_written else 'a',
            index=False,
            header=not header_written
        )

        header_written = True

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(skus)} SKUs")

    print(f"ETL completed successfully → {out_path}")


# ======================
# SCRIPT ENTRY POINT
# ======================
if __name__ == "__main__":
    INPUT_FILE = Path("data/sales.csv")
    OUTPUT_FILE = Path("data/features/features_daily_by_sku.csv")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    sales_df = pd.read_csv(INPUT_FILE)

    prepare_features(
        sales_df,
        out_path=OUTPUT_FILE,
        freq="D"   # change to "W" for weekly
    )
