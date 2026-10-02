import pandas as pd
from pathlib import Path

# ======================
# PROJECT ROOT
# ======================
CUR_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in CUR_FILE.parents if p.name == "bizchainconnect")

IN = PROJECT_ROOT / "data" / "features" / "features_daily_by_sku.csv"
OUT = PROJECT_ROOT / "data" / "features" / "Time series features_ml.csv"

print("Loading:", IN)

df = pd.read_csv(
    IN, 
    parse_dates=['date']
)
df = df.sort_values(['sku', 'date'])

# ======================
# FEATURE ENGINEERING
# ======================
df['dow'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['weekend'] = (df['dow'] >= 5).astype(int)

# 1. Price & Revenue Features
# avg_unit_price: handle division by zero safely
df['avg_unit_price'] = df['revenue'] / (df['qty'].replace(0, 1))
df.loc[df['qty'] == 0, 'avg_unit_price'] = 0.0

# 2. Channel Features
df['online_ratio'] = df['qty_online'] / (df['qty'].replace(0, 1))
df.loc[df['qty'] == 0, 'online_ratio'] = 0.0

# 3. LAG FEATURES (per SKU)
# Expanded lags to capture short-term patterns better
lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
for lag in lags:
    df[f'lag_{lag}'] = df.groupby('sku')['qty'].shift(lag)

# 4. ROLLING FEATURES
# Mean trends
df['roll_3_mean'] = df.groupby('sku')['qty'].shift(1).rolling(3).mean()
df['roll_7_mean'] = df.groupby('sku')['qty'].shift(1).rolling(7).mean()
df['roll_14_mean'] = df.groupby('sku')['qty'].shift(1).rolling(14).mean()
df['roll_28_mean'] = df.groupby('sku')['qty'].shift(1).rolling(28).mean()

# Rolling Max/Min (volatility extremes)
df['roll_7_max'] = df.groupby('sku')['qty'].shift(1).rolling(7).max()
df['roll_7_min'] = df.groupby('sku')['qty'].shift(1).rolling(7).min()

# Volatility
df['roll_3_std'] = df.groupby('sku')['qty'].shift(1).rolling(3).std()
df['roll_7_std'] = df.groupby('sku')['qty'].shift(1).rolling(7).std()
df['roll_14_std'] = df.groupby('sku')['qty'].shift(1).rolling(14).std()

# Exponential Moving Averages (EMA)
df['ema_7'] = df.groupby('sku')['qty'].shift(1).ewm(span=7, adjust=False).mean()
df['ema_14'] = df.groupby('sku')['qty'].shift(1).ewm(span=14, adjust=False).mean()
df['ema_28'] = df.groupby('sku')['qty'].shift(1).ewm(span=28, adjust=False).mean()

# Interaction & Relative Features
df['revenue_lag_1'] = df.groupby('sku')['revenue'].shift(1)

# Revenue Volatility
df['revenue_roll_7_std'] = df.groupby('sku')['revenue'].shift(1).rolling(7).std()

# Price features (detect discounts)
df['price_roll_7'] = df.groupby('sku')['avg_unit_price'].rolling(7).mean().reset_index(0, drop=True)
df['price_ratio'] = df['avg_unit_price'] / (df['price_roll_7'] + 1e-6)
df['price_change_7'] = df['avg_unit_price'] - df['price_roll_7']

# Volatility Ratio (is volatility increasing?)
df['vol_ratio'] = df['roll_3_std'] / (df['roll_14_std'] + 1e-6)

# Date Cyclic Features
import numpy as np
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 53)
df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 53)

# Trend Features
df['trend_ratio_7_28'] = df['roll_7_mean'] / (df['roll_28_mean'] + 1e-6)
df['diff_7_14'] = df['roll_7_mean'] - df['roll_14_mean']
df['diff_1'] = df['qty'].shift(1) - df['qty'].shift(2) # diff of lags to avoid leakage

# TARGET (predict next 7 days demand)
df['target_7'] = df.groupby('sku')['qty'].shift(-7)

# ======================
# CLEAN NA ROWS
# ======================
df = df.dropna()

# ======================
# SAVE
# ======================
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("Saved ML features to:", OUT)
print("Final dataset shape:", df.shape)
