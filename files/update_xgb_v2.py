import re

with open('src/train_xgb.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Revert to target_smooth
target_block = '''df["target_smooth"] = (
    df["target_7"]
    .rolling(window=7, min_periods=1)
    .mean()
)

df["target_log"] = np.log1p(df["target_smooth"])'''

content = re.sub(
    r'# Predict actual raw target_7 instead of smoothed mean\s*df\["target_log"\] = np\.log1p\(df\["target_7"\]\)', 
    target_block, 
    content
)

# 2. Add 'qty' (today's demand) to features list
features_clean = '''features = [
    "sku_id",
    "qty",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7",
    "lag_14", "lag_21", "lag_28", "lag_30", "lag_60",
    "roll_3_mean", "roll_7_mean", "roll_14_mean", "roll_28_mean", "roll_30_mean",
    "roll_7_max", "roll_7_min",
    "roll_3_std", "roll_7_std", "roll_14_std", "roll_30_std",
    "ema_7", "ema_28",
    "avg_unit_price", "online_ratio",
    "revenue_lag_1", "revenue_roll_7_std",
    "price_ratio", "price_change_7", "vol_ratio",
    "trend_ratio_7_28", "diff_7_14", "diff_1",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "week_sin", "week_cos",
    "weekend"
]'''

content = re.sub(r'features = \[.*?\]', features_clean, content, flags=re.DOTALL)

# 3. Increase search iter to 20, remove np.rint
content = content.replace('n_iter = 10', 'n_iter = 20')
content = content.replace('y_pred = np.rint(np.expm1(preds))', 'y_pred = np.expm1(preds)')
content = content.replace('y_pred = np.rint(np.expm1(y_pred_log))', 'y_pred = np.expm1(y_pred_log)')

with open('src/train_xgb.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated train_xgb.py with qty feature, target_smooth, and 20 iters without rint.')
