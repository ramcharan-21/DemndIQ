import matplotlib.pyplot as plt
import numpy as np

# Data extracted from metrics.md
categories = ['Overall Demand', 'High Volatility SKUs', 'Low Volatility SKUs']
r2_scores = [0.595, 0.583, 0.583]
mae_scores = [2.142, 2.231, 2.053]
rmse_scores = [2.764, 2.870, 2.653]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width, r2_scores, width, label='R2 Score', color='#4C72B0')
rects2 = ax.bar(x, mae_scores, width, label='MAE', color='#DD8452')
rects3 = ax.bar(x + width, rmse_scores, width, label='RMSE', color='#55A868')

ax.set_ylabel('Metric Value')
ax.set_title('Comparison of XGBoost Model Performance by Sub-Configurations')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()

plt.savefig('xgboost_configurations_comparison.png', dpi=300)
print("Saved xgboost_configurations_comparison.png")
