import matplotlib.pyplot as plt
import numpy as np

metrics = ['Inventory Turnover', 'Stockouts\n(per month)', 'Holding Cost\n($K/month)']
dqn_values = [13.50, 11.67, 0.10]
ref_values = [9.51,  3.02, 14.32]

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle('Double DQN vs. Reference - Inventory Metrics Bar Graph', fontsize=16, y=1.05)

colors = ['#2ca02c', '#1f77b4'] # Green for DQN, Blue for Reference

for i, ax in enumerate(axes):
    bars = ax.bar(['Double DQN', 'Reference'], [dqn_values[i], ref_values[i]], color=colors, width=0.5)
    ax.set_title(metrics[i])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    ax.margins(y=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('dqn_vs_reference_metrics.png', dpi=300, bbox_inches='tight')
print('Saved dqn_vs_reference_metrics.png')
