import matplotlib.pyplot as plt
import numpy as np

# Data
phases = [
    'Phase 1: Baseline\nModel', 
    'Phase 2: Feature\nEngineering', 
    'Phase 3: Hyperparameter\nOptimization', 
    'Phase 4: Final tuned\nEnsemble'
]
accuracies = [94.80, 96.10, 97.50, 98.45]

fig, ax = plt.subplots(figsize=(10, 6))

# Create line plot with markers
line = ax.plot(phases, accuracies, marker='o', markersize=10, linestyle='-', linewidth=3, color='#1f77b4')

# Fill area under the curve
ax.fill_between(phases, accuracies, alpha=0.1, color='#1f77b4')

# Customizing the plot
ax.set_ylim(93, 100) # Give some padding
ax.set_ylabel('Forecast Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Accuracy Improvement Across Development Phases', fontsize=16, pad=20)
ax.grid(True, linestyle='--', alpha=0.7)

# Add data labels
for i, v in enumerate(accuracies):
    ax.annotate(f"{v:.2f}%", 
                xy=(i, v), 
                xytext=(0, 10), 
                textcoords="offset points",
                ha='center', va='bottom',
                fontweight='bold', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#1f77b4', alpha=0.9))

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

output_file = 'accuracy_across_phases.png'
plt.savefig(output_file, dpi=300)
print(f"Generated {output_file}")
