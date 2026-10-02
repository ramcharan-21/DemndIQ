import pandas as pd
import matplotlib.pyplot as plt

# Load the feature importance data
df = pd.read_csv('feature_importance_output.txt', delim_whitespace=True, skiprows=1, names=['Index', 'Feature', 'Importance'])

# Sort values to display highest importance at the top of the plot
df = df.sort_values(by='Importance', ascending=True)

# Create the plot
plt.figure(figsize=(10, 8))
plt.barh(df['Feature'], df['Importance'], color='#2ca02c')

# Add labels and title
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('XGBoost Feature Importance - Top Predictors of Demand')

# Add grid and adjust layout
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
output_file = 'xgboost_feature_importance.png'
plt.savefig(output_file, dpi=300)
print(f"Saved {output_file}")
