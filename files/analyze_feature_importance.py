import joblib
import pandas as pd
from pathlib import Path

OUT = Path("models/xgb_model.joblib")
data = joblib.load(OUT)

models = data['model']
features = data['features']

importance = models[0].get_score(importance_type='gain')

importance_mapped = {}
for k, v in importance.items():
    idx = int(k[1:])
    if idx < len(features):
        importance_mapped[features[idx]] = v
    else:
        importance_mapped[k] = v

importance_df = pd.DataFrame({
    'Feature': importance_mapped.keys(),
    'Importance': importance_mapped.values()
}).sort_values('Importance', ascending=False)

with open('feature_importance_output.txt', 'w') as f:
    f.write(importance_df.head(20).to_string())
