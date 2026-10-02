import numpy as np
from sklearn.metrics import r2_score

# Simulate the effect of rint
np.random.seed(42)
y_true = np.random.uniform(35, 50, 1000)
# Let's say model is perfect
y_pred_perfect = y_true.copy()
# Now add the rint
y_pred_rinted = np.rint(y_pred_perfect)

print("R2 without rint:", r2_score(y_true, y_pred_perfect))
print("R2 with rint:", r2_score(y_true, y_pred_rinted))
