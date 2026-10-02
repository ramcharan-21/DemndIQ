import joblib
import joblib
xgb_model = joblib.load("models/xgb_model.joblib")
joblib.dump(xgb_model, "models/xgb_model.joblib")
# Make sure the input features match what the model was trained on
import pandas as pd
sample = pd.DataFrame({
    "lag_1": [12],
    "lag_7": [10],
    "roll_7_mean": [11],
    "roll_14_mean": [10.5],
    "dow": [2],
    "price": [45.0]
})

prediction = xgb_model.predict(sample)
print("Forecasted demand:", prediction[0])





