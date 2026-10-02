import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from functools import lru_cache

# Path to the model file
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / "xgb_model.joblib"

_MODELS = None

def load_models():
    global _MODELS
    if _MODELS is None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
        
        loaded = joblib.load(MODEL_FILE)
        
        # Handle both legacy single model and new ensemble list
        if isinstance(loaded, dict):
             _MODELS = loaded["model"]
        else:
             _MODELS = loaded
             
        # Normalize to list if it's a single booster
        if not isinstance(_MODELS, list):
            _MODELS = [_MODELS]
            
    return _MODELS

@lru_cache(maxsize=10000)
def cached_predict_demand(features_tuple):
    """
    Predict demand using the ensemble XGBoost model.
    features_tuple: A tuple of 27 features (must match training order).
    """
    models = load_models()
    
    # Convert input tuple to 2D numpy array (1, N_features)
    input_features = np.array(features_tuple, dtype=np.float32).reshape(1, -1)
    
    # Create DMatrix
    dmat = xgb.DMatrix(input_features)
    
    # Ensemble Prediction
    preds_log = []
    for model in models:
        # Each model predicts log(1+x)
        preds_log.append(model.predict(dmat)[0])
        
    # Average the log-predictions
    avg_log_pred = np.mean(preds_log)
    
    # Inverse transform and round
    final_pred = np.rint(np.expm1(avg_log_pred))
    
    return max(0.0, float(final_pred))
