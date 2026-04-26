import os
import joblib
import numpy as np

# =========================
# LOAD MODEL SAFELY
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
MODEL_PATH = os.path.join(BASE_DIR, "models", "buyer_model", "buyer_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print("✅ Buyer model loaded successfully")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_price(lat, lon, sqft, bhk):
    try:
        features = np.array([[lat, lon, sqft, bhk]])
        prediction = model.predict(features)[0]
        return float(prediction)
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")