import os
import joblib

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "seller_model", "seller_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print("✅ Seller model loaded")


# =========================
# PREDICT FUTURE PRICES
# =========================
def predict_future_prices(lat, lon, sqft, bhk, bought_price):
    # Ensure inputs are float
    bought_price = float(bought_price)

    # =========================
    # 1Y prediction (from model)
    # =========================
    features = [[float(lat), float(lon), float(sqft), int(bhk), bought_price, 1]]

    price_1y = float(model.predict(features)[0])  # ✅ FIXED

    # =========================
    # Calculate growth
    # =========================
    growth = float((price_1y - bought_price) / bought_price)  # ✅ FIXED

    # =========================
    # Apply compounding
    # =========================
    time_map = {
        "1y": 1,
        "5y": 5,
        "10y": 10
    }

    results = {}

    for key, t in time_map.items():
        price = bought_price * ((1 + growth) ** t)
        results[key] = float(price)  # ✅ FIXED

    return float(growth), results  # ✅ Ensure JSON safe