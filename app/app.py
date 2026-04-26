import sys
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# =========================
# FIX IMPORT PATH (ONLY ONCE)
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# =========================
# IMPORT MODULES
# =========================
from utils.geo_coder import geocode_location
from utils.buyer.buyer_predictor import predict_price
from utils.buyer.heatmap import generate_heatmap
from utils.seller.seller_predictor import predict_future_prices

# =========================
# INIT APP
# =========================
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

print("✅ Real Estate ML System Running")

# =========================
# 🏠 HOME
# =========================
@app.route('/')
def home():
    return render_template('index.html')

# =========================
# 🧑‍💼 BUYER PAGE
# =========================
@app.route('/buyer')
def buyer_page():
    return render_template('buyer.html')

# =========================
# 🧑‍💼 SELLER PAGE
# =========================
@app.route('/seller')
def seller_page():
    return render_template('seller.html')

# =========================
# 📄 ABOUT PAGE
# =========================
@app.route('/about')
def about_page():
    return render_template('about.html')


# =========================
# 📄 CONTACT PAGE
# =========================
@app.route('/contact')
def contact_page():
    return render_template('contact.html')

# =========================
# 🔮 BUYER API
# =========================
@app.route('/api/buyer/predict', methods=['POST'])
def buyer_predict():
    try:
        data = request.get_json()

        location = data.get("location")
        sqft = float(data.get("sqft"))
        bhk = int(data.get("bhk"))

        if not location:
            return jsonify({"status": "error", "message": "Location required"}), 400

        # Geocode
        lat, lon = geocode_location(location)

        # Predict
        price = predict_price(lat, lon, sqft, bhk)
        heatmap = generate_heatmap(lat, lon, sqft, bhk)

        return jsonify({
            "status": "success",
            "location": location,
            "lat": lat,
            "lon": lon,
            "predicted_price": price,
            "heatmap": heatmap
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================
# 🔮 SELLER API
# =========================
@app.route('/api/seller/predict', methods=['POST'])
def seller_predict():
    try:
        data = request.get_json()

        location = data.get("location")
        sqft = float(data.get("sqft"))
        bhk = int(data.get("bhk"))
        bought_price = float(data.get("bought_price"))

        if not location:
            return jsonify({"status": "error", "message": "Location required"}), 400

        # Geocode
        lat, lon = geocode_location(location)

        # Predict future prices
        growth, predictions = predict_future_prices(
            lat, lon, sqft, bhk, bought_price
        )

        return jsonify({
            "status": "success",
            "location": location,
            "growth_rate": growth,
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================
# 🔗 ANYLOGIC SIMULATION API
# =========================
@app.route('/api/simulation/predict', methods=['POST'])
def simulation_predict():
    try:
        data = request.get_json()

        # 🔥 DEBUG LOGS (ONLY ADDED)
        print("🔥 ANYLOGIC DATA:", data)

        demand = float(data.get("demand", 0))
        interest = float(data.get("interest", 0))
        supply = float(data.get("supply", 0))
        growth = float(data.get("growth", 0))
        economy = float(data.get("economy", 0))

        # simple dynamic logic
        price_change = (
            demand - interest - 0.5 * supply + economy + growth * 0.5
        ) * 1000000   # scale for visible graph

        # 🔥 DEBUG OUTPUT
        print("📈 Price Change:", price_change)

        return jsonify({
            "status": "success",
            "price_change": price_change
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================
# 🔍 TEST CONNECTION API
# =========================
@app.route('/api/test-anylogic', methods=['GET'])
def test_anylogic():
    print("🔥 Website triggered test API")
    return jsonify({"message": "API working"})

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=5000)