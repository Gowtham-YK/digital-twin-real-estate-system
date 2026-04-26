from flask import request, jsonify
import pandas as pd
import joblib

# Load model once (top of file)
model = joblib.load("models/xgb_model_real.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # 🔹 Location mapping (temporary fix)
        location_map = {
            "Whitefield": 1,
            "Indiranagar": 2,
            "Electronic City": 3,
            "BTM": 4
        }

        # Get location code (default = 0 if not found)
        loc = location_map.get(data['location'], 0)

        # Create dataframe EXACTLY like training
        df = pd.DataFrame([[
            loc,
            float(data['area']),
            float(data['bedrooms']),
            float(data['bath']),
            float(data['balcony'])
        ]], columns=['location', 'area', 'bedrooms', 'bath', 'balcony'])

        # 🔮 Predict
        prediction = model.predict(df)[0]

        return jsonify({
            "predicted_price": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })