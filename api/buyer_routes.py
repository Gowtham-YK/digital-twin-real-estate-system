from flask import Blueprint, request, jsonify
from utils.buyer.buyer_predictor import predict_price
from utils.buyer.heatmap import generate_heatmap

buyer_bp = Blueprint("buyer", __name__)

@buyer_bp.route("/buyer/predict", methods=["GET"])
def buyer_predict():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
        sqft = float(request.args.get("sqft"))
        bhk = int(request.args.get("bhk"))

        price = predict_price(lat, lon, sqft, bhk)
        heatmap = generate_heatmap(lat, lon, sqft, bhk)

        return jsonify({
            "predicted_price": price,
            "heatmap": heatmap
        })

    except Exception as e:
        return jsonify({"error": str(e)})