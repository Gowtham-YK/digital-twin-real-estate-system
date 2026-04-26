"""
predict_helper.py  —  drop this file into your Flask project root.

Usage in app.py:
    from predict_helper import load_artifacts, predict_price

    artifacts = load_artifacts()       # call ONCE at startup (outside route)

    @app.route('/predict', methods=['POST'])
    def predict():
        data  = request.get_json()
        price = predict_price(artifacts, data)
        return jsonify({'predicted_price': price})

Expected JSON body:
    {
        "location": "Whitefield",
        "area":      1200,
        "bedrooms":  3,
        "bath":      2,
        "balcony":   1
    }
"""

import joblib
import numpy as np
import pandas as pd


def load_artifacts(model_dir: str = "models") -> dict:
    return {
        "model":                    joblib.load(f"{model_dir}/xgb_model_real.pkl"),
        "feature_cols":             joblib.load(f"{model_dir}/feature_cols.pkl"),
        "location_mapping":         joblib.load(f"{model_dir}/location_mapping.pkl"),
        "reverse_location_mapping": joblib.load(f"{model_dir}/reverse_location_mapping.pkl"),
        "location_target_map":      joblib.load(f"{model_dir}/location_target_map.pkl"),
        "global_mean":              joblib.load(f"{model_dir}/location_global_mean.pkl"),
        "location_ppsf_map":        joblib.load(f"{model_dir}/location_ppsf_map.pkl"),
        "global_ppsf":              joblib.load(f"{model_dir}/location_global_ppsf.pkl"),
    }


def _area_bin(area: float) -> int:
    """
    Approximate 5-quantile bin matching training qcut.
    After retraining, get exact boundaries with:
        cuts = pd.qcut(df['area'], q=5, retbins=True, duplicates='drop')[1]
        print(cuts)
    """
    if   area < 650:  return 0
    elif area < 800:  return 1
    elif area < 1000: return 2
    elif area < 1350: return 3
    else:              return 4


def predict_price(artifacts: dict, data: dict) -> float:
    feature_cols             = artifacts["feature_cols"]
    model                    = artifacts["model"]
    reverse_location_mapping = artifacts["reverse_location_mapping"]
    location_target_map      = artifacts["location_target_map"]
    global_mean              = artifacts["global_mean"]
    location_ppsf_map        = artifacts["location_ppsf_map"]
    global_ppsf              = artifacts["global_ppsf"]

    location = str(data.get("location", "other")).strip()
    area     = float(data["area"])
    bedrooms = float(data["bedrooms"])
    bath     = float(data["bath"])
    balcony  = float(data.get("balcony", 0))

    if location not in location_target_map:
        location = "other"

    loc_label        = reverse_location_mapping.get(location, 0)
    loc_encoded      = location_target_map.get(location, global_mean)
    area_bin         = _area_bin(area)
    bed_bath_ratio   = bedrooms / (bath + 1e-5)
    total_rooms      = bedrooms + bath + balcony
    area_per_bedroom = area / (bedrooms + 1e-5)
    price_per_sqft   = location_ppsf_map.get(location, global_ppsf)

    row = {
        "location_encoded": loc_encoded,
        "location_label":   loc_label,
        "area":             area,
        "area_bin":         area_bin,
        "bedrooms":         bedrooms,
        "bath":             bath,
        "balcony":          balcony,
        "price_per_sqft":   price_per_sqft,
        "bed_bath_ratio":   bed_bath_ratio,
        "total_rooms":      total_rooms,
        "area_per_bedroom": area_per_bedroom,
    }

    df       = pd.DataFrame([row])[feature_cols]
    log_pred = model.predict(df)
    return round(float(np.expm1(log_pred[0])), 2)