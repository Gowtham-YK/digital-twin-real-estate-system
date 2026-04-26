import numpy as np
from utils.buyer.buyer_predictor import predict_price

# =========================
# LAKE FILTER (IMPROVED)
# =========================
def is_in_lake(lat, lon):
    # Madiwala Lake (expanded bounds)
    if 12.900 < lat < 12.920 and 77.620 < lon < 77.640:
        return True
    return False


# =========================
# GENERATE POINTS (FIXED)
# =========================
def generate_heatmap(lat, lon, sqft, bhk, radius=0.01, num_points=100):
    points = []
    attempts = 0

    while len(points) < num_points and attempts < num_points * 5:
        attempts += 1

        angle = np.random.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1))

        new_lat = lat + r * np.cos(angle)
        new_lon = lon + r * np.sin(angle)

        # ❌ Skip lakes
        if is_in_lake(new_lat, new_lon):
            continue

        price = predict_price(new_lat, new_lon, sqft, bhk)

        points.append({
            "lat": new_lat,
            "lon": new_lon,
            "price": price
        })

    return points