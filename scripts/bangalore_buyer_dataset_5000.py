import pandas as pd
import random
import os

# =========================
# SETTINGS
# =========================
NUM_SAMPLES = 5000

# Bangalore zones with realistic price ranges
zones = [
    {"name": "Whitefield", "lat": 12.9698, "lng": 77.7499, "price": 5500},
    {"name": "Indiranagar", "lat": 12.9719, "lng": 77.6412, "price": 12000},
    {"name": "BTM", "lat": 12.9166, "lng": 77.6101, "price": 7000},
    {"name": "Electronic City", "lat": 12.8456, "lng": 77.6603, "price": 4500},
    {"name": "Hebbal", "lat": 13.0358, "lng": 77.5970, "price": 8000},
    {"name": "Jayanagar", "lat": 12.9250, "lng": 77.5938, "price": 11000}
]

data = []

# =========================
# GENERATE BUYER DATA
# =========================
for i in range(NUM_SAMPLES):

    zone = random.choice(zones)

    # Location preference (slight variation)
    lat = zone["lat"] + random.uniform(-0.01, 0.01)
    lng = zone["lng"] + random.uniform(-0.01, 0.01)

    sqft = random.randint(600, 4000)
    bhk = random.randint(1, 5)

    base_price_sqft = zone["price"]

    # Buyer budget (slightly lower/higher than market)
    budget_price = sqft * base_price_sqft * random.uniform(0.8, 1.2)

    data.append({
        "buyer_id": i + 1,
        "preferred_location": zone["name"],
        "latitude": round(lat, 6),
        "longitude": round(lng, 6),
        "sqft": sqft,
        "BHK": bhk,
        "budget_price": int(budget_price)
    })

# =========================
# CREATE DATAFRAME
# =========================
df = pd.DataFrame(data)

# =========================
# SAVE FILE
# =========================
file_path = os.path.abspath("bangalore_buyer_dataset_5000.csv")
df.to_csv(file_path, index=False)

# =========================
# OUTPUT
# =========================
print("✅ Buyer dataset created successfully!")
print("📁 Saved at:", file_path)
print("📊 Shape:", df.shape)
print(df.head())