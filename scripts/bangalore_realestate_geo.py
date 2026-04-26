import pandas as pd
import numpy as np
import random
import os

# =========================
# SETTINGS
# =========================
NUM_SAMPLES = 5000

# Bangalore zones (with realistic price per sqft)
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
# GENERATE DATA
# =========================
for i in range(NUM_SAMPLES):

    zone = random.choice(zones)

    # Coordinates (slight variation)
    lat = zone["lat"] + random.uniform(-0.01, 0.01)
    lng = zone["lng"] + random.uniform(-0.01, 0.01)

    # Property features
    sqft = random.randint(600, 4000)
    bhk = random.randint(1, 5)

    # Years
    buy_year = random.randint(2015, 2022)
    sell_year = buy_year + random.randint(1, 5)

    # Price calculation
    base_price_sqft = zone["price"]

    buy_price = sqft * base_price_sqft * random.uniform(0.9, 1.1)

    growth_rate = random.uniform(0.05, 0.12)
    years_held = sell_year - buy_year

    sell_price = buy_price * ((1 + growth_rate) ** years_held)

    data.append({
        "property_id": i + 1,
        "location": zone["name"],
        "latitude": round(lat, 6),
        "longitude": round(lng, 6),
        "sqft": sqft,
        "BHK": bhk,
        "buy_year": buy_year,
        "sell_year": sell_year,
        "buy_price": int(buy_price),
        "sell_price": int(sell_price)
    })

# =========================
# CREATE DATAFRAME
# =========================
df = pd.DataFrame(data)

# =========================
# SAVE FILE (ABSOLUTE PATH FIX)
# =========================
file_path = os.path.abspath("bangalore_real_estate_5000.csv")
df.to_csv(file_path, index=False)

# =========================
# OUTPUT
# =========================
print("✅ Dataset created successfully!")
print("📁 Saved at:", file_path)
print("📊 Shape:", df.shape)
print(df.head())