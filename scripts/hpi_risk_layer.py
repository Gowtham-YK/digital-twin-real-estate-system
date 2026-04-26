# ==========================================
# REALTWINAI - ISOLATION FOREST RISK LAYER WITH METRICS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. LOAD YOUR HPI DATASET
# ==========================================
df = pd.read_csv("us_home_price_analysis_2004_2024 (lstm).csv")  # replace with your path
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)
print("Dataset loaded:", df.shape)

# ==========================================
# 2. SELECT FEATURES FOR RISK DETECTION
# ==========================================
features = [
    "Home_Price_Index",
    "Interest_Rate",
    "Unemployment_Rate",
    "Mortgage_Rate",
    "Inflation_CPI",
    "Building_Permits",
    "Consumer_Sentiment",
    "Housing_Starts",
    "US_Population",
    "Median_Income"
]

X = df[features]

# ==========================================
# 3. SCALE FEATURES
# ==========================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 4. FIT ISOLATION FOREST
# ==========================================
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # adjust expected anomaly %
    random_state=42
)
iso_forest.fit(X_scaled)

# ==========================================
# 5. PREDICT ANOMALIES
# ==========================================
df["anomaly_score"] = iso_forest.decision_function(X_scaled)
df["anomaly_flag"] = iso_forest.predict(X_scaled)
df["anomaly_flag"] = df["anomaly_flag"].map({1: 0, -1: 1})

# ==========================================
# 6. PERFORMANCE METRICS
# ==========================================
num_anomalies = df["anomaly_flag"].sum()
percent_anomalies = 100 * num_anomalies / len(df)
min_score = df["anomaly_score"].min()
max_score = df["anomaly_score"].max()
mean_score = df["anomaly_score"].mean()

print("\n===== ISOLATION FOREST PERFORMANCE =====")
print(f"Total data points     : {len(df)}")
print(f"Detected anomalies    : {num_anomalies}")
print(f"Percentage anomalies  : {percent_anomalies:.2f}%")
print(f"Anomaly score min     : {min_score:.4f}")
print(f"Anomaly score max     : {max_score:.4f}")
print(f"Anomaly score mean    : {mean_score:.4f}")

# ==========================================
# 7. LIST DETECTED ANOMALIES
# ==========================================
anomalies = df[df["anomaly_flag"] == 1]
print("\nSample anomalies:")
print(anomalies[["DATE", "Home_Price_Index", "anomaly_score"]].head())

# ==========================================
# 8. PLOT HPI + ANOMALIES
# ==========================================
plt.figure(figsize=(14,6))
plt.plot(df["DATE"], df["Home_Price_Index"], label="Home Price Index", color="blue")
plt.scatter(
    anomalies["DATE"],
    anomalies["Home_Price_Index"],
    color="red",
    marker="x",
    label="Anomaly (Risk)"
)
plt.title("Isolation Forest Risk Detection - HPI Anomalies")
plt.xlabel("Date")
plt.ylabel("Home Price Index")
plt.legend()
plt.show()