import os
os.environ["MLFLOW_TRACKING_URI"] = "mlruns"
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn

mlflow.set_experiment("Seller_Model")

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data", "seller_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "seller_model", "seller_model.pkl")

print(f"📂 Loading dataset from: {DATA_PATH}")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded")
print("Columns:", df.columns.tolist())

# =========================
# CLEAN DATA
# =========================
df = df[df["buy_price"] > 0]
df = df[df["sell_price"] > 0]

# =========================
# FEATURE ENGINEERING
# =========================
df["years"] = df["sell_year"] - df["buy_year"]
df = df[df["years"] > 0]

# =========================
# FEATURES & TARGET
# =========================
X = df[["latitude", "longitude", "sqft", "BHK", "buy_price", "years"]]
y = df["sell_price"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=6
)

print("⏳ Training seller model...")
model.fit(X_train, y_train)

# =========================
# SAVE MODEL
# =========================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model saved at: {MODEL_PATH}")

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

# Percentage Error (MAPE-like)
percentage_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# =========================
# PRINT RESULTS
# =========================
print("\n📊 MODEL PERFORMANCE:")
print(f"Error: {percentage_error:.2f}%")
print(f"R² Score: {r2:.4f}")

with mlflow.start_run():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    from sklearn.metrics import r2_score
    score = r2_score(y_test, preds)

    mlflow.log_param("model_type", "seller_model")
    mlflow.log_metric("r2_score", score)

    joblib.dump(model, "models/seller_model/model.pkl")

    mlflow.sklearn.log_model(model, "model")

    print(f"Seller Model R2 Score: {score}")