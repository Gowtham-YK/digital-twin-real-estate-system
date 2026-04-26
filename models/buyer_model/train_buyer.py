import os
os.environ["MLFLOW_TRACKING_URI"] = "mlruns"
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn

mlflow.set_experiment("Buyer_Model")

# =========================
# SET BASE PATH (IMPORTANT)
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_PATH = os.path.join(BASE_DIR, "data", "banglore.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "buyer_model", "buyer_model.pkl")

print(f"📂 Loading dataset from: {DATA_PATH}")

# =========================
# LOAD DATA
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded successfully")
print("Columns:", df.columns.tolist())

# =========================
# RENAME COLUMNS (SAFE)
# =========================
df = df.rename(columns={
    "latitude": "lat",
    "longitude": "lon",
    "sqft": "sqft",
    "BHK": "bhk",
    "budget_price": "price"
})

# =========================
# SELECT REQUIRED COLUMNS
# =========================
required_cols = ["lat", "lon", "sqft", "bhk", "price"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column: {col}")

df = df[required_cols]

# =========================
# CLEAN DATA
# =========================
df = df.dropna()

# Optional: basic filtering
df = df[df["sqft"] > 100]
df = df[df["price"] > 10000]

print(f"✅ Data cleaned. Rows remaining: {len(df)}")

# =========================
# FEATURES & TARGET
# =========================
X = df[["lat", "lon", "sqft", "bhk"]]
y = df["price"]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6
)

print("⏳ Training model...")
model.fit(X_train, y_train)

# =========================
# SAVE MODEL
# =========================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model saved at: {MODEL_PATH}")

# =========================
# SIMPLE EVALUATION
# =========================
score = model.score(X_test, y_test)
print(f"📊 Model R² Score: {score:.4f}")

with mlflow.start_run():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    from sklearn.metrics import r2_score
    score = r2_score(y_test, preds)

    mlflow.log_param("model_type", "buyer_model")
    mlflow.log_metric("r2_score", score)

    joblib.dump(model, "models/buyer_model/model.pkl")

    mlflow.sklearn.log_model(model, "model")

    print(f"Buyer Model R2 Score: {score}")