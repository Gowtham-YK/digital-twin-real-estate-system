import os
os.environ["MLFLOW_TRACKING_URI"] = "mlruns"
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import r2_score

# =========================
# DRIFT CREATION
# =========================
def introduce_drift(df):
    df_drift = df.copy()

    df_drift["price"] = df_drift["price"] * np.random.uniform(0.5, 1.5, len(df))
    df_drift["sqft"] = np.random.permutation(df_drift["sqft"].values)
    df_drift.loc[df_drift.sample(frac=0.1).index, "price"] *= 2

    return df_drift


# =========================
# CLEAN DATA
# =========================
def clean_data(df):
    df = df[df["price"] < df["price"].quantile(0.95)]
    df["price"] = df["price"].clip(lower=10000)
    return df

mlflow.set_experiment("Buyer_Model")

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data", "banglore.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "buyer_model", "model.pkl")

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
# RENAME COLUMNS
# =========================
df = df.rename(columns={
    "latitude": "lat",
    "longitude": "lon",
    "sqft": "sqft",
    "BHK": "bhk",
    "budget_price": "price"
})

required_cols = ["lat", "lon", "sqft", "bhk", "price"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column: {col}")

df = df[required_cols]

# =========================
# CLEAN DATA
# =========================
df = df.dropna()
df = df[df["sqft"] > 100]
df = df[df["price"] > 10000]

# ✅ FIX: take original AFTER cleaning
df_original = df.copy()

print(f"✅ Data cleaned. Rows remaining: {len(df)}")

# =========================
# INTRODUCE DRIFT
# =========================
df = introduce_drift(df)

# =========================
# FEATURES & TARGET
# =========================
X = df[["lat", "lon", "sqft", "bhk"]]
y = df["price"]

# =========================
# SPLIT
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

# =========================
# MLFLOW PIPELINE
# =========================
with mlflow.start_run():

    # Train on drifted data
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    drift_score = r2_score(y_test, preds)

    print("⚠️ Drifted R2 Score:", drift_score)
    mlflow.log_metric("drift_r2", drift_score)

    # Detect + Fix
    if drift_score < 0.85:
        print("🚨 Data drift detected! Fixing dataset...")

        df_clean = clean_data(df_original)

        X = df_clean[["lat", "lon", "sqft", "bhk"]]
        y = df_clean["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fixed_score = r2_score(y_test, preds)

        print("✅ Recovered R2 Score:", fixed_score)
        mlflow.log_metric("recovered_r2", fixed_score)

    else:
        fixed_score = drift_score

    # Save model (correct place)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    mlflow.sklearn.log_model(model, "model")