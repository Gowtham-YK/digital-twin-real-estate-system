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

# =========================
# DRIFT CREATION
# =========================
def introduce_drift(df):
    df_drift = df.copy()

    # disturb target
    df_drift["sell_price"] = df_drift["sell_price"] * np.random.uniform(0.6, 1.4, len(df))

    # break relationship
    df_drift["sqft"] = np.random.permutation(df_drift["sqft"].values)

    # add outliers
    df_drift.loc[df_drift.sample(frac=0.1).index, "sell_price"] *= 1.8

    return df_drift


# =========================
# CLEAN DATA
# =========================
def clean_data(df):
    df = df[df["sell_price"] < df["sell_price"].quantile(0.95)]
    df["sell_price"] = df["sell_price"].clip(lower=10000)
    return df

mlflow.set_experiment("Seller_Model")

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data", "seller_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "seller_model", "model.pkl")

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
df_original = df.copy()

# =========================
# INTRODUCE DRIFT
# =========================
df = introduce_drift(df)

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

# =========================
# MLFLOW TRAINING + DRIFT LOGIC
# =========================
with mlflow.start_run():

    # Train on drifted data
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    drift_score = r2_score(y_test, preds)

    print("⚠️ Seller Drifted R2:", drift_score)
    mlflow.log_metric("seller_drift_r2", drift_score)

    # Detect + Fix
    if drift_score < 0.85:
        print("🚨 Drift detected in seller model. Fixing...")

        df_clean = clean_data(df_original)

        # recreate features
        df_clean["years"] = df_clean["sell_year"] - df_clean["buy_year"]
        df_clean = df_clean[df_clean["years"] > 0]

        X = df_clean[["latitude", "longitude", "sqft", "BHK", "buy_price", "years"]]
        y = df_clean["sell_price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fixed_score = r2_score(y_test, preds)

        print("✅ Seller Recovered R2:", fixed_score)
        mlflow.log_metric("seller_recovered_r2", fixed_score)

    else:
        fixed_score = drift_score

    # Save model (correct place)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    mlflow.sklearn.log_model(model, "model")