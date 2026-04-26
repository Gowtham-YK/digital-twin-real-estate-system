# ==========================================
# REALTWINAI - LIGHTGBM CORE PREDICTION MODEL
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

# ==========================================
# 1. LOAD DATASET
# ==========================================
# Replace with your Kaggle CSV file
df = pd.read_csv("Real estate (XGBoost).csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
target_column = "Y house price of unit area"

df = df.dropna()

X = df.drop(columns=[target_column])
y = df[target_column]

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 4. LIGHTGBM MODEL
# ==========================================
lgbm_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgbm_model.fit(X_train, y_train)

# ==========================================
# 5. PREDICTIONS
# ==========================================
y_pred = lgbm_model.predict(X_test)

# ==========================================
# 6. EVALUATION METRICS
# ==========================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

accuracy = r2 * 100

print("\n===== LIGHTGBM MODEL PERFORMANCE =====")
print(f"MAE      : {mae:.2f}")
print(f"MSE      : {mse:.2f}")
print(f"RMSE     : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")
print(f"Accuracy : {accuracy:.2f}%")

# ==========================================
# 7. FEATURE IMPORTANCE
# ==========================================
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": lgbm_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n===== TOP 10 IMPORTANT FEATURES =====")
print(importance.head(10))

# ==========================================
# 8. INVESTMENT DECISION LOGIC
# ==========================================
results = X_test.copy()
results["Actual Price"] = y_test.values
results["Predicted Price"] = y_pred

results["Expected Appreciation %"] = (
    (results["Predicted Price"] - results["Actual Price"])
    / results["Actual Price"]
) * 100

def investment_decision(appreciation):
    if appreciation > 10:
        return "BUY"
    elif appreciation > 0:
        return "HOLD"
    else:
        return "SELL"

results["Recommendation"] = results["Expected Appreciation %"].apply(
    investment_decision
)

print("\n===== SAMPLE BUY / HOLD / SELL =====")
print(results[[
    "Actual Price",
    "Predicted Price",
    "Expected Appreciation %",
    "Recommendation"
]].head())

# ==========================================
# 9. SAVE MODEL FOR ANYLOGIC
# ==========================================
joblib.dump(lgbm_model, "real_twin_lightgbm_model.pkl")
print("\nModel saved as real_twin_lightgbm_model.pkl")