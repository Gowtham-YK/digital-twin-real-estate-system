# ==========================================
# REALTWINAI - MULTIVARIATE LSTM FORECAST (FIXED)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv("us_home_price_analysis_2004_2024 (lstm).csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
target = "Home_Price_Index"

# Only use lagged HPI features + 2–3 strong economic indicators
df['HPI_lag1'] = df[target].shift(1)
df['HPI_lag3'] = df[target].shift(3)
df['HPI_roll3'] = df[target].rolling(3).mean()

# Fill NaNs with previous value
df.fillna(method='bfill', inplace=True)

feature_columns = [
    'HPI_lag1', 'HPI_lag3', 'HPI_roll3',
    'Interest_Rate', 'Unemployment_Rate', 'Mortgage_Rate'
]

# ==========================================
# 3. SCALE DATA
# ==========================================
scaler_features = MinMaxScaler()
scaled_features = scaler_features.fit_transform(df[feature_columns])

scaler_target = MinMaxScaler()
scaled_target = scaler_target.fit_transform(df[[target]])

# ==========================================
# 4. CREATE SEQUENCES
# ==========================================
sequence_length = 6  # shorter sequence
X, y = [], []
for i in range(sequence_length, len(df)):
    X.append(scaled_features[i-sequence_length:i])
    y.append(scaled_target[i, 0])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==========================================
# 6. BUILD LSTM MODEL
# ==========================================
model = Sequential()
model.add(LSTM(16, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ==========================================
# 7. TRAIN
# ==========================================
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 8. PREDICTION
# ==========================================
predictions = model.predict(X_test)
predictions = scaler_target.inverse_transform(predictions)
actual = scaler_target.inverse_transform(y_test.reshape(-1,1))

# ==========================================
# 9. METRICS
# ==========================================
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
r2 = r2_score(actual, predictions)

print("\n===== MULTIVARIATE LSTM RESULTS =====")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")
print(f"Accuracy : {r2*100:.2f}%")

# ==========================================
# 10. PLOT
# ==========================================
plt.figure(figsize=(12,6))
plt.plot(actual, label="Actual HPI")
plt.plot(predictions, label="Predicted HPI")
plt.title("Multivariate LSTM - HPI Forecast (Fixed)")
plt.xlabel("Time")
plt.ylabel("HPI")
plt.legend()
plt.show()