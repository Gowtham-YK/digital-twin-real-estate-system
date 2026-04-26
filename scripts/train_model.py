import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.utils import shuffle
from xgboost import XGBRegressor

print("🚀 Training started...")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/banglore.csv")
print("✅ Data loaded")

# =========================
# FEATURE EXTRACTION
# =========================
df['bedrooms'] = df['size'].str.extract(r'(\d+)').astype(float)

def convert_sqft(x):
    try:
        if '-' in str(x):
            a, b = x.split('-')
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return None

df['area'] = df['total_sqft'].apply(convert_sqft)
df['price'] = df['price'] * 100000

# =========================
# SELECT COLUMNS
# =========================
df = df[['location', 'area', 'bedrooms', 'bath', 'balcony', 'price']]

# =========================
# CLEAN DATA
# =========================
df = df.dropna()

# 🔥 LESS AGGRESSIVE FILTER (better than quantile clipping)
df = df[(df['price'] > 2e5) & (df['price'] < 5e7)]

# =========================
# PRICE PER SQFT (FILTER ONLY)
# =========================
df['_ppsf'] = df['price'] / df['area']
df = df[(df['_ppsf'] < 20000) & (df['_ppsf'] > 1000)]
df = df.drop(columns=['_ppsf'])

# =========================
# LOCATION GROUPING
# =========================
location_counts = df['location'].value_counts()
df['location'] = df['location'].apply(
    lambda x: x if location_counts.get(x, 0) > 50 else 'other'
)

# =========================
# ENCODING
# =========================
df['location'] = df['location'].astype('category')
location_mapping = dict(enumerate(df['location'].cat.categories))
df['location'] = df['location'].cat.codes

joblib.dump(location_mapping, "models/location_mapping.pkl")

# =========================
# EXTRA CLEANING
# =========================
df = df[(df['area'] > 400) & (df['area'] < 3500)]
df = df[(df['bedrooms'] >= 1) & (df['bedrooms'] <= 5)]
df = df[(df['bath'] >= 1) & (df['bath'] <= 5)]
df = df[(df['balcony'] >= 0) & (df['balcony'] <= 3)]

# =========================
# SHUFFLE
# =========================
df = shuffle(df, random_state=42)

# =========================
# SAVE CLEAN DATA
# =========================
df.to_json("data/clean_data.json", orient="records", indent=4)

# =========================
# TRAINING
# =========================
X = df[['location', 'area', 'bedrooms', 'bath', 'balcony']]
y = np.log1p(df['price'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 FIX: ensure numeric types
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# =========================
# MODEL
# =========================
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred_log = model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

print("\n📊 RESULTS")
print("Final R2:", r2_score(y_test_actual, y_pred))
print("MAE:", mean_absolute_error(y_test_actual, y_pred))

# =========================
# CROSS VALIDATION (fresh model)
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

scores = cross_val_score(cv_model, X.astype(float), y, cv=kf, scoring='r2')

print("Cross-validated R2:", np.mean(scores))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "models/xgb_model_real.pkl")
print("✅ Model saved")

# =========================
# TEST SAMPLE (FIXED)
# =========================
sample_df = pd.DataFrame([{
    'location': 10,
    'area': 1200,
    'bedrooms': 3,
    'bath': 2,
    'balcony': 1
}])

sample_df = sample_df.astype(float)

pred_log = model.predict(sample_df)
pred_price = np.expm1(pred_log)

print("🎯 Predicted Price:", pred_price[0])