import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from data_loader import load_data
from config import MODEL_PATH

def train_model():
    data, le = load_data()

    X = data[['location', 'area', 'bedrooms']]
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model + encoder
    joblib.dump((model, le), MODEL_PATH)

    print("✅ XGBoost Model trained & saved successfully!")

if __name__ == "__main__":
    train_model()