import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "real_twin_xgboost_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "real_estate_xgboost.csv")