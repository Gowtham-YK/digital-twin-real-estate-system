import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import DATA_PATH

def load_data():
    data = pd.read_csv(DATA_PATH)

    data = data.dropna()

    le = LabelEncoder()
    data['location'] = le.fit_transform(data['location'])

    return data, le