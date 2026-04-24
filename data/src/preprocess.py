import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess(data):
    X = data.drop("Energy", axis=1)
    y = data["Energy"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
