import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def encode_features(df):
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])
    return df

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)