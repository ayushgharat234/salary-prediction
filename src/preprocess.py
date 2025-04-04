# preprocess.py
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.rename(columns={'Years of Experience': 'YearsExperience'}, inplace=True)
    return df.dropna()

def preprocess_data(df: pd.DataFrame):
    df["YearsExperience"], lambda_bc = boxcox(df["YearsExperience"] + 1e-6)
    df["Salary"] = np.log1p(df["Salary"])

    X = df[["YearsExperience"]]
    y = df["Salary"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=0
    )

    return X_train, X_test, y_train, y_test, scaler, lambda_bc