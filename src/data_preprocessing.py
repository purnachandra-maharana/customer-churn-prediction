"""
Data Preprocessing Module for Telco Customer Churn Prediction
"""

import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "Contract_One year",
    "Contract_Two year",
    "InternetService_Fiber optic",
    "InternetService_No",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check"
]


def load_data(file_path):
    """
    Load dataset from a file path or buffer.
    """
    return pd.read_csv(file_path)


def preprocess_data(df):
    """
    Clean and encode customer churn DataFrame for machine learning.
    Returns:
        X (pd.DataFrame): Normalized feature matrix.
        y (pd.Series or None): Target binary churn vector (1=Yes, 0=No).
    """
    df = df.copy()

    # Clean TotalCharges numeric column
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Extract target if present
    y = None
    if "Churn" in df.columns:
        y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)

    # One-hot encode key categorical columns
    categorical_cols = ["Contract", "InternetService", "PaymentMethod"]
    encoded_df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=False)

    # Ensure all required features are present
    X = pd.DataFrame(index=df.index)
    for col in FEATURE_COLUMNS:
        if col in encoded_df.columns:
            X[col] = pd.to_numeric(encoded_df[col], errors="coerce").fillna(0)
        else:
            X[col] = 0

    return X, y


if __name__ == "__main__":
    print("Data Preprocessing Module initialized.")
