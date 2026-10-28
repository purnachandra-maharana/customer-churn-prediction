"""
Prediction Module
"""

import joblib
import pandas as pd


def load_model(model_path):
    """
    Load a serialized scikit-learn model from disk.
    """
    return joblib.load(model_path)


def predict(model, input_features):
    """
    Predict binary churn labels (0 = Retention, 1 = Churn).
    """
    return model.predict(input_features)


def predict_probability(model, input_features):
    """
    Predict churn probability scores (float between 0.0 and 1.0).
    """
    probs = model.predict_proba(input_features)
    # Return probability of churn (class 1)
    if probs.shape[1] > 1:
        return probs[:, 1]
    return probs[:, 0]


if __name__ == "__main__":
    print("Prediction Module initialized.")
