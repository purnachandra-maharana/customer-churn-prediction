import os
import pytest
import pandas as pd
import numpy as np

from src.data_preprocessing import preprocess_data, FEATURE_COLUMNS
from src.train_model import train_model
from src.predict import load_model, predict, predict_probability


@pytest.fixture
def raw_churn_df():
    return pd.DataFrame({
        "customerID": ["C1", "C2", "C3", "C4"],
        "tenure": [1, 24, 60, 12],
        "MonthlyCharges": [70.5, 25.0, 95.0, 50.0],
        "TotalCharges": ["70.5", "600.0", "5700.0", " "],  # whitespace in TotalCharges to test coercion
        "SeniorCitizen": [0, 0, 1, 0],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "InternetService": ["Fiber optic", "DSL", "Fiber optic", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        "Churn": ["Yes", "No", "No", "Yes"]
    })


def test_preprocess_data_shapes_and_features(raw_churn_df):
    X, y = preprocess_data(raw_churn_df)
    assert isinstance(X, pd.DataFrame)
    assert len(X) == 4
    assert list(X.columns) == FEATURE_COLUMNS
    assert y is not None
    assert list(y) == [1, 0, 0, 1]
    # Check that whitespace TotalCharges was converted to numeric median without NaN
    assert not X["TotalCharges"].isnull().any()


def test_preprocess_without_target(raw_churn_df):
    df_no_target = raw_churn_df.drop(columns=["Churn"])
    X, y = preprocess_data(df_no_target)
    assert len(X) == 4
    assert y is None
    assert list(X.columns) == FEATURE_COLUMNS


def test_train_model(raw_churn_df):
    X, y = preprocess_data(raw_churn_df)
    model = train_model(X, y, n_estimators=10, max_depth=3)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    preds = model.predict(X)
    assert len(preds) == len(X)


def test_predict_and_probability(raw_churn_df):
    X, y = preprocess_data(raw_churn_df)
    model = train_model(X, y, n_estimators=10, max_depth=3)
    preds = predict(model, X)
    probs = predict_probability(model, X)

    assert len(preds) == 4
    assert len(probs) == 4
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_serialized_model_inference():
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "churn_model.joblib"
    )
    if os.path.exists(model_path):
        model = load_model(model_path)
        dummy_input = pd.DataFrame(
            np.zeros((2, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
        )
        probs = predict_probability(model, dummy_input)
        assert len(probs) == 2
        assert all(0.0 <= p <= 1.0 for p in probs)
