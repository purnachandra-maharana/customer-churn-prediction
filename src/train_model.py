"""
Model Training Module
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_preprocessing import load_data, preprocess_data


def train_model(X_train, y_train, n_estimators=100, max_depth=6, random_state=42):
    """
    Train a Random Forest classifier for churn classification.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


def train_and_save_model(data_path, output_model_path):
    """
    Convenience function to train from CSV and serialize to joblib.
    """
    df = load_data(data_path)
    X, y = preprocess_data(df)
    if y is None:
        raise ValueError("Dataset does not contain a target 'Churn' column.")

    model = train_model(X, y)
    os.makedirs(Path(output_model_path).parent, exist_ok=True)
    joblib.dump(model, output_model_path)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    print(f"Model trained successfully. Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
    return model, acc, auc


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_file = base_dir / "data" / "sample_churn.csv"
    model_file = base_dir / "models" / "churn_model.joblib"
    if data_file.exists():
        train_and_save_model(data_file, model_file)
