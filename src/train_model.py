"""
Model Training Module
"""

from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":
    print("Model Training Module")
