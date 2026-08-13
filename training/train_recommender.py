"""
Trains the recommendation-engine classifier (RandomForest) on the seed
dataset and saves it to models/recommender.pkl.

Run (after generate_recommendation_dataset.py):
    python training/train_recommender.py
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "recommendation_seed_data.csv")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run "
            "training/generate_recommendation_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df[config.RECOMMENDER_FEATURE_COLUMNS]
    y = df["care_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {f1:.3f}")
    print(
        classification_report(
            y_test, y_pred, target_names=config.CARE_CATEGORIES
        )
    )

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(model, config.RECOMMENDER_MODEL_PATH)
    print(f"Saved trained recommender to {config.RECOMMENDER_MODEL_PATH}")


if __name__ == "__main__":
    main()
