import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

PROCESSED = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

FEATURES = [
    "home_form", "away_form",
    "home_fixture_density", "away_fixture_density",
    "elo_delta", "home_advantage",
    "B365H", "B365D", "B365A"
]

def train():
    df = pd.read_parquet(PROCESSED / "features.parquet")

    le = LabelEncoder()
    df["result_enc"] = le.fit_transform(df["result"])

    X = df[FEATURES]
    y = df["result_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline = (y_test == le.transform(["H"])[0]).mean()
    print(f"Naive baseline (always Home): {baseline:.2%}")

    mlflow.set_experiment("match_predictor")
    with mlflow.start_run():
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="mlogloss",
            random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.2%}")
        print(classification_report(y_test, preds, target_names=le.classes_))

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 4)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("baseline", baseline)
        mlflow.log_metric("lift", acc - baseline)

        importance = model.feature_importances_
        print("\nTop features (importance):")
        for f, s in sorted(zip(FEATURES, importance), key=lambda x: -x[1]):
            print(f"  {f}: {s:.4f}")

        shap_dict = dict(zip(FEATURES, importance.tolist()))

        joblib.dump(model, MODELS / "match_predictor.pkl")
        joblib.dump(le, MODELS / "label_encoder.pkl")
        joblib.dump(shap_dict, MODELS / "shap_importance.pkl")
        mlflow.sklearn.log_model(model, "model")
        print("Model saved to models/match_predictor.pkl")

if __name__ == "__main__":
    train()