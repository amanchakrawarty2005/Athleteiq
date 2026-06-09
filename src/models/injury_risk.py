import pandas as pd
import numpy as np
import mlflow
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import sqlite3

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
MODELS = Path("models")

def build_injury_features():
    conn = sqlite3.connect(RAW / "database.sqlite")
    attrs = pd.read_sql("SELECT * FROM Player_Attributes", conn)
    players = pd.read_sql("SELECT * FROM Player", conn)
    conn.close()

    attrs["date"] = pd.to_datetime(attrs["date"])
    attrs = attrs.sort_values(["player_api_id", "date"])

    # Fixture density: number of records in last 90 days (proxy for minutes played)
    def fixture_density(group):
        group = group.sort_values("date")
        group["fixture_density"] = group["date"].apply(
            lambda d: ((group["date"] >= d - pd.Timedelta(days=90)) &
                       (group["date"] < d)).sum()
        )
        return group

    attrs = attrs.groupby("player_api_id", group_keys=False).apply(fixture_density)

    # Merge age
    players["birthday"] = pd.to_datetime(players["birthday"])
    attrs = attrs.merge(players[["player_api_id", "birthday"]], on="player_api_id", how="left")
    attrs["age"] = ((attrs["date"] - attrs["birthday"]).dt.days / 365).round(1)

    # Injury risk label: high stamina drop + high fixture density = risk
    attrs["stamina_drop"] = attrs.groupby("player_api_id")["stamina"].diff().fillna(0)
    attrs["injury_risk"] = ((attrs["stamina_drop"] < -3) & (attrs["fixture_density"] > 2)).astype(int)

    features = ["stamina", "strength", "sprint_speed", "fixture_density", "age"]
    df = attrs[features + ["injury_risk"]].dropna()
    print(f"Records: {len(df)}, Injury cases: {df['injury_risk'].sum()}")
    return df

def train():
    df = build_injury_features()

    X = df.drop(columns=["injury_risk"])
    y = df["injury_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlflow.set_experiment("injury_risk")
    with mlflow.start_run():
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        model = CalibratedClassifierCV(rf, cv=3, method="sigmoid")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)

        print(f"ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, preds))

        mlflow.log_metric("roc_auc", auc)

        joblib.dump(model, MODELS / "injury_risk.pkl")
        joblib.dump(scaler, MODELS / "injury_scaler.pkl")
        print("Model saved to models/injury_risk.pkl")

if __name__ == "__main__":
    train()