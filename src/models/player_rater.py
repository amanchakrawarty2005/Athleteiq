import pandas as pd
import numpy as np
import mlflow
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

RAW = Path("data/raw")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

STAT_COLS = [
    "overall_rating", "potential", "crossing", "finishing",
    "heading_accuracy", "short_passing", "volleys", "dribbling",
    "curve", "free_kick_accuracy", "long_passing", "ball_control",
    "acceleration", "sprint_speed", "agility", "reactions",
    "balance", "shot_power", "jumping", "stamina", "strength",
    "long_shots", "aggression", "interceptions", "positioning",
    "vision", "penalties", "marking", "standing_tackle", "sliding_tackle"
]

def train():
    import sqlite3
    conn = sqlite3.connect(RAW / "database.sqlite")
    df = pd.read_sql("SELECT * FROM Player_Attributes", conn)
    conn.close()

    df = df[STAT_COLS].dropna()
    print(f"Player records: {len(df)}")

    # Target: overall_rating (FIFA composite score)
    y = df["overall_rating"].values
    X = df.drop(columns=["overall_rating", "potential"]).values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = y / 100.0  # normalize to 0-1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("player_rater")
    with mlflow.start_run():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.fit(X_train, y_train, epochs=30, batch_size=32,
                  validation_split=0.1, verbose=0)

        preds = model.predict(X_test).flatten()
        mae = mean_absolute_error(y_test, preds)
        print(f"MAE: {mae:.4f} (on 0-1 scale = {mae*100:.2f} rating points)")

        mlflow.log_metric("mae", mae)

        model.save(MODELS / "player_rater.keras")
        joblib.dump(scaler, MODELS / "player_scaler.pkl")
        print("Model saved to models/player_rater.keras")

if __name__ == "__main__":
    train()