import pandas as pd
import numpy as np
import mlflow
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import sqlite3

RAW = Path("data/raw")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

STAT_COLS = [
    "finishing", "heading_accuracy", "short_passing", "dribbling",
    "ball_control", "acceleration", "sprint_speed", "stamina",
    "strength", "long_shots", "aggression", "interceptions",
    "positioning", "vision", "marking", "standing_tackle"
]

ARCHETYPE_NAMES = {
    0: "Defensive Anchor",
    1: "Box-to-Box Midfielder",
    2: "False 9",
    3: "Target Striker",
    4: "Creative Playmaker",
    5: "Pressing Forward"
}

def train():
    conn = sqlite3.connect(RAW / "database.sqlite")
    players = pd.read_sql("SELECT * FROM Player", conn)
    attrs = pd.read_sql("SELECT * FROM Player_Attributes", conn)
    conn.close()

    # Latest attributes per player
    attrs = attrs.sort_values("date").groupby("player_api_id").last().reset_index()
    df = attrs[["player_api_id"] + STAT_COLS].dropna()
    print(f"Players: {len(df)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[STAT_COLS])

    # KMeans k=6
    mlflow.set_experiment("archetype_clusterer")
    with mlflow.start_run():
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels, sample_size=5000)
        print(f"Silhouette score: {score:.4f}")

        # PCA to 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        df["cluster"] = labels
        df["archetype"] = df["cluster"].map(ARCHETYPE_NAMES)
        df["pca_x"] = X_2d[:, 0]
        df["pca_y"] = X_2d[:, 1]

        # Merge player names
        df = df.merge(players[["player_api_id", "player_name"]], on="player_api_id", how="left")

        print("\nArchetype distribution:")
        print(df["archetype"].value_counts())

        mlflow.log_metric("silhouette_score", score)
        mlflow.log_param("k", 6)

        joblib.dump(kmeans, MODELS / "archetype_clusterer.pkl")
        joblib.dump(scaler, MODELS / "archetype_scaler.pkl")
        joblib.dump(pca, MODELS / "archetype_pca.pkl")
        df.to_parquet(Path("data/processed") / "player_archetypes.parquet")
        print("Model saved to models/archetype_clusterer.pkl")

if __name__ == "__main__":
    train()