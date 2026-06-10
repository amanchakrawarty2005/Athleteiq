from fastapi import FastAPI, HTTPException
from api.schemas import MatchInput, PlayerRatingOutput, TopPerformersInput
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
import sqlite3

app = FastAPI(title="AthletIQ API", version="1.0")

MODELS = Path("models")
RAW = Path("data/raw")
PROCESSED = Path("data/processed")

# Load models at startup
match_model = joblib.load(MODELS / "match_predictor.pkl")
label_encoder = joblib.load(MODELS / "label_encoder.pkl")
shap_importance = joblib.load(MODELS / "shap_importance.pkl")
player_scaler = joblib.load(MODELS / "player_scaler.pkl")
player_model = tf.keras.models.load_model(MODELS / "player_rater.keras")
archetype_model = joblib.load(MODELS / "archetype_clusterer.pkl")
archetype_scaler = joblib.load(MODELS / "archetype_scaler.pkl")
archetype_pca = joblib.load(MODELS / "archetype_pca.pkl")
injury_model = joblib.load(MODELS / "injury_risk.pkl")
injury_scaler = joblib.load(MODELS / "injury_scaler.pkl")

FEATURES = [
    "home_form", "away_form",
    "home_fixture_density", "away_fixture_density",
    "elo_delta", "home_advantage",
    "B365H", "B365D", "B365A"
]

ARCHETYPE_NAMES = {
    0: "Defensive Anchor", 1: "Box-to-Box Midfielder",
    2: "False 9", 3: "Target Striker",
    4: "Creative Playmaker", 5: "Pressing Forward"
}


@app.get("/")
def root():
    return {"message": "AthletIQ API running"}


@app.post("/predict-match")
def predict_match(data: MatchInput):
    X = pd.DataFrame([data.dict()])
    X = X[FEATURES]
    proba = match_model.predict_proba(X)[0]
    classes = label_encoder.classes_
    return {
        "probabilities": dict(zip(classes.tolist(), proba.round(4).tolist())),
        "prediction": classes[np.argmax(proba)],
        "feature_importance": shap_importance
    }


@app.get("/rate-player/{player_id}")
def rate_player(player_id: int):
    conn = sqlite3.connect(RAW / "database.sqlite")
    df = pd.read_sql(
        f"SELECT * FROM Player_Attributes WHERE player_api_id={player_id} ORDER BY date DESC LIMIT 1",
        conn
    )
    conn.close()

    if df.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    stat_cols = [
        "crossing", "finishing", "heading_accuracy", "short_passing",
        "volleys", "dribbling", "curve", "free_kick_accuracy", "long_passing",
        "ball_control", "acceleration", "sprint_speed", "agility", "reactions",
        "balance", "shot_power", "jumping", "stamina", "strength", "long_shots",
        "aggression", "interceptions", "positioning", "vision", "penalties",
        "marking", "standing_tackle", "sliding_tackle"
    ]
    stats = df[stat_cols].fillna(0).values
    scaled = player_scaler.transform(stats)
    score = float(player_model.predict(scaled)[0][0]) * 100

    return {
        "player_id": player_id,
        "performance_score": round(score, 2),
        "stats": df[stat_cols].iloc[0].to_dict()
    }


@app.get("/cluster-archetypes")
def cluster_archetypes():
    df = pd.read_parquet(PROCESSED / "player_archetypes.parquet")
    result = df[["player_api_id", "player_name", "archetype", "pca_x", "pca_y"]].head(500)
    return result.to_dict(orient="records")


@app.get("/top-performers")
def top_performers(metric: str = "finishing", limit: int = 10):
    conn = sqlite3.connect(RAW / "database.sqlite")
    df = pd.read_sql(
        f"""
        SELECT p.player_api_id, p.player_name, MAX(pa.{metric}) as {metric}
        FROM Player_Attributes pa
        JOIN Player p ON pa.player_api_id = p.player_api_id
        WHERE pa.{metric} IS NOT NULL
        GROUP BY p.player_api_id, p.player_name
        ORDER BY {metric} DESC
        LIMIT {limit}
        """,
        conn
    )
    conn.close()
    return df.to_dict(orient="records")