import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED = Path("data/processed")


def add_result_label(df):
    df["result"] = df.apply(lambda r:
        "H" if r.home_team_goal > r.away_team_goal else
        ("D" if r.home_team_goal == r.away_team_goal else "A"), axis=1)
    return df


def rolling_form(df, n=5):
    """Last-n-game rolling avg goals for each team."""
    df = df.sort_values("date").copy()
    records = []

    for _, row in df.iterrows():
        match_date = row["date"]
        ht = row["home_team_api_id"]
        at = row["away_team_api_id"]

        # Home team last 5
        past_home = df[
            ((df["home_team_api_id"] == ht) | (df["away_team_api_id"] == ht)) &
            (df["date"] < match_date)
        ].tail(n)

        h_goals = []
        for _, r in past_home.iterrows():
            h_goals.append(r.home_team_goal if r.home_team_api_id == ht else r.away_team_goal)

        # Away team last 5
        past_away = df[
            ((df["home_team_api_id"] == at) | (df["away_team_api_id"] == at)) &
            (df["date"] < match_date)
        ].tail(n)

        a_goals = []
        for _, r in past_away.iterrows():
            a_goals.append(r.home_team_goal if r.home_team_api_id == at else r.away_team_goal)

        records.append({
            "id": row["id"],
            "home_form": np.mean(h_goals) if h_goals else 0.0,
            "away_form": np.mean(a_goals) if a_goals else 0.0,
        })

    return pd.DataFrame(records)


def add_elo_delta(euro, spi):
    """Merge SPI ratings as ELO proxy into euro matches."""
    spi["date"] = pd.to_datetime(spi["date"])
    euro["date"] = pd.to_datetime(euro["date"])

    # Use spi1 - spi2 as ELO delta per match
    spi_slim = spi[["date", "team1", "team2", "spi1", "spi2", "xg1", "xg2"]].copy()
    spi_slim["elo_delta"] = spi_slim["spi1"] - spi_slim["spi2"]
    return spi_slim


def add_fixture_density(df, window_days=14):
    """Fatigue proxy: number of games played in last N days."""
    df = df.sort_values("date").copy()
    records = []

    for _, row in df.iterrows():
        match_date = row["date"]
        ht = row["home_team_api_id"]
        at = row["away_team_api_id"]

        cutoff = match_date - pd.Timedelta(days=window_days)

        home_density = len(df[
            ((df["home_team_api_id"] == ht) | (df["away_team_api_id"] == ht)) &
            (df["date"] >= cutoff) & (df["date"] < match_date)
        ])

        away_density = len(df[
            ((df["home_team_api_id"] == at) | (df["away_team_api_id"] == at)) &
            (df["date"] >= cutoff) & (df["date"] < match_date)
        ])

        records.append({
            "id": row["id"],
            "home_fixture_density": home_density,
            "away_fixture_density": away_density
        })

    return pd.DataFrame(records)


def build_feature_matrix(sample=True):
    euro = pd.read_parquet(PROCESSED / "euro_matches_clean.parquet")
    spi  = pd.read_parquet(PROCESSED / "spi_clean.parquet")

    # Sample one league/season for speed during dev
    if sample:
        euro = euro[euro["season"] == "2015/2016"].copy()
        print(f"Using sample: {len(euro)} matches")

    euro = add_result_label(euro)

    print("Computing rolling form...")
    form = rolling_form(euro)
    euro = euro.merge(form, on="id", how="left")

    print("Computing fixture density...")
    density = add_fixture_density(euro)
    euro = euro.merge(density, on="id", how="left")

    # ELO delta from SPI (match by match where available)
    spi_features = add_elo_delta(euro, spi)
    euro["elo_delta"] = spi_features["elo_delta"].mean()  # global mean as fallback

    # Home advantage flag
    euro["home_advantage"] = 1

    # Final feature cols
    feature_cols = [
        "home_form", "away_form",
        "home_fixture_density", "away_fixture_density",
        "elo_delta", "home_advantage",
        "B365H", "B365D", "B365A"
    ]

    euro = euro.dropna(subset=feature_cols)
    euro[feature_cols + ["result"]].to_parquet(PROCESSED / "features.parquet")
    print(f"Features saved: {len(euro)} rows, {len(feature_cols)} features")
    return euro


if __name__ == "__main__":
    build_feature_matrix(sample=True)