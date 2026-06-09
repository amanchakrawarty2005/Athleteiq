import pandas as pd
import sqlite3
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(exist_ok=True)


def clean_euro_matches():
    conn = sqlite3.connect(RAW / "database.sqlite")
    df = pd.read_sql("SELECT * FROM Match", conn)
    teams = pd.read_sql("SELECT team_api_id, team_long_name FROM Team", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df.dropna(subset=["home_team_goal", "away_team_goal"], inplace=True)

    # Keep only useful columns
    cols = ["id", "country_id", "league_id", "season", "date", "stage",
            "home_team_api_id", "away_team_api_id",
            "home_team_goal", "away_team_goal",
            "B365H", "B365D", "B365A"]
    df = df[cols]

    # Add result label
    df["result"] = df.apply(lambda r:
        "H" if r.home_team_goal > r.away_team_goal else
        ("D" if r.home_team_goal == r.away_team_goal else "A"), axis=1)

    # Merge team names
    df = df.merge(teams.rename(columns={"team_api_id": "home_team_api_id", "team_long_name": "home_team"}), on="home_team_api_id", how="left")
    df = df.merge(teams.rename(columns={"team_api_id": "away_team_api_id", "team_long_name": "away_team"}), on="away_team_api_id", how="left")

    return df


def clean_spi():
    df = pd.read_csv(RAW / "spi_matches.csv")
    df["date"] = pd.to_datetime(df["date"])
    df.dropna(subset=["score1", "score2"], inplace=True)
    df["result"] = df.apply(lambda r:
        "H" if r.score1 > r.score2 else
        ("D" if r.score1 == r.score2 else "A"), axis=1)
    return df


if __name__ == "__main__":
    euro = clean_euro_matches()
    euro.to_parquet(PROCESSED / "euro_matches_clean.parquet")
    print(f"Euro matches: {len(euro)}, cols: {euro.shape[1]}")

    spi = clean_spi()
    spi.to_parquet(PROCESSED / "spi_clean.parquet")
    print(f"SPI matches: {len(spi)}, cols: {spi.shape[1]}")

    print("Done.")