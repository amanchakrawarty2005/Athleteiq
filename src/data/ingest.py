import pandas as pd
import sqlite3
from statsbombpy import sb
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(exist_ok=True)


def load_european_soccer_db():
    conn = sqlite3.connect(RAW / "database.sqlite")
    matches = pd.read_sql("SELECT * FROM Match", conn)
    teams = pd.read_sql("SELECT * FROM Team", conn)
    players = pd.read_sql("SELECT * FROM Player", conn)
    conn.close()
    return matches, teams, players


def load_spi_matches():
    return pd.read_csv(RAW / "spi_matches.csv")


def load_statsbomb(competition_id=2, season_id=44):
    # competition_id=2 → EPL, season_id=44 → 2003/04 (free tier)
    # To see all available: sb.competitions()
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    return matches


def load_statsbomb_events(match_id):
    return sb.events(match_id=match_id)


if __name__ == "__main__":
    print("Loading European Soccer DB...")
    matches, teams, players = load_european_soccer_db()
    matches.to_parquet(PROCESSED / "euro_matches.parquet")
    teams.to_parquet(PROCESSED / "teams.parquet")
    players.to_parquet(PROCESSED / "players.parquet")
    print(f"  Matches: {len(matches)}, Teams: {len(teams)}, Players: {len(players)}")

    print("Loading SPI matches...")
    spi = load_spi_matches()
    spi.to_parquet(PROCESSED / "spi_matches.parquet")
    print(f"  Rows: {len(spi)}")

    print("Loading StatsBomb EPL...")
    sb_matches = load_statsbomb()
    sb_matches.to_parquet(PROCESSED / "statsbomb_matches.parquet")
    print(f"  Matches: {len(sb_matches)}")

    print("Done. Files saved to data/processed/")