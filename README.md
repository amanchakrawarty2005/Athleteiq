# AthletIQ — Sports Performance Analytics & Outcome Predictor

## Overview
End-to-end sports analytics platform built on 180K+ football records.
Predicts match outcomes, rates players, clusters archetypes, and scores injury risk.

## Results
| Model | Algorithm | Result |
|---|---|---|
| Match outcome classifier | XGBoost | 50.77% accuracy (+6.36% over baseline) |
| Player performance rater | TensorFlow NN | MAE 6.33 rating points |
| Player archetype clusterer | K-Means (k=6) | Silhouette 0.21, 10,582 players |
| Injury risk scorer | Random Forest | ROC-AUC 0.894 |

## Datasets
| Dataset | Records | Use |
|---|---|---|
| European Soccer DB (Kaggle) | 25,979 matches | Match prediction, player stats |
| FiveThirtyEight SPI | 10,710 matches | ELO/SPI features, xG |
| StatsBomb Open Data | EPL via API | Match events |

## ML Features
- Rolling form (last 5 games)
- Fixture density (games in last 14 days)
- ELO delta (SPI rating difference)
- Home advantage flag
- Betting odds (Bet365)

## Stack
Python, Pandas, Scikit-learn, XGBoost, TensorFlow, MLflow, FastAPI, Streamlit, Docker

## Project Structure
athleteiq/
├── data/raw/               # raw datasets
├── data/processed/         # cleaned parquet files
├── notebooks/              # EDA + experiments
├── src/data/               # ingest, clean, features
├── src/models/             # 4 ML models
├── api/                    # FastAPI
├── dashboard/              # Streamlit
├── models/                 # saved model files
└── tests/
## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/data/ingest.py
python src/data/clean.py
python src/data/features.py
python src/models/match_predictor.py
python src/models/player_rater.py
python src/models/archetype_clusterer.py
python src/models/injury_risk.py
```

## API Endpoints
| Method | Endpoint | Output |
|---|---|---|
| POST | /predict-match | Win/Draw/Loss probability + feature importance |
| GET | /rate-player/{player_id} | Performance score + stat breakdown |
| GET | /cluster-archetypes | All players with archetype + PCA coordinates |
| GET | /top-performers | Ranked player list by position/metric |

## Dashboard Screenshots

### Match Predictor
![Match Predictor](https://raw.githubusercontent.com/yourusername/athleteiq/main/assets/tab1_match_predictor.png)

### Player Rater
![Player Rater](https://raw.githubusercontent.com/yourusername/athleteiq/main/assets/tab2_player_rater.png)

### Archetypes
![Archetypes](https://raw.githubusercontent.com/yourusername/athleteiq/main/assets/tab3_archetypes.png)

### Top Performers
![Top Performers](https://raw.githubusercontent.com/yourusername/athleteiq/main/assets/tab4_top_performers.png)

## MLflow
```bash
mlflow ui
```
Open http://localhost:5000