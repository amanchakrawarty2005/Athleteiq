# AthletIQ — Sports Performance Analytics & Outcome Predictor

## Project Overview
End-to-end sports analytics platform using football/soccer data.
Predicts match outcomes, rates players, clusters archetypes, and scores injury risk.

## Datasets
| Dataset | Source | Records |
|---|---|---|
| European Soccer DB | Kaggle | 25,979 matches |
| FiveThirtyEight SPI | FiveThirtyEight | 11,636 matches |
| StatsBomb Open Data | StatsBombpy | EPL (free tier) |

## ML Models
- Match outcome classifier (XGBoost)
- Player performance rater (TensorFlow)
- Player archetype clusterer (K-Means)
- Injury risk scorer (Random Forest)

## Stack
Python, Pandas, Scikit-learn, XGBoost, TensorFlow, MLflow, FastAPI, Streamlit, Docker

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/data/ingest.py
```

## Project Structure
See folder tree in repo root.