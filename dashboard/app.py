import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

API = "http://localhost:8000"

st.set_page_config(page_title="AthletIQ", layout="wide")
st.title("AthletIQ — Sports Analytics Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Match Predictor", "Player Rater", "Archetypes", "Top Performers"
])


# ── Tab 1: Match Predictor ──────────────────────────────────────────
with tab1:
    st.header("Match Outcome Predictor")
    st.caption("Fill in match stats and betting odds to predict the outcome.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏠 Home Team")
        home_form = st.slider(
            "Home Form — avg goals scored in last 5 games (0=poor, 5=excellent)",
            0.0, 5.0, 1.5
        )
        home_density = st.slider(
            "Home Fixture Density — games played in last 14 days (higher = more fatigued)",
            0, 6, 2
        )
        b365h = st.number_input(
            "Bet365 Home Win Odds — lower = more likely to win (e.g. 1.5 = heavy favourite)",
            value=2.0, min_value=1.0
        )
        b365d = st.number_input(
            "Bet365 Draw Odds — typical range 3.0–4.0",
            value=3.4, min_value=1.0
        )

    with col2:
        st.subheader("✈️ Away Team")
        away_form = st.slider(
            "Away Form — avg goals scored in last 5 games (0=poor, 5=excellent)",
            0.0, 5.0, 1.2
        )
        away_density = st.slider(
            "Away Fixture Density — games played in last 14 days (higher = more fatigued)",
            0, 6, 2
        )
        b365a = st.number_input(
            "Bet365 Away Win Odds — lower = more likely to win (e.g. 4.0 = underdog)",
            value=3.8, min_value=1.0
        )
        elo_delta = st.number_input(
            "ELO Delta — home team SPI minus away team SPI (positive = home stronger)",
            value=0.0
        )

    st.info("💡 Tip: Betting odds encode team strength. A home odds of 1.5 vs away odds of 4.0 means home team is heavily favoured.")

    if st.button("Predict Match Outcome"):
        payload = {
            "home_form": home_form, "away_form": away_form,
            "home_fixture_density": home_density,
            "away_fixture_density": away_density,
            "elo_delta": elo_delta, "home_advantage": 1,
            "B365H": b365h, "B365D": b365d, "B365A": b365a
        }
        r = requests.post(f"{API}/predict-match", json=payload)
        data = r.json()

        result_map = {"H": "🏠 Home Win", "D": "🤝 Draw", "A": "✈️ Away Win"}
        st.subheader(f"Prediction: {result_map.get(data['prediction'], data['prediction'])}")

        col_a, col_b = st.columns(2)
        with col_a:
            probs = data["probabilities"]
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.bar(["Away", "Draw", "Home"], probs.values(),
                   color=["#e74c3c", "#3498db", "#2ecc71"])
            ax.set_ylabel("Probability")
            ax.set_title("Win/Draw/Loss Probabilities")
            st.pyplot(fig, use_container_width=False)

        with col_b:
            imp = data["feature_importance"]
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.barh(list(imp.keys()), list(imp.values()), color="#3498db")
            ax2.set_title("Feature Importance")
            st.pyplot(fig2, use_container_width=False)


# ── Tab 2: Player Rater ─────────────────────────────────────────────
with tab2:
    st.header("Player Performance Rater")
    st.caption("Enter a Player ID from the European Soccer DB to get their performance score.")

    st.markdown("""
    **How to find a Player ID:**
    - Player IDs come from the Kaggle European Soccer DB
    - Example IDs to try: `30893` (Lionel Messi), `37412`, `40636`
    - Score is out of 100 — higher is better
    """)

    player_id = st.number_input(
        "Player ID (integer from European Soccer DB)",
        value=30893, step=1,
        help="Try 30893 for Messi"
    )

    if st.button("Rate Player"):
        r = requests.get(f"{API}/rate-player/{int(player_id)}")
        if r.status_code == 200:
            data = r.json()
            st.metric(
                label="Overall Performance Score",
                value=f"{data['performance_score']:.1f} / 100",
                help="Predicted by TensorFlow neural network trained on FIFA attributes"
            )

            stats = data["stats"]
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.bar(stats.keys(), stats.values(), color="#2ecc71")
            plt.xticks(rotation=45, ha="right")
            ax.set_title("Player Stats Breakdown")
            ax.set_ylabel("Rating (0–100)")
            st.pyplot(fig, use_container_width=False)
        else:
            st.error("Player not found. Try a different ID.")


# ── Tab 3: Archetypes ───────────────────────────────────────────────
with tab3:
    st.header("Player Archetype Clusters")
    st.caption("K-Means clustering discovers 6 player archetypes from stats — no labels used.")

    st.markdown("""
    **6 Discovered Archetypes:**
    - 🔴 **Defensive Anchor** — high marking, tackling, low finishing
    - 🔵 **Box-to-Box Midfielder** — balanced across all stats
    - 🟢 **False 9** — high vision, dribbling, low heading
    - 🟡 **Target Striker** — high heading, strength, finishing
    - 🟣 **Creative Playmaker** — high vision, passing, low aggression
    - 🩵 **Pressing Forward** — high sprint speed, stamina, aggression
    """)

    if st.button("Load Archetype Scatter Plot"):
        r = requests.get(f"{API}/cluster-archetypes")
        df = pd.DataFrame(r.json())

        colors = {
            "Defensive Anchor": "#e74c3c",
            "Box-to-Box Midfielder": "#3498db",
            "False 9": "#2ecc71",
            "Target Striker": "#f39c12",
            "Creative Playmaker": "#9b59b6",
            "Pressing Forward": "#1abc9c"
        }

        fig, ax = plt.subplots(figsize=(5, 4))
        for archetype, group in df.groupby("archetype"):
            ax.scatter(group["pca_x"], group["pca_y"],
                      label=archetype, alpha=0.6,
                      color=colors.get(archetype, "#999"))
        ax.legend(fontsize=8)
        ax.set_title("Player Archetypes (PCA 2D projection)")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        st.pyplot(fig, use_container_width=False)

        st.subheader("Player List")
        st.dataframe(df[["player_name", "archetype"]].head(50))


# ── Tab 4: Top Performers ───────────────────────────────────────────
with tab4:
    st.header("Top Performers")
    st.caption("Rank players by any stat from the European Soccer DB.")

    metric = st.selectbox(
        "Select stat to rank by",
        ["finishing", "dribbling", "short_passing",
         "sprint_speed", "stamina", "vision", "marking"],
        help="Each stat is rated 0–100 in the FIFA database"
    )
    limit = st.slider("Number of players to show", 5, 20, 10)

    if st.button("Get Top Performers"):
        r = requests.get(f"{API}/top-performers?metric={metric}&limit={limit}")
        df = pd.DataFrame(r.json())

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(df["player_name"], df[metric], color="#f39c12")
        ax.set_title(f"Top {limit} Players by {metric.replace('_', ' ').title()}")
        ax.set_xlabel(f"{metric} rating (0–100)")
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=False)

        st.dataframe(df)