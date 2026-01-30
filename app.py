import streamlit as st
import pandas as pd
from model import calculate_xg, MonteCarloEngine

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Football Predictor", layout="centered")

st.title("⚽ Monte Carlo Match Predictor")
st.markdown("Enter team stats below to simulate the match 10,000 times.")

# --- SIDEBAR: LEAGUE SETTINGS ---
st.sidebar.header("League Averages")
league_avg_home = st.sidebar.number_input("Avg Home Goals", value=1.58, step=0.01)
league_avg_away = st.sidebar.number_input("Avg Away Goals", value=1.19, step=0.01)

# --- MAIN INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_name = st.text_input("Home Team Name", "Everton")
    h_played = st.number_input("Games Played (Home)", value=11, min_value=1)
    h_scored = st.number_input("Goals Scored (Home)", value=14, min_value=0)
    h_conceded = st.number_input("Goals Conceded (Home)", value=15, min_value=0)

with col2:
    st.subheader("✈️ Away Team")
    away_name = st.text_input("Away Team Name", "Leeds Utd")
    a_played = st.number_input("Games Played (Away)", value=11, min_value=1)
    a_scored = st.number_input("Goals Scored (Away)", value=11, min_value=0)
    a_conceded = st.number_input("Goals Conceded (Away)", value=24, min_value=0)

# --- RUN SIMULATION ---
if st.button("🚀 Run Monte Carlo Simulation"):
    
    # 1. Calculate xG using the math from model.py
    h_xg, a_xg = calculate_xg(h_scored, h_conceded, h_played, 
                              a_scored, a_conceded, a_played, 
                              league_avg_home, league_avg_away)
    
    # 2. Run the Engine
    engine = MonteCarloEngine(h_xg, a_xg)
    results = engine.run_simulation()
    
    # --- DISPLAY RESULTS ---
    st.divider()
    
    # xG Display
    c1, c2 = st.columns(2)
    c1.metric(f"{home_name} xG", f"{h_xg:.2f}")
    c2.metric(f"{away_name} xG", f"{a_xg:.2f}")
    
    st.subheader("📊 Match Probabilities")
    
    # Winner Bars
    win_data = pd.DataFrame({
        'Outcome': [home_name, 'Draw', away_name],
        'Probability': [results['Home Win'], results['Draw'], results['Away Win']]
    })
    st.bar_chart(win_data.set_index('Outcome'))
    
    # Detailed Stats Table
    st.write("### 💰 Fair Odds Calculation")
    
    odds_data = {
        "Outcome": [f"{home_name} Win", "Draw", f"{away_name} Win", "Over 2.5 Goals", "Both Teams Score"],
        "Probability": [
            f"{results['Home Win']*100:.1f}%",
            f"{results['Draw']*100:.1f}%",
            f"{results['Away Win']*100:.1f}%",
            f"{results['Over 2.5']*100:.1f}%",
            f"{results['BTTS']*100:.1f}%"
        ],
        "Fair Odds (Decimal)": [
            f"{1/results['Home Win']:.2f}",
            f"{1/results['Draw']:.2f}",
            f"{1/results['Away Win']:.2f}",
            f"{1/results['Over 2.5']:.2f}",
            f"{1/results['BTTS']:.2f}"
        ]
    }
    st.table(pd.DataFrame(odds_data))