import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Models
model_home = joblib.load('home_goals_model.pkl')
model_away = joblib.load('away_goals_model.pkl')
team_history = joblib.load('team_history.pkl') # <-- NEW FILE

st.title("🤖 AI-Powered xG Predictor")

# ... (Previous code remains the same)

# Load data
team_history = joblib.load('team_history.pkl')

# Get list of teams and SORT them alphabetically
teams = sorted(team_history.keys())

st.title("🤖 AI Super-League Predictor")

col1, col2 = st.columns(2)
with col1:
    # 'index=None' makes the box empty at start so you can type
    home_team = st.selectbox("Home Team", teams, index=None, placeholder="Type to search...")
with col2:
    away_team = st.selectbox("Away Team", teams, index=None, placeholder="Type to search...")

if st.button("Predict Match") and home_team and away_team:
    # 1. CALCULATE CURRENT FORM
    window = 3
    
    # Helper function to get avg of last 3 games
    def get_form(team_name):
        if team_name in team_history:
            last_scored = team_history[team_name]['scored'][-window:]
            last_conceded = team_history[team_name]['conceded'][-window:]
            return np.mean(last_scored), np.mean(last_conceded)
        return 1.5, 1.5 # Default if team is new

    h_att, h_def = get_form(home_team)
    a_att, a_def = get_form(away_team)
    
    # 2. ASK THE AI
    input_data = pd.DataFrame([[h_att, h_def, a_att, a_def]], 
                              columns=['Home_Form_Goals', 'Home_Form_Conceded', 
                                       'Away_Form_Goals', 'Away_Form_Conceded'])
    
    pred_home_xg = model_home.predict(input_data)[0]
    pred_away_xg = model_away.predict(input_data)[0]
    
    # 3. MONTE CARLO SIMULATION
    home_goals_sim = np.random.poisson(pred_home_xg, 10000)
    away_goals_sim = np.random.poisson(pred_away_xg, 10000)
    
    home_win_prob = np.sum(home_goals_sim > away_goals_sim) / 100
    away_win_prob = np.sum(home_goals_sim < away_goals_sim) / 100
    draw_prob = np.sum(home_goals_sim == away_goals_sim) / 100
    
    # 4. DISPLAY RESULTS
    st.subheader(f"Projected Score: {home_team} {pred_home_xg:.2f} - {pred_away_xg:.2f} {away_team}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Home Win", f"{home_win_prob}%")
    c2.metric("Draw", f"{draw_prob}%")
    c3.metric("Away Win", f"{away_win_prob}%")