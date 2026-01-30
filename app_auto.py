import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

st.set_page_config(page_title="Football Predictor Pro", page_icon="⚽", layout="wide")

# --- LOAD DATA ---
try:
    team_history = joblib.load('team_history.pkl')
    teams = sorted(team_history.keys())
except FileNotFoundError:
    st.error("❌ Data missing. Run 'python train_ai.py' first.")
    st.stop()

st.title("⚽ Football Predictor Pro")

# --- SIDEBAR: GLOBAL SETTINGS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Checkbox for European Mode
    is_euro_match = st.checkbox("🇪🇺 Champions/Europa League Mode", value=False)
    
    if is_euro_match:
        st.info("Using Standard European Averages (Fairness Mode).")
        league_avg_home = 1.65 
        league_avg_away = 1.35
    else:
        # These inputs will now UPDATE the math INSTANTLY
        league_avg_home = st.number_input("League Avg Home Goals", value=1.60, step=0.05)
        league_avg_away = st.number_input("League Avg Away Goals", value=1.25, step=0.05)
        
    window = st.slider("Analyze Last N Games", 1, 30, 6)
    
    st.divider()
    st.caption("ℹ️ **Tip:** Increasing 'Avg Goals' lowers a team's Attack Rating (because scoring in a high-scoring league is easier).")

# --- HELPER FUNCTION ---
def get_stats(team_name, n_games, venue_type):
    if team_name in team_history:
        data = team_history[team_name][venue_type]
        scored = data['scored'][-n_games:]
        conceded = data['conceded'][-n_games:]
        dates = data['dates'][-n_games:]
        last_date = dates[-1].strftime('%d-%b-%Y') if dates else "N/A"
        return len(scored), sum(scored), sum(conceded), last_date
    return 0, 0, 0, "None"

# --- MAIN INTERFACE ---
col1, col2 = st.columns(2)

# --- HOME TEAM ---
with col1:
    st.subheader("🏠 Home Team")
    home_select = st.selectbox("Home Team", teams, index=None, key="home_select")
    
    if st.button(f"⬇️ LOAD {window} HOME GAMES", use_container_width=True, type="primary"):
        if home_select:
            g, s, c, d = get_stats(home_select, window, 'home')
            st.session_state['h_games'] = g
            st.session_state['h_scored'] = s
            st.session_state['h_conceded'] = c
            st.success(f"Loaded {g} Matches. Last: {d}")
            st.rerun()

    h_games = st.number_input("Home Games", key="h_games", min_value=0)
    h_scored = st.number_input("Home Goals Scored", key="h_scored", min_value=0)
    h_conceded = st.number_input("Home Goals Conceded", key="h_conceded", min_value=0)

# --- AWAY TEAM ---
with col2:
    st.subheader("✈️ Away Team")
    away_select = st.selectbox("Away Team", teams, index=None, key="away_select")

    if st.button(f"⬇️ LOAD {window} AWAY GAMES", use_container_width=True, type="primary"):
        if away_select:
            g, s, c, d = get_stats(away_select, window, 'away')
            st.session_state['a_games'] = g
            st.session_state['a_scored'] = s
            st.session_state['a_conceded'] = c
            st.success(f"Loaded {g} Matches. Last: {d}")
            st.rerun()

    a_games = st.number_input("Away Games", key="a_games", min_value=0)
    a_scored = st.number_input("Away Goals Scored", key="a_scored", min_value=0)
    a_conceded = st.number_input("Away Goals Conceded", key="a_conceded", min_value=0)

# --- AUTOMATIC PREDICTION SECTION ---
st.divider()

# Only run if we have valid data
if h_games > 0 and a_games > 0:
    
    # --- 1. MATH LOGIC ---
    # We calculate ratings relative to the League Average
    # If the League Average is higher, your attack rating goes DOWN (it's easier to score)
    h_att = (h_scored / h_games) / league_avg_home
    h_def = (h_conceded / h_games) / league_avg_away
    a_att = (a_scored / a_games) / league_avg_away
    a_def = (a_conceded / a_games) / league_avg_home
    
    h_xg = h_att * a_def * league_avg_home
    a_xg = a_att * h_def * league_avg_away
    
    # --- 2. MONTE CARLO ---
    sims = 10000
    h_sim = np.random.poisson(h_xg, sims)
    a_sim = np.random.poisson(a_xg, sims)
    
    h_win = np.sum(h_sim > a_sim) / sims * 100
    a_win = np.sum(h_sim < a_sim) / sims * 100
    draw = np.sum(h_sim == a_sim) / sims * 100
    
    # --- 3. DISPLAY ---
    st.subheader("📊 Live Prediction")
    
    # Scoreboard
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"<h1 style='text-align: center; color: #ff4b4b; font-size: 50px;'>{h_xg:.2f} - {a_xg:.2f}</h1>", unsafe_allow_html=True)
        st.caption(f"<p style='text-align: center;'>Expected Goals (xG)</p>", unsafe_allow_html=True)
    
    # Donut Chart
    source = pd.DataFrame({"Result": ["Home Win", "Draw", "Away Win"], "Prob": [h_win, draw, a_win]})
    chart = alt.Chart(source).mark_arc(outerRadius=100).encode(
        theta=alt.Theta("Prob", stack=True),
        color=alt.Color("Result", scale=alt.Scale(domain=['Home Win', 'Draw', 'Away Win'], range=['#1f77b4', '#7f7f7f', '#ff7f0e'])),
        tooltip=["Result", "Prob"],
        order=alt.Order("Prob", sort="descending")
    )
    text = chart.mark_text(radius=130).encode(
        text=alt.Text("Prob", format=".1f"),
        order=alt.Order("Prob", sort="descending")
    )
    st.altair_chart(chart + text, use_container_width=True)
    
    # Odds Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Home Win", f"{h_win:.1f}%")
    m2.metric("Draw", f"{draw:.1f}%")
    m3.metric("Away Win", f"{a_win:.1f}%")

else:
    st.info("👈 Load stats for both teams to see the prediction.")