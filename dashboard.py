import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Multi-Sport Predictor", page_icon="🏆", layout="wide")

# --- CSS STYLES ---
st.markdown("""
<style>
    /* Card Styles */
    .match-card { background-color: #262730; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #444; }
    
    /* Sport Specific Colors */
    .nfl-header { color: #5bc0de; font-weight: bold; font-size: 14px; text-transform: uppercase; }
    
    /* Layout Helpers */
    .team-row { display: flex; align-items: center; margin-bottom: 8px; }
    .team-img { width: 30px; height: 30px; margin-right: 12px; object-fit: contain; }
    .league-img { width: 18px; height: 18px; margin-right: 8px; vertical-align: middle; }
    
    /* Badges */
    .elo-tag { background-color: #ffd700; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 11px; margin-left: 8px; }
    .spread-tag { background-color: #333; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-family: monospace; border: 1px solid #555; }
    
    /* Text */
    .team-name { font-size: 18px; font-weight: bold; margin: 0; }
    .xg-text { font-size: 12px; color: #aaa; margin-left: 45px; margin-top: -5px; display: block; }
    
    /* Header Info */
    .match-date { font-size: 13px; color: #bbb; display: block; margin-top: 4px; }
    .league-title { font-size: 12px; font-weight: bold; color: #fff; text-transform: uppercase; display: flex; align-items: center; }

    /* Stats & Value */
    .stat-box { background-color: #1e1e1e; padding: 8px; border-radius: 6px; text-align: center; margin-bottom: 6px; border: 1px solid #333; }
    .stat-label { font-size: 11px; color: #aaa; }
    .stat-value { font-size: 14px; font-weight: bold; color: #fff; }
    
    .value-badge { background-color: #00cc96; color: black; padding: 5px; border-radius: 5px; font-weight: bold; text-align: center; font-size: 14px; margin-top: 10px; }
    .no-value-badge { background-color: #ff4b4b; color: white; padding: 5px; border-radius: 5px; font-weight: bold; text-align: center; font-size: 14px; margin-top: 10px; }
    
    .stRadio > label { font-size: 18px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
try:
    history = joblib.load('team_history.pkl')
    upcoming = joblib.load('upcoming_matches.pkl')
    elo_ratings = joblib.load('elo_ratings.pkl')
    model = joblib.load('logistic_model.pkl')
    standings = joblib.load('standings.pkl')
    logos = joblib.load('logos.pkl') 
    nba_data = joblib.load('nba_data.pkl') 
    nfl_data = joblib.load('nfl_data.pkl') # NEW
except:
    st.error("⚠️ Data missing. Run 'python train_ai.py' first.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏆 Sport Selection")
    sport_mode = st.radio("", ["⚽ Football", "🏀 Basketball (NBA)", "🏈 American Football (NFL)"])
    st.divider()

# ==========================================
# ⚽ FOOTBALL LOGIC
# ==========================================
if sport_mode == "⚽ Football":
    upcoming.sort(key=lambda x: x['date']) # Chronological Sort

    def get_poisson_probs(home, away):
        h_scored = np.mean(history[home]['home']['scored'][-10:]) if history[home]['home']['scored'] else 1.5
        h_conceded = np.mean(history[home]['home']['conceded'][-10:]) if history[home]['home']['conceded'] else 1.2
        a_scored = np.mean(history[away]['away']['scored'][-10:]) if history[away]['away']['scored'] else 1.2
        a_conceded = np.mean(history[away]['away']['conceded'][-10:]) if history[away]['away']['conceded'] else 1.5
        h_xg = (h_scored + a_conceded) / 2
        a_xg = (a_scored + h_conceded) / 2
        h_sim = np.random.poisson(h_xg, 10000)
        a_sim = np.random.poisson(a_xg, 10000)
        return np.mean(h_sim > a_sim), np.mean(h_sim == a_sim), np.mean(h_sim < a_sim), h_xg, a_xg, np.mean((h_sim + a_sim) > 2.5), np.mean((h_sim > 0) & (a_sim > 0))

    st.title("⚽ Football Predictor")
    leagues = sorted(list(set([m['league'] for m in upcoming])))
    default_sel = [l for l in leagues if "Premier League" in l or "Bundesliga" in l]
    sel_league = st.multiselect("Filter League", leagues, default=default_sel if default_sel else leagues[:2])
    st.divider()

    for match in upcoming:
        if match['league'] not in sel_league: continue
        home, away = match['home'], match['away']
        if home not in history or away not in history: continue
        
        p_h, p_d, p_a, h_xg, a_xg, p_o, p_b = get_poisson_probs(home, away)
        h_logo = logos['teams'].get(home, "")
        a_logo = logos['teams'].get(away, "")
        
        try:
            date_str = datetime.strptime(match['date'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Berlin")).strftime("%d %b %H:%M")
        except: date_str = match['date']

        with st.container():
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(f"<div class='league-title'>{match['league']}</div><span class='match-date'>📅 {date_str}</span>", unsafe_allow_html=True)
                st.write("")
                st.markdown(f"<div class='team-row'><img src='{h_logo}' class='team-img'><span class='team-name'>{home}</span><span class='elo-tag'>{int(elo_ratings.get(home,1500))}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<span class='xg-text'>xG: <span class='xg-val-h'>{h_xg:.2f}</span></span>", unsafe_allow_html=True)
                st.markdown(f"<div class='team-row'><img src='{a_logo}' class='team-img'><span class='team-name'>{away}</span><span class='elo-tag'>{int(elo_ratings.get(away,1500))}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<span class='xg-text'>xG: <span class='xg-val-a'>{a_xg:.2f}</span></span>", unsafe_allow_html=True)
            with c2:
                st.markdown("**Home**"); st.progress(int(p_h*100)); st.caption(f"{p_h*100:.1f}%")
                st.markdown("**Away**"); st.progress(int(p_a*100)); st.caption(f"{p_a*100:.1f}%")
                st.markdown("**Draw**"); st.progress(int(p_d*100)); st.caption(f"{p_d*100:.1f}%")
            with c3:
                st.write("")
                st.markdown(f"<div class='stat-box'><div class='stat-label'>Over 2.5</div><div class='stat-value'>{p_o*100:.1f}%</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='stat-box'><div class='stat-label'>BTTS</div><div class='stat-value'>{p_b*100:.1f}%</div></div>", unsafe_allow_html=True)
            
            with st.expander("💰 Hunt for Value"):
                vc1, vc2, vc3 = st.columns(3)
                with vc1: 
                    o = st.number_input("Home", 0.0, key=f"h{home}")
                    if o > 1: st.markdown(f"<div class='{'value-badge' if (p_h*100)-(1/o*100)>0 else 'no-value-badge'}'>Edge: {(p_h*100)-(1/o*100):.1f}%</div>", unsafe_allow_html=True)
                with vc2:
                    o = st.number_input("Draw", 0.0, key=f"d{home}")
                    if o > 1: st.markdown(f"<div class='{'value-badge' if (p_d*100)-(1/o*100)>0 else 'no-value-badge'}'>Edge: {(p_d*100)-(1/o*100):.1f}%</div>", unsafe_allow_html=True)
                with vc3:
                    o = st.number_input("Away", 0.0, key=f"a{home}")
                    if o > 1: st.markdown(f"<div class='{'value-badge' if (p_a*100)-(1/o*100)>0 else 'no-value-badge'}'>Edge: {(p_a*100)-(1/o*100):.1f}%</div>", unsafe_allow_html=True)
            st.divider()

# ==========================================
# 🏀 BASKETBALL LOGIC
# ==========================================
elif sport_mode == "🏀 Basketball (NBA)":
    st.title("🏀 NBA Predictor")
    schedule = nba_data.get('schedule', [])
    if not schedule: st.info("No games found.")
    
    for match in schedule:
        try: date_str = datetime.strptime(match['date'].split("T")[0], "%Y-%m-%d").strftime("%d %b")
        except: date_str = match['date']
        
        with st.container():
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(f"<div class='league-title'>NBA</div><span class='match-date'>📅 {date_str}</span>", unsafe_allow_html=True)
                st.markdown(f"<h3>{match['home']} vs {match['away']}</h3>", unsafe_allow_html=True)
            with c2:
                st.markdown("**Home Win**"); st.progress(50); st.caption("50%")
                st.markdown("**Away Win**"); st.progress(50); st.caption("50%")
            with c3:
                st.info("Stats Loading...")
            st.divider()

# ==========================================
# 🏈 NFL LOGIC (NEW)
# ==========================================
elif sport_mode == "🏈 American Football (NFL)":
    st.title("🏈 NFL Predictor")
    schedule = nfl_data.get('schedule', [])
    
    if not schedule:
        st.info("No live or upcoming NFL games found (Is it the off-season?).")
    
    for match in schedule:
        home = match['home']
        away = match['away']
        odds = match['odds'] # Spread
        
        # Simple Logic: If spread is -3.5 (Home favored), give home higher probability
        h_prob = 50
        try:
            # Parse spread roughly
            if "-" in odds: # Favored
                val = float(odds.split(" ")[-1])
                if val < 0: h_prob = 50 + (abs(val) * 3) # Approx 3% per point
        except: pass
        if h_prob > 95: h_prob = 95
        a_prob = 100 - h_prob

        try:
            date_str = datetime.strptime(match['date'], "%Y-%m-%dT%H:%MZ").replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Berlin")).strftime("%d %b %H:%M")
        except: date_str = match['date']

        with st.container():
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(f"<div class='nfl-header'>NFL</div><span class='match-date'>📅 {date_str} (CET)</span>", unsafe_allow_html=True)
                st.write("")
                st.markdown(f"""
                    <div class='team-row'><img src='{match['home_logo']}' class='team-img'><span class='team-name'>{home}</span></div>
                    <div class='team-row'><img src='{match['away_logo']}' class='team-img'><span class='team-name'>{away}</span></div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown("**Home Win Probability**")
                st.progress(int(h_prob))
                st.caption(f"{h_prob:.0f}%")
                
                st.markdown("**Away Win Probability**")
                st.progress(int(a_prob))
                st.caption(f"{a_prob:.0f}%")

            with c3:
                st.write("")
                st.markdown(f"""
                    <div class="stat-box"><div class="stat-label">Current Spread</div><div class="stat-value" style="color:#5bc0de">{odds}</div></div>
                    <div class="stat-box"><div class="stat-label">Status</div><div class="stat-value">{match['status']}</div></div>
                """, unsafe_allow_html=True)
            
            st.divider()