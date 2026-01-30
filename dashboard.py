import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Ensemble Predictor", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    /* Main Card */
    .match-card { background-color: #262730; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #444; }
    
    /* Layout Helpers */
    .team-row { display: flex; align-items: center; margin-bottom: 8px; }
    .team-img { width: 25px; height: 25px; margin-right: 10px; object-fit: contain; }
    .league-img { width: 18px; height: 18px; margin-right: 8px; vertical-align: middle; }
    
    /* Badges */
    .elo-tag { background-color: #ffd700; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 11px; margin-left: 8px; }
    .rank-tag { background-color: #444; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-right: 8px; min-width: 25px; text-align: center; display: inline-block; }
    .form-box { margin-left: 10px; font-size: 10px; letter-spacing: 2px; display: inline-block; }
    
    /* Text */
    .team-name { font-size: 18px; font-weight: bold; margin: 0; }
    .xg-text { font-size: 12px; color: #aaa; margin-left: 45px; margin-top: -5px; display: block; }
    .xg-val-h { color: #00cc96; font-weight: bold; }
    .xg-val-a { color: #ef553b; font-weight: bold; }
    
    /* Header Info */
    .match-date { font-size: 13px; color: #bbb; display: block; margin-top: 4px; }
    .league-title { font-size: 12px; font-weight: bold; color: #fff; text-transform: uppercase; display: flex; align-items: center; }

    /* Stats & Value */
    .stat-box { background-color: #1e1e1e; padding: 8px; border-radius: 6px; text-align: center; margin-bottom: 6px; border: 1px solid #333; }
    .stat-label { font-size: 11px; color: #aaa; }
    .stat-value { font-size: 14px; font-weight: bold; color: #fff; }
    
    /* Value Badges (Green vs Red) */
    .value-badge { background-color: #00cc96; color: black; padding: 5px; border-radius: 5px; font-weight: bold; text-align: center; font-size: 14px; margin-top: 10px; }
    .no-value-badge { background-color: #ff4b4b; color: white; padding: 5px; border-radius: 5px; font-weight: bold; text-align: center; font-size: 14px; margin-top: 10px; }
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
except:
    st.error("⚠️ Models missing. Run 'python train_ai.py' first.")
    st.stop()

# --- HELPER: GET FORM ---
def get_form_html(team):
    if team not in history: return ""
    scored = history[team]['all']['scored'][-5:]
    conceded = history[team]['all']['conceded'][-5:]
    html = ""
    for s, c in zip(scored, conceded):
        if s > c: html += "✅"
        elif s == c: html += "➖"
        else: html += "❌"
    return html

# --- PREDICTION ENGINES ---
def get_poisson_probs(home, away):
    h_scored = np.mean(history[home]['home']['scored'][-10:]) if history[home]['home']['scored'] else 1.5
    h_conceded = np.mean(history[home]['home']['conceded'][-10:]) if history[home]['home']['conceded'] else 1.2
    a_scored = np.mean(history[away]['away']['scored'][-10:]) if history[away]['away']['scored'] else 1.2
    a_conceded = np.mean(history[away]['away']['conceded'][-10:]) if history[away]['away']['conceded'] else 1.5
    
    h_xg = (h_scored + a_conceded) / 2
    a_xg = (a_scored + h_conceded) / 2
    
    h_sim = np.random.poisson(h_xg, 10000)
    a_sim = np.random.poisson(a_xg, 10000)
    
    p_home = np.mean(h_sim > a_sim)
    p_draw = np.mean(h_sim == a_sim)
    p_away = np.mean(h_sim < a_sim)
    
    p_over_2_5 = np.mean((h_sim + a_sim) > 2.5)
    p_btts = np.mean((h_sim > 0) & (a_sim > 0))
    
    return p_home, p_draw, p_away, h_xg, a_xg, p_over_2_5, p_btts

def get_logistic_probs(home, away):
    h_elo = elo_ratings.get(home, 1500)
    a_elo = elo_ratings.get(away, 1500)
    features = pd.DataFrame([[ (h_elo + 100) - a_elo ]], columns=['elo_diff'])
    probs = model.predict_proba(features)[0]
    return probs[2], probs[1], probs[0] 

def get_elo_probs(home, away):
    h_elo = elo_ratings.get(home, 1500) + 100 
    a_elo = elo_ratings.get(away, 1500)
    prob_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
    prob_a = 1 / (1 + 10 ** ((h_elo - a_elo) / 400))
    prob_d = 1 - (prob_h + prob_a) 
    if prob_d < 0: prob_d = 0.25 
    total = prob_h + prob_a + prob_d
    return prob_h/total, prob_d/total, prob_a/total

# --- UI ---
st.title("🧠 Advanced AI Predictor")
st.caption("Ensemble Model + Form Guide + Value Detector + Official Badges")

# --- FILTER ---
leagues = sorted(list(set([m['league'] for m in upcoming])))
default_selection = []
if "Premier League" in leagues: default_selection.append("Premier League")
if "Bundesliga" in leagues: default_selection.append("Bundesliga")
if not default_selection: default_selection = leagues[:2]

sel_league = st.multiselect("Filter League", leagues, default=default_selection)

st.divider()

for match in upcoming:
    if match['league'] not in sel_league: continue
    
    home = match['home']
    away = match['away']
    
    if home not in history or away not in history: continue
    
    # Run Models
    p_home, p_draw, p_away, h_xg, a_xg, p_over, p_btts = get_poisson_probs(home, away)
    l_home, l_draw, l_away = get_logistic_probs(home, away)
    e_home, e_draw, e_away = get_elo_probs(home, away)
    
    final_home = (p_home + l_home + e_home) / 3 * 100
    final_draw = (p_draw + l_draw + e_draw) / 3 * 100
    final_away = (p_away + l_away + e_away) / 3 * 100
    
    # Convert probabilities to percentages for Over/BTTS
    prob_over = p_over * 100
    prob_btts = p_btts * 100

    # Metadata
    h_elo_val = int(elo_ratings.get(home, 1500))
    a_elo_val = int(elo_ratings.get(away, 1500))
    h_rank = standings.get(home, "-")
    a_rank = standings.get(away, "-")
    h_form = get_form_html(home)
    a_form = get_form_html(away)
    
    h_logo = logos['teams'].get(home, "")
    a_logo = logos['teams'].get(away, "")
    l_logo = logos['leagues'].get(match['league'], "")

    # Date
    try:
        dt_utc = datetime.strptime(match['date'], "%Y-%m-%dT%H:%M:%SZ")
        dt_cet = dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Berlin"))
        date_str = dt_cet.strftime("%d %b %H:%M") 
    except:
        date_str = match['date']

    # --- CARD LAYOUT ---
    with st.container():
        c1, c2, c3 = st.columns([3, 2, 2])
        
        # Col 1: Match Info
        with c1:
            st.markdown(f"""
                <div class='league-title'>
                    <img src='{l_logo}' class='league-img'> {match['league']}
                </div>
                <span class='match-date'>📅 {date_str} (CET)</span>
            """, unsafe_allow_html=True)
            
            st.write("") 

            # Home
            st.markdown(f"""
                <div class='team-row'>
                    <span class='rank-tag'>#{h_rank}</span>
                    <img src='{h_logo}' class='team-img'>
                    <span class='team-name'>{home}</span>
                    <span class='elo-tag'>{h_elo_val}</span>
                    <span class='form-box'>{h_form}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<span class='xg-text'>xG: <span class='xg-val-h'>{h_xg:.2f}</span></span>", unsafe_allow_html=True)
            
            # Away
            st.markdown(f"""
                <div class='team-row'>
                    <span class='rank-tag'>#{a_rank}</span>
                    <img src='{a_logo}' class='team-img'>
                    <span class='team-name'>{away}</span>
                    <span class='elo-tag'>{a_elo_val}</span>
                    <span class='form-box'>{a_form}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<span class='xg-text'>xG: <span class='xg-val-a'>{a_xg:.2f}</span></span>", unsafe_allow_html=True)

        # Col 2: Win Probs
        with c2:
            st.markdown("**Home Win**")
            st.progress(int(final_home))
            st.caption(f"{final_home:.1f}%")
            
            st.markdown("**Away Win**")
            st.progress(int(final_away))
            st.caption(f"{final_away:.1f}%")
            
            st.markdown("**Draw**")
            st.progress(int(final_draw))
            st.caption(f"{final_draw:.1f}%")

        # Col 3: Stats
        with c3:
            st.write("") 
            st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-label">Over 2.5 Goals</div>
                    <div class="stat-value">{prob_over:.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Both Teams Score</div>
                    <div class="stat-value">{prob_btts:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        # --- VALUE DETECTOR (RED/GREEN LOGIC) ---
        with st.expander("💰 Hunt for Value"):
            # Helper Function for Badges
            def check_value(odds, prob):
                if odds <= 1.0: return
                implied = (1 / odds) * 100
                edge = prob - implied
                if edge > 0:
                    st.markdown(f"<div class='value-badge'>💚 VALUE (+{edge:.1f}%)</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='no-value-badge'>🔻 NO VALUE ({edge:.1f}%)</div>", unsafe_allow_html=True)

            # Row 1: Match Result
            st.markdown("**Match Result Odds**")
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                o_h = st.number_input("Home", 0.0, step=0.01, key=f"oh_{home}{away}")
                if o_h > 1.0: check_value(o_h, final_home)
            with vc2:
                o_d = st.number_input("Draw", 0.0, step=0.01, key=f"od_{home}{away}")
                if o_d > 1.0: check_value(o_d, final_draw)
            with vc3:
                o_a = st.number_input("Away", 0.0, step=0.01, key=f"oa_{home}{away}")
                if o_a > 1.0: check_value(o_a, final_away)

            st.write("") # Spacer

            # Row 2: Goals & BTTS
            st.markdown("**Goals & BTTS Odds**")
            vc4, vc5 = st.columns(2)
            with vc4:
                o_over = st.number_input("Over 2.5", 0.0, step=0.01, key=f"o25_{home}{away}")
                if o_over > 1.0: check_value(o_over, prob_over)
            with vc5:
                o_btts = st.number_input("BTTS (Yes)", 0.0, step=0.01, key=f"btts_{home}{away}")
                if o_btts > 1.0: check_value(o_btts, prob_btts)

        st.divider()