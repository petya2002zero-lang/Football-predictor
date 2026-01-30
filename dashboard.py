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
    /* Note: Background color is now handled dynamically in Python */
    .elo-tag { color: #1e1e1e; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 11px; margin-left: 8px; box-shadow: 0 0 5px rgba(0,0,0,0.2); }
    
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
    
    /* Value Badges (Green/Red) */
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
    nfl_data = joblib.load('nfl_data.pkl') 
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

    # --- NEW: DYNAMIC ELO COLORS ---
    def get_elo_color(rating):
        if rating >= 1800: return "#0abde3" # Cyan (Elite)
        if rating >= 1650: return "#1dd1a1" # Green (Strong)
        if rating >= 1500: return "#feca57" # Yellow (Good)
        if rating >= 1350: return "#ff9f43" # Orange (Below Avg)
        return "#ff6b6b"                    # Red (Struggling)

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

    st.title("⚽ Football Predictor")
    
    leagues = sorted(list(set([m['league'] for m in upcoming])))
    default_sel = [l for l in leagues if "Premier League" in l or "Bundesliga" in l]
    sel_league = st.multiselect("Filter League", leagues, default=default_sel if default_sel else leagues[:2])
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
        
        prob_over = p_over * 100
        prob_btts = p_btts * 100

        # Metadata
        h_elo_val = int(elo_ratings.get(home, 1500))
        a_elo_val = int(elo_ratings.get(away, 1500))
        
        # Get Dynamic Colors
        h_elo_col = get_elo_color(h_elo_val)
        a_elo_col = get_elo_color(a_elo_val)

        h_rank = standings.get(home, "-")
        a_rank = standings.get(away, "-")
        h_form = get_form_html(home)
        a_form = get_form_html(away)
        
        h_logo = logos['teams'].get(home, "")
        a_logo = logos['teams'].get(away, "")
        l_logo = logos['leagues'].get(match['league'], "")

        try:
            dt_utc = datetime.strptime(match['date'], "%Y-%m-%dT%H:%M:%SZ")
            dt_cet = dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Berlin"))
            date_str = dt_cet.strftime("%d %b %H:%M") 
        except:
            date_str = match['date']

        # --- MATCH CARD ---
        with st.container():
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(f"""
                    <div class='league-title'><img src='{l_logo}' class='league-img'> {match['league']}</div>
                    <span class='match-date'>📅 {date_str} (CET)</span>
                """, unsafe_allow_html=True)
                st.write("") 
                
                # HOME ROW
                st.markdown(f"""
                    <div class='team-row'>
                        <span class='rank-tag'>#{h_rank}</span><img src='{h_logo}' class='team-img'>
                        <span class='team-name'>{home}</span>
                        <span class='elo-tag' style='background-color:{h_elo_col}'>{h_elo_val}</span>
                        <span class='form-box'>{h_form}</span>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<span class='xg-text'>xG: <span class='xg-val-h'>{h_xg:.2f}</span></span>", unsafe_allow_html=True)
                
                # AWAY ROW
                st.markdown(f"""
                    <div class='team-row'>
                        <span class='rank-tag'>#{a_rank}</span><img src='{a_logo}' class='team-img'>
                        <span class='team-name'>{away}</span>
                        <span class='elo-tag' style='background-color:{a_elo_col}'>{a_elo_val}</span>
                        <span class='form-box'>{a_form}</span>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<span class='xg-text'>xG: <span class='xg-val-a'>{a_xg:.2f}</span></span>", unsafe_allow_html=True)

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

            with c3:
                st.write("") 
                st.markdown(f"""
                    <div class="stat-box"><div class="stat-label">Over 2.5 Goals</div><div class="stat-value">{prob_over:.1f}%</div></div>
                    <div class="stat-box"><div class="stat-label">Both Teams Score</div><div class="stat-value">{prob_btts:.1f}%</div></div>
                """, unsafe_allow_html=True)
            
            with st.expander("💰 Hunt for Value"):
                # Helper Function
                def check_value(odds, prob):
                    if odds <= 1.0: return
                    implied = (1 / odds) * 100
                    edge = prob - implied
                    if edge > 0:
                        st.markdown(f"<div class='value-badge'>💚 VALUE (+{edge:.1f}%)</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='no-value-badge'>🔻 NO VALUE ({edge:.1f}%)</div>", unsafe_allow_html=True)

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

                st.write("") 
                st.markdown("**Goals & BTTS Odds**")
                vc4, vc5 = st.columns(2)
                with vc4:
                    o_over = st.number_input("Over 2.5", 0.0, step=0.01, key=f"o25_{home}{away}")
                    if o_over > 1.0: check_value(o_over, prob_over)
                with vc5:
                    o_btts = st.number_input("BTTS (Yes)", 0.0, step=0.01, key=f"btts_{home}{away}")
                    if o_btts > 1.0: check_value(o_btts, prob_btts)
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
# 🏈 NFL LOGIC
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
        
        h_prob = 50
        try:
            if "-" in odds: 
                val = float(odds.split(" ")[-1])
                if val < 0: h_prob = 50 + (abs(val) * 3) 
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