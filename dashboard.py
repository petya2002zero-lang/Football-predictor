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
    .match-card { background-color: #262730; padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #444; }
    
    /* Sport Specific Colors */
    .nfl-header { color: #5bc0de; font-weight: bold; font-size: 14px; text-transform: uppercase; }
    
    /* Layout Helpers */
    .team-row { display: flex; align-items: center; margin-bottom: 8px; }
    .team-img { width: 30px; height: 30px; margin-right: 12px; object-fit: contain; }
    
    /* Badges */
    .elo-tag { color: #1e1e1e; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 11px; margin-left: 8px; box-shadow: 0 0 5px rgba(0,0,0,0.2); }
    .rank-tag { background-color: #444; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-right: 8px; min-width: 25px; text-align: center; display: inline-block; }
    .form-box { margin-left: 10px; font-size: 10px; letter-spacing: 2px; display: inline-block; }
    
    /* Confidence Tiers */
    .tier-diamond { background-color: #0abde3; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; border: 1px solid #00d2d3; display: inline-block; margin-bottom: 8px; }
    .tier-gold { background-color: #feca57; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; border: 1px solid #ff9f43; display: inline-block; margin-bottom: 8px; }
    .tier-silver { background-color: #c8d6e5; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; border: 1px solid #8395a7; display: inline-block; margin-bottom: 8px; }
    
    /* Text & Stats */
    .team-name { font-size: 18px; font-weight: bold; margin: 0; }
    .xg-text { font-size: 12px; color: #aaa; margin-left: 45px; margin-top: -5px; display: block; }
    .match-date { font-size: 13px; color: #bbb; display: block; margin-top: 4px; }
    .stat-box { background-color: #1e1e1e; padding: 8px; border-radius: 6px; text-align: center; margin-bottom: 6px; border: 1px solid #333; }
    .stat-label { font-size: 11px; color: #aaa; }
    .stat-value { font-size: 14px; font-weight: bold; color: #fff; }
    
    /* Smart Insights List */
    .insight-list { font-size: 13px; color: #dfe6e9; margin-top: 10px; padding-left: 20px; }
    .insight-list li { margin-bottom: 4px; }

    /* League Logo (Larger since text is gone) */
    .league-img { width: 40px; height: 40px; margin-right: 8px; vertical-align: middle; }
    
    /* Value Badges */
    .value-badge { background-color: #00cc96; color: black; padding: 5px; border-radius: 5px; font-weight: bold; text-align: center; font-size: 14px; margin-top: 5px; }
    .no-value-badge { background-color: #ff4b4b; color: white; padding: 5px; border-radius: 5px; font-weight: bold; text-align: center; font-size: 14px; margin-top: 5px; }

    .stRadio > label { font-size: 18px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
try:
    history = joblib.load('team_history.pkl')
    upcoming = joblib.load('upcoming_matches.pkl')
    elo_ratings = joblib.load('elo_ratings.pkl')
    elo_history = joblib.load('elo_history.pkl') 
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
# 🧠 SMART LOGIC
# ==========================================
def get_confidence_tier(prob_win):
    if prob_win >= 70:
        return "<span class='tier-diamond'>💎 DIAMOND TIER (High Confidence)</span>"
    elif prob_win >= 55:
        return "<span class='tier-gold'>🥇 GOLD TIER (Moderate Confidence)</span>"
    else:
        return "<span class='tier-silver'>🥈 SILVER TIER (Risky / Close Match)</span>"

def get_smart_insights(team_name):
    if team_name not in history: return []
    insights = []
    scored = history[team_name]['all']['scored'][-5:]
    conceded = history[team_name]['all']['conceded'][-5:]
    
    if not scored: return []

    wins = 0
    for s, c in zip(reversed(scored), reversed(conceded)):
        if s > c: wins += 1
        else: break
    if wins >= 3: insights.append(f"🔥 <b>{team_name}</b> is on a {wins}-game winning streak.")
    
    avg_goals = np.mean(scored)
    if avg_goals >= 2.0: insights.append(f"⚽ <b>{team_name}</b> is scoring heavily ({avg_goals:.1f} goals/game).")
    
    clean_sheets = conceded.count(0)
    if clean_sheets >= 2: insights.append(f"🛡️ <b>{team_name}</b> kept {clean_sheets} clean sheets recently.")
    
    losses = 0
    for s, c in zip(reversed(scored), reversed(conceded)):
        if s < c: losses += 1
        else: break
    if losses >= 3: insights.append(f"⚠️ <b>{team_name}</b> lost last {losses} games.")
    
    return insights

# ==========================================
# ⚽ FOOTBALL LOGIC
# ==========================================
if sport_mode == "⚽ Football":
    upcoming.sort(key=lambda x: x['date']) 

    def get_elo_color(rating):
        if rating >= 1800: return "#0abde3" 
        if rating >= 1650: return "#1dd1a1" 
        if rating >= 1500: return "#feca57" 
        if rating >= 1350: return "#ff9f43" 
        return "#ff6b6b"                    

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
        
        return np.mean(h_sim > a_sim), np.mean(h_sim == a_sim), np.mean(h_sim < a_sim), h_xg, a_xg, np.mean((h_sim + a_sim) > 2.5), np.mean((h_sim > 0) & (a_sim > 0))

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
        except: date_str = match['date']

        max_prob = max(final_home, final_away)
        tier_badge = get_confidence_tier(max_prob)
        all_insights = get_smart_insights(home) + get_smart_insights(away)

        # --- MATCH CARD ---
        with st.container():
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(tier_badge, unsafe_allow_html=True)
                # UPDATED: Text removed, only logo shown (with tooltip)
                st.markdown(f"""
                    <div class='league-title'><img src='{l_logo}' class='league-img' title='{match['league']}'></div>
                    <span class='match-date'>📅 {date_str} (CET)</span>
                """, unsafe_allow_html=True)
                
                st.write("") 
                st.markdown(f"""
                    <div class='team-row'><span class='rank-tag'>#{h_rank}</span><img src='{h_logo}' class='team-img'><span class='team-name'>{home}</span><span class='elo-tag' style='background-color:{h_elo_col}'>{h_elo_val}</span><span class='form-box'>{h_form}</span></div>
                """, unsafe_allow_html=True)
                st.markdown(f"<span class='xg-text'>xG: <span class='xg-val-h'>{h_xg:.2f}</span></span>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='team-row'><span class='rank-tag'>#{a_rank}</span><img src='{a_logo}' class='team-img'><span class='team-name'>{away}</span><span class='elo-tag' style='background-color:{a_elo_col}'>{a_elo_val}</span><span class='form-box'>{a_form}</span></div>
                """, unsafe_allow_html=True)
                st.markdown(f"<span class='xg-text'>xG: <span class='xg-val-a'>{a_xg:.2f}</span></span>", unsafe_allow_html=True)

            with c2:
                st.markdown("**Home Win**"); st.progress(int(final_home)); st.caption(f"{final_home:.1f}%")
                st.markdown("**Away Win**"); st.progress(int(final_away)); st.caption(f"{final_away:.1f}%")
                st.markdown("**Draw**"); st.progress(int(final_draw)); st.caption(f"{final_draw:.1f}%")

            with c3:
                st.write("") 
                st.markdown(f"<div class='stat-box'><div class='stat-label'>Over 2.5 Goals</div><div class='stat-value'>{prob_over:.1f}%</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='stat-box'><div class='stat-label'>BTTS</div><div class='stat-value'>{prob_btts:.1f}%</div></div>", unsafe_allow_html=True)
            
            # --- EXPANDER: INSIGHTS & VALUE ---
            with st.expander("📊 Smart Insights & Value"):
                if all_insights:
                    st.markdown("**🧠 AI Smart Insights:**")
                    st.markdown("<ul class='insight-list'>" + "".join([f"<li>{i}</li>" for i in all_insights]) + "</ul>", unsafe_allow_html=True)
                else:
                    st.caption("No strong trends detected.")
                
                st.divider()
                
                # UPDATED: Added Over/BTTS to Value Calculator
                st.markdown("**💰 Value Calculator**")
                
                def check_value(odds, prob):
                    if odds > 1:
                        edge = prob - (1/odds*100)
                        if edge > 0: st.markdown(f"<div class='value-badge'>💚 +{edge:.1f}%</div>", unsafe_allow_html=True)
                        else: st.markdown(f"<div class='no-value-badge'>🔻 {edge:.1f}%</div>", unsafe_allow_html=True)

                # Row 1: Match Result
                c_h, c_d, c_a = st.columns(3)
                with c_h: 
                    o = st.number_input("Home Odds", 0.0, key=f"h{home}")
                    check_value(o, final_home)
                with c_d:
                    o = st.number_input("Draw Odds", 0.0, key=f"d{home}")
                    check_value(o, final_draw)
                with c_a:
                    o = st.number_input("Away Odds", 0.0, key=f"a{home}")
                    check_value(o, final_away)
                
                st.write("") # Spacer
                
                # Row 2: Goals
                c_o, c_b = st.columns(2)
                with c_o:
                    o = st.number_input("Over 2.5 Odds", 0.0, key=f"o25{home}")
                    check_value(o, prob_over)
                with c_b:
                    o = st.number_input("BTTS (Yes) Odds", 0.0, key=f"btts{home}")
                    check_value(o, prob_btts)
                
                st.divider()
                st.caption("Elo Momentum (Last 15 Updates)")
                
                if home in elo_history and away in elo_history:
                    chart_data = pd.DataFrame({home: elo_history[home][-15:], away: elo_history[away][-15:]})
                    st.line_chart(chart_data)

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
    if not schedule: st.info("No games found.")
    
    for match in schedule:
        home, away, odds = match['home'], match['away'], match['odds']
        h_prob = 50
        try:
            if "-" in odds: val = float(odds.split(" ")[-1]); h_prob = 50 + (abs(val)*3) if val < 0 else 50 - (abs(val)*3)
        except: pass
        if h_prob > 95: h_prob = 95; 
        if h_prob < 5: h_prob = 5
        a_prob = 100 - h_prob

        try: date_str = datetime.strptime(match['date'], "%Y-%m-%dT%H:%MZ").replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Berlin")).strftime("%d %b %H:%M")
        except: date_str = match['date']

        with st.container():
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(f"<div class='nfl-header'>NFL</div><span class='match-date'>📅 {date_str} (CET)</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='team-row'><img src='{match['home_logo']}' class='team-img'><span class='team-name'>{home}</span></div><div class='team-row'><img src='{match['away_logo']}' class='team-img'><span class='team-name'>{away}</span></div>", unsafe_allow_html=True)
            with c2:
                st.markdown("**Home Win**"); st.progress(int(h_prob)); st.caption(f"{h_prob:.0f}%")
                st.markdown("**Away Win**"); st.progress(int(a_prob)); st.caption(f"{a_prob:.0f}%")
            with c3:
                st.write(""); st.markdown(f"<div class='stat-box'><div class='stat-label'>Spread</div><div class='stat-value' style='color:#5bc0de'>{odds}</div></div>", unsafe_allow_html=True)
            st.divider()