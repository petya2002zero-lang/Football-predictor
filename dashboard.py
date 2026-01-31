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
    
    /* Top Picks Box */
    .top-picks-box { background-color: #1e272e; padding: 15px; border-radius: 10px; border: 1px solid #0abde3; margin-bottom: 25px; }
    .top-picks-header { color: #0abde3; font-size: 20px; font-weight: bold; margin-bottom: 10px; }
    .pick-row { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding: 8px 0; }
    .pick-match { font-size: 14px; font-weight: bold; color: white; }
    .pick-badge { background-color: #0abde3; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
    
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

    .league-img { width: 40px; height: 40px; margin-right: 8px; vertical-align: middle; }
    
    /* Value Badges */
    .value-badge { background-color: #00cc96; color: black; padding: 5px; border-radius: 5px; font-weight: bold; text-align: center; font-size: 14px; margin-top: 5px; }
    .no-value-badge { background-color: #ff4b4b; color: white; padding: 5px; border-radius: 5px; font-weight: bold; text-align: center; font-size: 14px; margin-top: 5px; }

    /* Sidebar Radio Styling */
    .stRadio > label { font-size: 18px !important; font-weight: bold; }
    .disabled-sport { color: #555; margin-left: 2px; margin-top: 8px; font-size: 14px; cursor: not-allowed; display: flex; align-items: center;}
    .disabled-icon { opacity: 0.5; margin-right: 8px; font-size: 16px; }
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
    
    # --- RENAME FIX (Primera Division -> LALIGA) ---
    for m in upcoming:
        if m['league'] == "Primera Division":
            m['league'] = "LALIGA"
            
    if 'Primera Division' in logos['leagues']:
        logos['leagues']['LALIGA'] = logos['leagues'].pop('Primera Division')

except:
    st.error("⚠️ Data missing. Run 'python train_ai.py' first.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏆 Sport Selection")
    
    # Active Selection
    sport_mode = st.radio("", ["⚽ Football"])

    # Disabled Options (Visual Only)
    st.markdown("""
    <div class="disabled-sport"><span class="disabled-icon">🏀</span> Basketball (Coming Soon)</div>
    <div class="disabled-sport"><span class="disabled-icon">🏈</span> American Football (Coming Soon)</div>
    """, unsafe_allow_html=True)

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

    # --- ELO COLORS ---
    def get_elo_color(rating):
        if rating >= 1800: return "#0abde3" # Cyan (Elite)
        if rating >= 1600: return "#1dd1a1" # Green (Strong)
        if rating >= 1450: return "#ff9f43" # Orange (Average)
        return "#ff6b6b"                    # Red (Weak)

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

    # --- 🧠 TOP PREDICTIONS (DAILY FILTER) ---
    all_predictions = []
    
    # Get Today's Date in CET
    now_cet = datetime.now(ZoneInfo("Europe/Berlin"))
    today_date = now_cet.date()
    
    for match in upcoming:
        if match['home'] not in history or match['away'] not in history: continue
        
        # Parse Match Date
        try:
            m_dt = datetime.strptime(match['date'], "%Y-%m-%dT%H:%M:%SZ")
            m_date_cet = m_dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Berlin")).date()
        except: continue

        # FILTER: ONLY TODAY
        if m_date_cet != today_date:
            continue
        
        # Run Calc
        p_h, p_d, p_a, _, _, _, _ = get_poisson_probs(match['home'], match['away'])
        l_h, l_d, l_a = get_logistic_probs(match['home'], match['away'])
        e_h, e_d, e_a = get_elo_probs(match['home'], match['away'])
        
        final_home = (p_h + l_h + e_h) / 3 * 100
        final_draw = (p_d + l_d + e_d) / 3 * 100
        final_away = (p_a + l_a + e_a) / 3 * 100
        
        if final_home > final_away and final_home > final_draw:
            outcome = f"Home Win ({match['home']})"
            conf = final_home
        elif final_away > final_home and final_away > final_draw:
            outcome = f"Away Win ({match['away']})"
            conf = final_away
        else:
            outcome = "Draw"
            conf = final_draw
            
        all_predictions.append({
            'match': f"{match['home']} vs {match['away']}",
            'outcome': outcome,
            'confidence': conf,
            'league': match['league']
        })

    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    top_3 = all_predictions[:3]

    if top_3:
        st.markdown(f"""
        <div class="top-picks-box">
            <div class="top-picks-header">🔥 Top Picks for the Day ({today_date.strftime('%d %b')})</div>
        """, unsafe_allow_html=True)
        
        for pick in top_3:
            st.markdown(f"""
            <div class="pick-row">
                <span class="pick-match">{pick['match']} <span style='color:#888; font-size:12px; margin-left:10px'>({pick['league']})</span></span>
                <div>
                    <span style="color:#aaa; font-size:12px; margin-right:10px;">Predicted: {pick['outcome']}</span>
                    <span class="pick-badge">{pick['confidence']:.1f}% Confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # --- LEAGUE FILTER & SEARCH ---
    leagues = sorted(list(set([m['league'] for m in upcoming])))
    default_sel = [l for l in leagues if "Premier League" in l or "Bundesliga" in l or "LALIGA" in l]
    
    # 1. League Filter
    sel_league = st.multiselect("Filter League", leagues, default=default_sel if default_sel else leagues[:2])
    
    # 2. Search Bar
    search_team = st.text_input("🔍 Search Team", placeholder="e.g. Real Madrid, Arsenal...")
    
    st.divider()

    for match in upcoming:
        if match['league'] not in sel_league: continue
        
        # Search Filter
        if search_team:
            if search_team.lower() not in match['home'].lower() and search_team.lower() not in match['away'].lower():
                continue
        
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
            
            with st.expander("📊 Smart Insights & Value"):
                if all_insights:
                    st.markdown("**🧠 AI Smart Insights:**")
                    st.markdown("<ul class='insight-list'>" + "".join([f"<li>{i}</li>" for i in all_insights]) + "</ul>", unsafe_allow_html=True)
                else:
                    st.caption("No strong trends detected.")
                
                st.divider()
                st.markdown("**💰 Value Calculator**")
                def check_value(odds, prob):
                    if odds > 1:
                        edge = prob - (1/odds*100)
                        if edge > 0: st.markdown(f"<div class='value-badge'>💚 +{edge:.1f}%</div>", unsafe_allow_html=True)
                        else: st.markdown(f"<div class='no-value-badge'>🔻 {edge:.1f}%</div>", unsafe_allow_html=True)

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
                
                st.write("") 
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