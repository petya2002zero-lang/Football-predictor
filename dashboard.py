import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo
import altair as alt

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Multi-Sport Predictor", page_icon="🏆", layout="wide")

# --- CSS STYLES ---
st.markdown("""
<style>
    .match-card { background-color: #262730; padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #444; }
    .top-picks-box { background-color: #1e272e; padding: 15px; border-radius: 10px; border: 1px solid #0abde3; margin-bottom: 25px; }
    .top-picks-header { color: #0abde3; font-size: 20px; font-weight: bold; margin-bottom: 10px; }
    .pick-row { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding: 8px 0; }
    .tier-diamond { background-color: #0abde3; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; border: 1px solid #00d2d3; }
    .tier-gold { background-color: #feca57; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; border: 1px solid #ff9f43; }
    .tier-silver { background-color: #c8d6e5; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; border: 1px solid #8395a7; }
    .league-img { width: 40px; height: 40px; margin-right: 8px; vertical-align: middle; }
    .team-img { width: 30px; height: 30px; margin-right: 12px; object-fit: contain; }
    .insight-list { font-size: 13px; color: #dfe6e9; margin-top: 10px; padding-left: 20px; }
    .insight-list li { margin-bottom: 4px; }
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
    bet_log = joblib.load('bet_log.pkl') # NEW
    nba_data = joblib.load('nba_data.pkl') 
    nfl_data = joblib.load('nfl_data.pkl')
    
    # Fix League Names
    for m in upcoming:
        if m['league'] == "Primera Division": m['league'] = "LALIGA"
    if 'Primera Division' in logos['leagues']: logos['leagues']['LALIGA'] = logos['leagues'].pop('Primera Division')

except:
    st.error("⚠️ Data missing. Run 'python train_ai.py' first.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🏆 AI Sports")
    page = st.radio("Menu", ["🔮 Predictions", "📈 Profit Tracker"])
    st.divider()

# ==========================================
# PAGE 1: PREDICTIONS
# ==========================================
if page == "🔮 Predictions":
    
    # Sport Selector
    with st.sidebar:
        sport_mode = st.radio("Sport", ["⚽ Football"])
        st.markdown("""<div style='color:#666; font-size:14px; margin-top:5px'>🏀 NBA (Coming Soon)<br>🏈 NFL (Coming Soon)</div>""", unsafe_allow_html=True)

    # --- FUNCTIONS ---
    def get_confidence_tier(prob_win):
        if prob_win >= 70: return "<span class='tier-diamond'>💎 DIAMOND TIER</span>"
        elif prob_win >= 55: return "<span class='tier-gold'>🥇 GOLD TIER</span>"
        else: return "<span class='tier-silver'>🥈 SILVER TIER</span>"

    def get_comparison_insights(home, away):
        """Generates 'Glass Box' Comparison Insights"""
        if home not in history or away not in history: return []
        insights = []
        
        # Data
        h_scored = np.mean(history[home]['all']['scored'][-5:])
        a_conceded = np.mean(history[away]['all']['conceded'][-5:])
        h_elo = elo_ratings.get(home, 1500)
        a_elo = elo_ratings.get(away, 1500)
        
        # 1. Attack vs Defense Mismatch
        if h_scored > 2.0 and a_conceded > 1.5:
            insights.append(f"⚡ <b>Mismatch:</b> {home}'s strong attack ({h_scored:.1f} goals/game) vs {away}'s leaky defense ({a_conceded:.1f} conceded). Expect goals.")
            
        # 2. Elo Gap
        elo_diff = h_elo - a_elo
        if elo_diff > 200:
            insights.append(f"🧠 <b>Class Difference:</b> {home} is significantly stronger (+{int(elo_diff)} Elo points).")
        elif abs(elo_diff) < 30:
            insights.append(f"⚖️ <b>Tight Match:</b> Teams are rated almost equally. Draw probability is elevated.")
            
        # 3. Form Check
        h_wins = 0
        for s, c in zip(reversed(history[home]['all']['scored'][-5:]), reversed(history[home]['all']['conceded'][-5:])):
            if s > c: h_wins += 1
            else: break
        if h_wins >= 3:
             insights.append(f"🔥 <b>Momentum:</b> {home} is on a {h_wins}-game winning streak.")

        return insights

    # --- MAIN LOGIC ---
    upcoming.sort(key=lambda x: x['date']) 
    now_cet = datetime.now(ZoneInfo("Europe/Berlin"))
    today_date = now_cet.date()

    # --- 1. TOP PICKS WIDGET ---
    daily_picks = []
    for match in upcoming:
        if match['home'] not in history: continue
        try:
            m_date = datetime.strptime(match['date'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Berlin")).date()
            if m_date != today_date: continue
        except: continue
        
        # Probabilities
        h_elo, a_elo = elo_ratings.get(match['home'], 1500), elo_ratings.get(match['away'], 1500)
        log_prob = model.predict_proba([[(h_elo+100)-a_elo]])[0]
        # Simplified for speed in widget
        conf_h = log_prob[2] * 100
        conf_a = log_prob[0] * 100
        
        if conf_h > 65: daily_picks.append({'match': f"{match['home']} vs {match['away']}", 'pick': f"Home ({match['home']})", 'conf': conf_h, 'league': match['league']})
        elif conf_a > 65: daily_picks.append({'match': f"{match['home']} vs {match['away']}", 'pick': f"Away ({match['away']})", 'conf': conf_a, 'league': match['league']})

    daily_picks.sort(key=lambda x: x['conf'], reverse=True)
    
    st.title("⚽ Football Predictor")
    
    if daily_picks:
        st.markdown(f"""<div class="top-picks-box"><div class="top-picks-header">🔥 Top Picks for the Day ({today_date.strftime('%d %b')})</div>""", unsafe_allow_html=True)
        for p in daily_picks[:3]:
            st.markdown(f"""
            <div class="pick-row">
                <span style="color:white; font-weight:bold;">{p['match']} <span style="color:#888; font-size:12px">({p['league']})</span></span>
                <div><span style="color:#aaa; font-size:12px; margin-right:10px">{p['pick']}</span><span class="tier-diamond">{p['conf']:.1f}% Conf</span></div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- 2. MATCH LIST ---
    leagues = sorted(list(set([m['league'] for m in upcoming])))
    sel_league = st.multiselect("Filter League", leagues, default=[l for l in leagues if "Premier League" in l or "Bundesliga" in l])
    search = st.text_input("🔍 Search Team")
    st.divider()

    for match in upcoming:
        if match['league'] not in sel_league: continue
        if search and (search.lower() not in match['home'].lower() and search.lower() not in match['away'].lower()): continue
        
        home, away = match['home'], match['away']
        if home not in history: continue

        # Calc Probs (Using Logic from train_ai.py style)
        h_elo, a_elo = elo_ratings.get(home, 1500), elo_ratings.get(away, 1500)
        
        # Note: In full app we'd call the Poisson function here too, approximating for display
        log_probs = model.predict_proba([[ (h_elo + 100) - a_elo ]])[0]
        final_home = log_probs[2] * 100
        final_draw = log_probs[1] * 100
        final_away = log_probs[0] * 100
        
        # Colors & badges
        tier_badge = get_confidence_tier(max(final_home, final_away))
        insights = get_comparison_insights(home, away)
        
        # Render
        with st.container():
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(tier_badge, unsafe_allow_html=True)
                st.markdown(f"<div style='font-weight:bold; font-size:18px; margin-top:5px'>{home} vs {away}</div>", unsafe_allow_html=True)
                st.caption(f"{match['league']} • {match['date'][11:16]}")
            with c2:
                st.progress(int(final_home)); st.caption(f"Home Win: {final_home:.1f}%")
                st.progress(int(final_away)); st.caption(f"Away Win: {final_away:.1f}%")
            with c3:
                with st.expander("🧠 AI Analysis"):
                    if insights:
                        st.markdown("<ul class='insight-list'>" + "".join([f"<li>{i}</li>" for i in insights]) + "</ul>", unsafe_allow_html=True)
                    else: st.caption("No specific statistical edge found.")
            st.divider()

# ==========================================
# PAGE 2: PROFIT TRACKER
# ==========================================
elif page == "📈 Profit Tracker":
    st.title("📈 AI Performance Tracker")
    st.caption("Automated tracking of all 'Diamond Tier' (>70% Confidence) bets.")
    
    if not bet_log:
        st.info("No bets have been settled yet. Check back tomorrow!")
    else:
        df_bets = pd.DataFrame(bet_log)
        
        # KPI Cards
        total_bets = len(df_bets)
        pending = len(df_bets[df_bets['status'] == 'Pending'])
        settled = df_bets[df_bets['status'] == 'Settled']
        
        profit = settled['profit'].sum() if not settled.empty else 0
        roi = (profit / (len(settled) * 10)) * 100 if not settled.empty else 0
        win_rate = (len(settled[settled['result']=='Won']) / len(settled) * 100) if not settled.empty else 0
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Bets", total_bets)
        k2.metric("Win Rate", f"{win_rate:.1f}%")
        k3.metric("Profit (Units)", f"{profit:.1f}u", delta_color="normal")
        k4.metric("ROI", f"{roi:.1f}%")
        
        st.divider()
        
        # Graph
        if not settled.empty:
            settled['cumulative_profit'] = settled['profit'].cumsum()
            chart = alt.Chart(settled).mark_line(point=True).encode(
                x='date',
                y='cumulative_profit',
                tooltip=['date', 'match', 'pick', 'result', 'profit']
            ).properties(title="Bankroll Growth (Simulation)")
            st.altair_chart(chart, use_container_width=True)
        
        # Table
        st.subheader("📜 Bet History")
        st.dataframe(df_bets[['date', 'match', 'pick', 'result', 'profit', 'status']].sort_values(by='date', ascending=False), use_container_width=True)