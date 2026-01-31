import requests
import joblib
import time
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression

# --- CONFIGURATION ---
FOOTBALL_KEY = os.environ.get("FOOTBALL_KEY")
NBA_KEY = os.environ.get("NBA_KEY")
NFL_KEY = os.environ.get("NFL_KEY")

FOOTBALL_HEADERS = {'X-Auth-Token': FOOTBALL_KEY}
NBA_HEADERS = {'Authorization': NBA_KEY}
NFL_HEADERS = {'Ocp-Apim-Subscription-Key': NFL_KEY}

COMPETITIONS = ['PL', 'BL1', 'SA', 'PD', 'FL1', 'DED', 'PPL', 'CL']

print("🚀 STARTING AI ENGINE & PROFIT TRACKER...")

# --- STORAGE ---
team_history = {}
elo_ratings = {}  
elo_history = {} 
training_data = [] 
standings = {}
logos = {'leagues': {}, 'teams': {}} 
bet_log = [] # Stores: {date, match, pick, odds, result, profit}

# Load existing bet log if available
try: bet_log = joblib.load('bet_log.pkl')
except: pass

nba_data = {'schedule': []}
nfl_data = {'schedule': []}

# --- MATH FUNCTIONS (Moved here for Automation) ---
def get_poisson_probs(home, away):
    if home not in team_history or away not in team_history: return 0,0,0
    h_scored = np.mean(team_history[home]['home']['scored'][-10:]) if team_history[home]['home']['scored'] else 1.5
    h_conceded = np.mean(team_history[home]['home']['conceded'][-10:]) if team_history[home]['home']['conceded'] else 1.2
    a_scored = np.mean(team_history[away]['away']['scored'][-10:]) if team_history[away]['away']['scored'] else 1.2
    a_conceded = np.mean(team_history[away]['away']['conceded'][-10:]) if team_history[away]['away']['conceded'] else 1.5
    
    h_xg = (h_scored + a_conceded) / 2
    a_xg = (a_scored + h_conceded) / 2
    h_sim = np.random.poisson(h_xg, 5000)
    a_sim = np.random.poisson(a_xg, 5000)
    return np.mean(h_sim > a_sim), np.mean(h_sim == a_sim), np.mean(h_sim < a_sim)

def get_elo(team):
    return elo_ratings.get(team, 1500)

def update_elo(home, away, h_goals, a_goals):
    K = 30
    R_home = get_elo(home)
    R_away = get_elo(away)
    E_home = 1 / (1 + 10 ** ((R_away - (R_home + 100)) / 400))
    if h_goals > a_goals: S_home = 1
    elif h_goals == a_goals: S_home = 0.5
    else: S_home = 0
    change = K * (S_home - E_home)
    elo_ratings[home] = R_home + change
    elo_ratings[away] = R_away - change
    if home not in elo_history: elo_history[home] = [1500]
    if away not in elo_history: elo_history[away] = [1500]
    elo_history[home].append(elo_ratings[home])
    elo_history[away].append(elo_ratings[away])

def init_team(team_name):
    if team_name not in team_history:
        team_history[team_name] = {'home': {'scored': [], 'conceded': []}, 'away': {'scored': [], 'conceded': []}, 'all': {'scored': [], 'conceded': []}}
        elo_ratings[team_name] = 1500 
        elo_history[team_name] = [1500]

def smart_fetch(url, headers=None):
    for i in range(3):
        try:
            res = requests.get(url, headers=headers)
            if res.status_code == 200: return res.json()
            if res.status_code == 429: time.sleep(60)
        except: pass
        time.sleep(5)
    return None

# ==========================================
# ⚽ PART 1: SOCCER ENGINE & BET RESOLVER
# ==========================================
print("\n⚽ FOOTBALL: Updating Data...")

# 1. Fetch Matches
for comp in COMPETITIONS:
    url = f"https://api.football-data.org/v4/competitions/{comp}/matches?status=FINISHED"
    data = smart_fetch(url, FOOTBALL_HEADERS)
    
    if data:
        matches = data.get('matches', [])
        matches.sort(key=lambda x: x['utcDate'])
        
        for m in matches:
            if m['score']['fullTime']['home'] is None: continue
            
            home = m['homeTeam']['name']
            away = m['awayTeam']['name']
            hg = int(m['score']['fullTime']['home'])
            ag = int(m['score']['fullTime']['away'])
            
            # Init Data
            logos['teams'][home] = m['homeTeam']['crest']
            logos['teams'][away] = m['awayTeam']['crest']
            logos['leagues'][m['competition']['name']] = m['competition']['emblem']
            init_team(home); init_team(away)
            
            # Result for Elo
            if hg > ag: res = 2
            elif hg == ag: res = 1
            else: res = 0
            
            # Save Training Data
            h_elo = get_elo(home)
            a_elo = get_elo(away)
            training_data.append({'elo_diff': (h_elo + 100) - a_elo, 'result': res})
            
            update_elo(home, away, hg, ag)
            
            # Stats Update
            team_history[home]['home']['scored'].append(hg); team_history[home]['home']['conceded'].append(ag)
            team_history[home]['all']['scored'].append(hg); team_history[home]['all']['conceded'].append(ag)
            team_history[away]['away']['scored'].append(ag); team_history[away]['away']['conceded'].append(hg)
            team_history[away]['all']['scored'].append(ag); team_history[away]['all']['conceded'].append(hg)

            # --- BET RESOLVER (New!) ---
            # Check if we have a pending bet on this match
            match_id = f"{home} vs {away}" # Simple ID
            match_date = m['utcDate'][:10]
            
            for bet in bet_log:
                if bet['status'] == 'Pending' and bet['match'] == match_id:
                    # Resolve Bet
                    actual = 'Draw'
                    if hg > ag: actual = 'Home'
                    elif ag > hg: actual = 'Away'
                    
                    bet['result'] = 'Won' if bet['pick'] == actual else 'Lost'
                    bet['status'] = 'Settled'
                    
                    # Assume $10 bet unit
                    if bet['result'] == 'Won':
                        # Profit = (Stake * Odds) - Stake
                        # Since we don't have real odds at prediction time, we simulate 1.80 for Home/Away
                        sim_odds = 1.80 
                        bet['profit'] = (10 * sim_odds) - 10 
                    else:
                        bet['profit'] = -10
                    print(f"💰 Resolved Bet: {match_id} -> {bet['result']}")

    time.sleep(1)

# 2. Train Model
if training_data:
    df_train = pd.DataFrame(training_data)
    model = LogisticRegression(solver='lbfgs') 
    model.fit(df_train[['elo_diff']], df_train['result'])
    joblib.dump(model, 'logistic_model.pkl')

# 3. Schedule & New Bets
print("📅 Generating Predictions & Diamond Picks...")
upcoming = []
today_str = datetime.now().strftime('%Y-%m-%d')
future_str = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d') # 7 Days

for comp in COMPETITIONS:
    url = f"https://api.football-data.org/v4/competitions/{comp}/matches?status=SCHEDULED&dateFrom={today_str}&dateTo={future_str}"
    data = smart_fetch(url, FOOTBALL_HEADERS)
    if data:
        for m in data.get('matches', []):
            home = m['homeTeam']['name']
            away = m['awayTeam']['name']
            
            # Add to Upcoming List
            upcoming.append({
                'home': home, 'away': away,
                'date': m['utcDate'], 'league': m['competition']['name']
            })
            logos['leagues'][m['competition']['name']] = m['competition']['emblem']

            # --- PAPER TRADING (New!) ---
            # If match is TODAY/TOMORROW, calculate confidence
            if home in team_history and away in team_history:
                p_h, p_d, p_a = get_poisson_probs(home, away)
                
                # Combine with Elo/Logistic (Simplified for speed)
                h_elo = get_elo(home); a_elo = get_elo(away)
                log_probs = model.predict_proba([[ (h_elo + 100) - a_elo ]])[0]
                
                final_h = (p_h + log_probs[2]) / 2
                final_a = (p_a + log_probs[0]) / 2
                
                # DIAMOND TIER THRESHOLD (>70%)
                pick = None
                if final_h > 0.70: pick = 'Home'
                elif final_a > 0.70: pick = 'Away'
                
                if pick:
                    match_id = f"{home} vs {away}"
                    # Avoid duplicates
                    if not any(b['match'] == match_id for b in bet_log):
                        bet_log.append({
                            'date': m['utcDate'][:10],
                            'match': match_id,
                            'pick': pick,
                            'confidence': max(final_h, final_a),
                            'status': 'Pending',
                            'result': '-',
                            'profit': 0
                        })
                        print(f"💎 New Bet Placed: {match_id} ({pick})")

# 4. Other Sports (Basic Fetch)
# (Keep your existing NBA/NFL code here... I'll skip strictly for brevity but keep it in your file)

# --- SAVE EVERYTHING ---
joblib.dump(team_history, 'team_history.pkl')
joblib.dump(upcoming, 'upcoming_matches.pkl')
joblib.dump(elo_ratings, 'elo_ratings.pkl')
joblib.dump(elo_history, 'elo_history.pkl') 
joblib.dump(standings, 'standings.pkl')
joblib.dump(logos, 'logos.pkl') 
joblib.dump(bet_log, 'bet_log.pkl') # SAVING THE PROFIT TRACKER
# Save NBA/NFL placeholders if needed
joblib.dump(nba_data, 'nba_data.pkl') 
joblib.dump(nfl_data, 'nfl_data.pkl')

print("\n✅ DONE. Database & Bankroll Updated.")