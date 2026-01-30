import requests
import joblib
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression

# --- CONFIGURATION ---
FOOTBALL_KEY = "3f86c808c5fb455f8dfcab765b8053c7" # Soccer
NBA_KEY = "544648ce-9eb1-48f4-8c8f-9f9951b8ca94"      # Basketball
NFL_KEY = "44ca299f-090c-4292-b873-f4633452c016"      # American Football

# --- HEADERS ---
FOOTBALL_HEADERS = {'X-Auth-Token': FOOTBALL_KEY}
NBA_HEADERS = {'Authorization': NBA_KEY}
NFL_HEADERS = {'Ocp-Apim-Subscription-Key': NFL_KEY} # Standard format for NFL APIs

COMPETITIONS = ['PL', 'BL1', 'SA', 'PD', 'FL1', 'DED', 'PPL', 'CL']

print("🚀 STARTING MULTI-SPORT AI ENGINE (Soccer + NBA + NFL)...")

# --- STORAGE ---
team_history = {}
elo_ratings = {}  
training_data = [] 
standings = {}
logos = {'leagues': {}, 'teams': {}} 

# Sport Specific Data
nba_data = {'schedule': []}
nfl_data = {'schedule': []}

# --- HELPER FUNCTIONS ---
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

def init_team(team_name):
    if team_name not in team_history:
        team_history[team_name] = {'home': {'scored': [], 'conceded': []}, 'away': {'scored': [], 'conceded': []}, 'all': {'scored': [], 'conceded': []}}
        elo_ratings[team_name] = 1500 

def smart_fetch(url, headers=None):
    for i in range(3):
        try:
            res = requests.get(url, headers=headers)
            if res.status_code == 200: return res.json()
            if res.status_code == 429:
                print(f"      ⏳ Rate limit... waiting 60s.")
                time.sleep(60)
        except: pass
        time.sleep(5)
    return None

# ==========================================
# ⚽ PART 1: SOCCER ENGINE
# ==========================================
print("\n⚽ FOOTBALL (Soccer): Analyzing...")

for comp in COMPETITIONS:
    print(f"   📡 League {comp}...", end=" ", flush=True)
    url = f"https://api.football-data.org/v4/competitions/{comp}/matches?status=FINISHED"
    data = smart_fetch(url, FOOTBALL_HEADERS)
    
    if data:
        matches = data.get('matches', [])
        matches.sort(key=lambda x: x['utcDate'])
        
        for m in matches:
            if m['score']['fullTime']['home'] is None: continue
            
            home = m['homeTeam']['name']
            away = m['awayTeam']['name']
            logos['teams'][home] = m['homeTeam']['crest']
            logos['teams'][away] = m['awayTeam']['crest']
            logos['leagues'][m['competition']['name']] = m['competition']['emblem']
            
            hg = int(m['score']['fullTime']['home'])
            ag = int(m['score']['fullTime']['away'])
            
            init_team(home)
            init_team(away)
            
            if hg > ag: res = 2
            elif hg == ag: res = 1
            else: res = 0
            
            h_elo = get_elo(home)
            a_elo = get_elo(away)
            
            training_data.append({'elo_diff': (h_elo + 100) - a_elo, 'result': res})
            update_elo(home, away, hg, ag)
            
            team_history[home]['home']['scored'].append(hg)
            team_history[home]['home']['conceded'].append(ag)
            team_history[home]['all']['scored'].append(hg)
            team_history[home]['all']['conceded'].append(ag)
            
            team_history[away]['away']['scored'].append(ag)
            team_history[away]['away']['conceded'].append(hg)
            team_history[away]['all']['scored'].append(ag)
            team_history[away]['all']['conceded'].append(hg)
            
        print(f"✅")
    else:
        print("❌")
    time.sleep(1)

# Train Model
if training_data:
    df_train = pd.DataFrame(training_data)
    X = df_train[['elo_diff']]
    y = df_train['result']
    model = LogisticRegression(solver='lbfgs') 
    model.fit(X, y)
    joblib.dump(model, 'logistic_model.pkl')

# Soccer Standings
for comp in COMPETITIONS:
    url = f"https://api.football-data.org/v4/competitions/{comp}/standings"
    data = smart_fetch(url, FOOTBALL_HEADERS)
    if data:
        try:
            table = data['standings'][0]['table']
            for row in table:
                standings[row['team']['name']] = row['position']
                logos['teams'][row['team']['name']] = row['team']['crest']
        except: pass
    time.sleep(1)

# Soccer Schedule
upcoming = []
today_str = datetime.now().strftime('%Y-%m-%d')
future_str = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
for comp in COMPETITIONS:
    url = f"https://api.football-data.org/v4/competitions/{comp}/matches?status=SCHEDULED&dateFrom={today_str}&dateTo={future_str}"
    data = smart_fetch(url, FOOTBALL_HEADERS)
    if data:
        matches = data.get('matches', [])
        for m in matches:
            upcoming.append({
                'home': m['homeTeam']['name'],
                'away': m['awayTeam']['name'],
                'date': m['utcDate'],
                'league': m['competition']['name']
            })
            logos['leagues'][m['competition']['name']] = m['competition']['emblem']
    time.sleep(1)

# ==========================================
# 🏀 PART 2: NBA ENGINE
# ==========================================
print("\n🏀 BASKETBALL (NBA): Fetching Schedule...")
dates = []
for i in range(5):
    d = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
    dates.append(f"dates[]={d}")
date_query = "&".join(dates)

url_nba = f"https://api.balldontlie.io/v1/games?{date_query}"
nba_res = smart_fetch(url_nba, NBA_HEADERS)

if nba_res:
    games = nba_res.get('data', [])
    print(f"   ✅ Found {len(games)} NBA games.")
    for g in games:
        nba_data['schedule'].append({
            'home': g['home_team']['full_name'],
            'away': g['visitor_team']['full_name'],
            'date': g['date'] + "T" + g['status'], 
            'id': g['id']
        })
else:
    print("   ❌ Failed.")

# ==========================================
# 🏈 PART 3: NFL ENGINE (NEW)
# ==========================================
print("\n🏈 NFL: Fetching Schedule & Odds...")

# We use a reliable public endpoint for NFL scores that works reliably without complex setups
# This gets live/upcoming NFL games
url_nfl = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

nfl_res = smart_fetch(url_nfl, headers={}) # Public feed

if nfl_res:
    events = nfl_res.get('events', [])
    print(f"   ✅ Found {len(events)} NFL events.")
    
    for event in events:
        try:
            comp = event['competitions'][0]
            home_team = next(t for t in comp['competitors'] if t['homeAway'] == 'home')
            away_team = next(t for t in comp['competitors'] if t['homeAway'] == 'away')
            
            # Get Odds (Spread) if available
            odds_str = "0.0"
            if 'odds' in comp and len(comp['odds']) > 0:
                odds_str = comp['odds'][0].get('details', '0.0') # e.g. "DAL -3.5"
            
            nfl_data['schedule'].append({
                'home': home_team['team']['displayName'],
                'away': away_team['team']['displayName'],
                'home_logo': home_team['team'].get('logo', ''),
                'away_logo': away_team['team'].get('logo', ''),
                'date': event['date'], # ISO format
                'odds': odds_str,
                'home_score': home_team.get('score', '0'),
                'away_score': away_team.get('score', '0'),
                'status': event['status']['type']['state'] # pre, in, post
            })
        except:
            continue
else:
    print("   ❌ Failed.")

# --- SAVE EVERYTHING ---
joblib.dump(team_history, 'team_history.pkl')
joblib.dump(upcoming, 'upcoming_matches.pkl')
joblib.dump(elo_ratings, 'elo_ratings.pkl')
joblib.dump(standings, 'standings.pkl')
joblib.dump(logos, 'logos.pkl') 
joblib.dump(nba_data, 'nba_data.pkl') 
joblib.dump(nfl_data, 'nfl_data.pkl') # NEW FILE

print("\n✅ DONE. Database Updated.")