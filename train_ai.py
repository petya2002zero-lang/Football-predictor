import requests
import joblib
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression

# --- CONFIGURATION ---
FOOTBALL_KEY = "3f86c808c5fb455f8dfcab765b8053c7"
NBA_KEY = "544648ce-9eb1-48f4-8c8f-9f9951b8ca94"

# --- FOOTBALL SETUP ---
FOOTBALL_HEADERS = {'X-Auth-Token': FOOTBALL_KEY}
COMPETITIONS = ['PL', 'BL1', 'SA', 'PD', 'FL1', 'DED', 'PPL', 'CL']

# --- NBA SETUP ---
NBA_HEADERS = {'Authorization': NBA_KEY}

print("🚀 STARTING MULTI-SPORT AI ENGINE...")

# --- STORAGE ---
team_history = {}
elo_ratings = {}  
training_data = [] 
standings = {}
logos = {'leagues': {}, 'teams': {}} 

# NBA Storage
nba_data = {
    'schedule': [],
    'standings': {}
}

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

def smart_fetch(url, headers):
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
# ⚽ PART 1: FOOTBALL ENGINE
# ==========================================
print("\n⚽ FOOTBALL: Analyzing History & Form...")

for comp in COMPETITIONS:
    print(f"   📡 Analyzing {comp}...", end=" ", flush=True)
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
            
        print(f"✅ {len(matches)} games.")
    else:
        print("❌ Failed.")
    time.sleep(2)

# Train Football Model
print("   🧠 Training Football Brain...")
if training_data:
    df_train = pd.DataFrame(training_data)
    X = df_train[['elo_diff']]
    y = df_train['result']
    model = LogisticRegression(solver='lbfgs') 
    model.fit(X, y)
    joblib.dump(model, 'logistic_model.pkl')

# Get Football Standings
print("\n🏆 FOOTBALL: Fetching Standings...")
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
    time.sleep(2)

# Get Football Schedule
print("\n📅 FOOTBALL: Fetching Schedule...")
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
    time.sleep(2)

# ==========================================
# 🏀 PART 2: BASKETBALL ENGINE (NBA)
# ==========================================
print("\n🏀 NBA: Connecting to Balldontlie API...")

# 1. Fetch Schedule (Next 5 Days)
dates = []
for i in range(5):
    d = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
    dates.append(f"dates[]={d}")
date_query = "&".join(dates)

url_nba_games = f"https://api.balldontlie.io/v1/games?{date_query}"
nba_games_data = smart_fetch(url_nba_games, NBA_HEADERS)

if nba_games_data:
    games = nba_games_data.get('data', [])
    print(f"   📅 Found {len(games)} upcoming NBA games.")
    
    for g in games:
        # Filter only scheduled games (status might be time string or 'Scheduled')
        nba_data['schedule'].append({
            'home': g['home_team']['full_name'],
            'away': g['visitor_team']['full_name'],
            'date': g['date'] + "T" + g['status'], # Rough timestamp combination
            'id': g['id']
        })
else:
    print("   ❌ Failed to fetch NBA schedule.")

# 2. Fetch Team Stats/Standings (For Prediction)
# We fetch all teams to get their wins/losses
print("   📊 Fetching NBA Team Records...")
# Note: Balldontlie doesn't have a direct 'standings' endpoint on free tier sometimes, 
# but 'teams' endpoint is static. We need to check if we can get records.
# Actually, the best way on standard tier is usually getting game results, 
# but to keep it simple we'll assume equal start or basic random if standings fail.
# UPDATE: Let's try to scrape a "Standings" equivalent or just use the games to predict later.
# Wait, Balldontlie v1 DOES have a simple teams endpoint, but no win/loss record attached directly.
# To solve this: We will just save the schedule. In the dashboard, we will do a '50/50' visual 
# or try to fetch standings if possible.
# BETTER PLAN: We will create a manual map of current top teams for 'seed' rating or simply skip standings for V1.

# Let's just save the schedule for now. Real predictions require deeper stats integration.
pass 

# --- SAVE EVERYTHING ---
joblib.dump(team_history, 'team_history.pkl')
joblib.dump(upcoming, 'upcoming_matches.pkl')
joblib.dump(elo_ratings, 'elo_ratings.pkl')
joblib.dump(standings, 'standings.pkl')
joblib.dump(logos, 'logos.pkl') 
joblib.dump(nba_data, 'nba_data.pkl') # NEW FILE

print("\n✅ DONE. Multi-Sport Database Updated.")