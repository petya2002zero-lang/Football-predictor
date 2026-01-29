import requests
import joblib
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression

# --- CONFIGURATION ---
API_KEY = "3f86c808c5fb455f8dfcab765b8053c7"
HEADERS = {'X-Auth-Token': API_KEY}

COMPETITIONS = ['PL', 'BL1', 'SA', 'PD', 'FL1', 'DED', 'PPL', 'CL']

print("🚀 STARTING AI ENGINE (With Logos)...")

# --- DATA STRUCTURES ---
team_history = {}
elo_ratings = {}  
training_data = [] 
standings = {}
logos = {'leagues': {}, 'teams': {}} # NEW: Stores Image URLs

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
        team_history[team_name] = {
            'home': {'scored': [], 'conceded': []},
            'away': {'scored': [], 'conceded': []},
            'all':  {'scored': [], 'conceded': []}
        }
        elo_ratings[team_name] = 1500 

def smart_fetch(url):
    for i in range(3):
        try:
            res = requests.get(url, headers=HEADERS)
            if res.status_code == 200: return res.json()
            if res.status_code == 429:
                print(f"      ⏳ Rate limit... waiting 60s.")
                time.sleep(60)
        except: pass
        time.sleep(5)
    return None

# --- 1. HISTORICAL DATA ---
print("\n📚 PHASE 1: Training Models & Calculating Elo...")

for comp in COMPETITIONS:
    print(f"   📡 Analyzing {comp}...", end=" ", flush=True)
    url = f"https://api.football-data.org/v4/competitions/{comp}/matches?status=FINISHED"
    data = smart_fetch(url)
    
    if data:
        matches = data.get('matches', [])
        matches.sort(key=lambda x: x['utcDate'])
        
        # Capture League Logo
        if 'competition' in data and 'emblem' in data['competition']: # specific endpoint structure might vary
             pass 
        # Actually easier to get it from match objects usually
        
        for m in matches:
            if m['score']['fullTime']['home'] is None: continue
            
            home = m['homeTeam']['name']
            away = m['awayTeam']['name']
            
            # --- CAPTURE LOGOS (NEW) ---
            logos['teams'][home] = m['homeTeam']['crest']
            logos['teams'][away] = m['awayTeam']['crest']
            logos['leagues'][m['competition']['name']] = m['competition']['emblem']
            
            hg = int(m['score']['fullTime']['home'])
            ag = int(m['score']['fullTime']['away'])
            
            init_team(home)
            init_team(away)
            
            # Training Data
            if hg > ag: res = 2
            elif hg == ag: res = 1
            else: res = 0
            
            h_elo = get_elo(home)
            a_elo = get_elo(away)
            
            training_data.append({
                'elo_diff': (h_elo + 100) - a_elo,
                'result': res
            })

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

# --- 2. TRAIN BRAIN ---
print("   🧠 Training Logistic Regression Brain...")
if not training_data: exit()

df_train = pd.DataFrame(training_data)
X = df_train[['elo_diff']]
y = df_train['result']
model = LogisticRegression(solver='lbfgs') 
model.fit(X, y)

# --- 3. GET STANDINGS ---
print("\n🏆 PHASE 2: Fetching League Tables...")
for comp in COMPETITIONS:
    print(f"   📊 Standings: {comp}...", end=" ", flush=True)
    url = f"https://api.football-data.org/v4/competitions/{comp}/standings"
    data = smart_fetch(url)
    
    if data:
        try:
            table = data['standings'][0]['table']
            for row in table:
                team_name = row['team']['name']
                rank = row['position']
                standings[team_name] = rank
                # Backup logo capture
                logos['teams'][team_name] = row['team']['crest']
            print("✅ Done.")
        except:
            print("⚠️ Format Error.")
    else:
        print("❌ Failed.")
    time.sleep(2)

# --- 4. UPCOMING SCHEDULE ---
print("\n📅 PHASE 3: Fetching Schedule...")
upcoming = []
today = datetime.now().strftime('%Y-%m-%d')
future = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')

for comp in COMPETITIONS:
    print(f"   🔮 Checking {comp}...", end=" ", flush=True)
    url = f"https://api.football-data.org/v4/competitions/{comp}/matches?status=SCHEDULED&dateFrom={today}&dateTo={future}"
    data = smart_fetch(url)
    
    if data:
        matches = data.get('matches', [])
        for m in matches:
            upcoming.append({
                'home': m['homeTeam']['name'],
                'away': m['awayTeam']['name'],
                'date': m['utcDate'],
                'league': m['competition']['name']
            })
            # Capture league logo if missed
            logos['leagues'][m['competition']['name']] = m['competition']['emblem']
        print(f"✅ Found {len(matches)}.")
    else:
        print("❌.")
    time.sleep(2)

# --- SAVE ---
joblib.dump(team_history, 'team_history.pkl')
joblib.dump(upcoming, 'upcoming_matches.pkl')
joblib.dump(elo_ratings, 'elo_ratings.pkl')
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(standings, 'standings.pkl')
joblib.dump(logos, 'logos.pkl') # NEW FILE

print("\n✅ DONE. Database with Logos Updated.")