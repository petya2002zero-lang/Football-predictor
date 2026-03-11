# Skill: API-Football

## What This Skill Is For
This skill teaches you everything about how this project uses API-Football v3.
Read this before writing or editing any file in `scripts/` that touches the API.

---

## Base URL & Auth

```python
import requests
import os

BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-rapidapi-key": os.environ.get("API_FOOTBALL_KEY"),
    "x-rapidapi-host": "v3.football.api-sports.io"
}
```

Always use `os.environ.get("API_FOOTBALL_KEY")` — never hardcode the key.

---

## Rate Limits — CRITICAL

- **Pro plan: 300 requests per day**
- Each match needs approximately 4 API calls
- Always check remaining requests before a large run

```python
def get_remaining_requests(response):
    """Extract remaining API calls from response headers."""
    return int(response.headers.get("x-ratelimit-requests-remaining", 0))
```

**Rules to stay under limits:**
1. Cache team stats — if you fetched them today, don't fetch again
2. Only fetch fixtures for the next 3 days, not 7
3. Skip leagues with zero upcoming matches
4. Stop the script if remaining requests drop below 20

---

## Endpoints Used In This Project

### 1. Get Fixtures (upcoming matches)

```python
def get_fixtures(league_id: int, next_days: int = 3) -> list:
    """
    Fetch upcoming fixtures for a league.
    Returns a list of fixture objects.
    """
    response = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={
            "league": league_id,
            "season": 2025,      # Update each season
            "next": next_days    # Fixtures in next N days
        }
    )
    data = response.json()
    
    if data.get("errors"):
        print(f"API error for league {league_id}: {data['errors']}")
        return []
    
    return data.get("response", [])
```

**Key fields in each fixture:**
```
fixture.fixture.id          → unique match ID (store this)
fixture.fixture.date        → ISO datetime string
fixture.fixture.venue.name  → stadium name
fixture.fixture.referee     → referee name (can be null)
fixture.teams.home.id       → home team ID
fixture.teams.home.name     → home team name
fixture.teams.home.logo     → logo URL
fixture.teams.away.id       → away team ID
fixture.teams.away.name     → away team name
fixture.league.name         → league name
fixture.league.country      → country name
```

---

### 2. Get Team Statistics

```python
def get_team_stats(team_id: int, league_id: int, season: int = 2025) -> dict:
    """
    Fetch season statistics for a team in a specific league.
    Returns stats dict or empty dict on failure.
    """
    response = requests.get(
        f"{BASE_URL}/teams/statistics",
        headers=HEADERS,
        params={
            "team": team_id,
            "league": league_id,
            "season": season
        }
    )
    data = response.json()
    
    if not data.get("response"):
        return {}
    
    return data["response"]
```

**Key fields:**
```
stats.form                              → string like "WWDLW" (most recent last)
stats.fixtures.wins.home.total          → home wins
stats.fixtures.wins.away.total          → away wins
stats.fixtures.draws.home.total         → home draws
stats.fixtures.loses.home.total         → home losses
stats.goals.for.average.home           → avg goals scored at home
stats.goals.for.average.away           → avg goals scored away
stats.goals.against.average.home       → avg goals conceded at home
stats.goals.against.average.away       → avg goals conceded away
stats.biggest.streak.wins              → current/best win streak
```

---

### 3. Get Head-to-Head History

```python
def get_h2h(home_team_id: int, away_team_id: int, last: int = 5) -> list:
    """
    Fetch last N head-to-head matches between two teams.
    Returns list of past fixture objects.
    """
    response = requests.get(
        f"{BASE_URL}/fixtures/headtohead",
        headers=HEADERS,
        params={
            "h2h": f"{home_team_id}-{away_team_id}",
            "last": last
        }
    )
    data = response.json()
    return data.get("response", [])
```

**Summarize H2H like this:**
```python
def summarize_h2h(fixtures: list, home_id: int, away_id: int) -> str:
    home_wins = draws = away_wins = 0
    for f in fixtures:
        home_goals = f["goals"]["home"]
        away_goals = f["goals"]["away"]
        if home_goals > away_goals:
            home_wins += 1
        elif home_goals == away_goals:
            draws += 1
        else:
            away_wins += 1
    return f"Last {len(fixtures)}: {home_wins}W {draws}D {away_wins}L"
```

---

### 4. Get Injuries & Suspensions

```python
def get_injuries(team_id: int, fixture_id: int) -> list:
    """
    Fetch injury/suspension list for a team ahead of a specific fixture.
    Returns list of player objects (can be empty).
    """
    response = requests.get(
        f"{BASE_URL}/injuries",
        headers=HEADERS,
        params={
            "team": team_id,
            "fixture": fixture_id
        }
    )
    data = response.json()
    return data.get("response", [])
```

**Key fields:**
```
player.player.name      → player name
player.player.type      → "injury" or "suspension"
player.player.reason    → reason (e.g., "Knee Injury")
```

---

### 5. Get League Standings

```python
def get_standings(league_id: int, season: int = 2025) -> list:
    """
    Fetch current league standings.
    Returns list of team standing objects.
    """
    response = requests.get(
        f"{BASE_URL}/standings",
        headers=HEADERS,
        params={"league": league_id, "season": season}
    )
    data = response.json()
    
    try:
        return data["response"][0]["league"]["standings"][0]
    except (IndexError, KeyError):
        return []
```

**Key fields per team:**
```
standing.rank               → league position
standing.team.id            → team ID
standing.team.name          → team name
standing.points             → points
standing.goalsDiff          → goal difference
standing.form               → recent form string
standing.all.played         → matches played
```

---

## Covered League IDs

```python
LEAGUES = {
    # Tier 1
    "Premier League":       39,
    "La Liga":              140,
    "Bundesliga":           78,
    "Serie A":              135,
    "Ligue 1":              61,
    "Champions League":     2,
    # Tier 2
    "Europa League":        3,
    "Conference League":    848,
    "AFC Champions League": 17,
    "Eredivisie":           88,
    "Belgian Pro League":   144,
    "Primeira Liga":        94,
    "Championship":         40,
    "Saudi Pro League":     307,
    # Cups
    "FA Cup":               45,
    "Copa del Rey":         143,
    "DFB-Pokal":            81,
    "Coppa Italia":         137,
    "Coupe de France":      66,
}
```

---

## Error Handling Rules

Always wrap API calls and never let one failure crash the whole script:

```python
def safe_api_call(func, *args, **kwargs):
    """Wrapper that catches errors and returns None instead of crashing."""
    try:
        result = func(*args, **kwargs)
        return result
    except requests.exceptions.Timeout:
        print(f"Timeout calling {func.__name__}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Connection error calling {func.__name__}")
        return None
    except Exception as e:
        print(f"Unexpected error in {func.__name__}: {e}")
        return None
```

---

## Caching Rules

Store today's fetched data locally to avoid re-fetching:

```python
import json
from pathlib import Path
from datetime import date

CACHE_DIR = Path("data/cache")

def save_cache(key: str, data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path = CACHE_DIR / f"{today}_{key}.json"
    path.write_text(json.dumps(data, indent=2))

def load_cache(key: str) -> dict | None:
    today = date.today().isoformat()
    path = CACHE_DIR / f"{today}_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None
```

Usage:
```python
# Try cache first, then API
stats = load_cache(f"team_{team_id}")
if stats is None:
    stats = get_team_stats(team_id, league_id)
    save_cache(f"team_{team_id}", stats)
```
