# Skill: Python Pipeline

## What This Skill Is For
This skill describes how the full data pipeline works — from fetching fixtures
to generating predictions and saving JSON files. Read this before editing any
script in the `scripts/` folder.

---

## Pipeline Overview

```
scripts/predict.py  ← Main entry point (this is what GitHub Actions runs)
       │
       ├── fetch_fixtures.py    Step 1: Get today's upcoming matches
       ├── fetch_stats.py       Step 2: Get stats for each match
       ├── predict.py           Step 3: Send to Claude, get predictions
       └── save_data.py         Step 4: Write predictions to JSON
```

The entire pipeline is triggered by running `python scripts/predict.py`.

---

## Main Entry Point (`scripts/predict.py`)

```python
"""
Main pipeline entry point.
Run this script to fetch fixtures, generate predictions, and save them.

Usage:
    python scripts/predict.py
    python scripts/predict.py --date 2026-03-15  # predict for a specific date
    python scripts/predict.py --league 39         # only one league (for testing)
"""

import argparse
import sys
from datetime import date, datetime
from fetch_fixtures import get_all_fixtures
from fetch_stats import enrich_with_stats
from predictor import predict_all_matches
from save_data import save_predictions
from notify import send_failure_alert

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Target date YYYY-MM-DD", default=None)
    parser.add_argument("--league", help="Single league ID for testing", default=None)
    args = parser.parse_args()
    
    target_date = date.fromisoformat(args.date) if args.date else date.today()
    
    print(f"=== Football Predictor V2 | {target_date} ===")
    
    try:
        # Step 1: Get fixtures
        print("\n[1/4] Fetching fixtures...")
        fixtures = get_all_fixtures(target_date, league_filter=args.league)
        print(f"  Found {len(fixtures)} matches")
        
        if not fixtures:
            print("  No matches found for today. Exiting.")
            sys.exit(0)
        
        # Step 2: Enrich with stats
        print("\n[2/4] Fetching stats...")
        enriched = enrich_with_stats(fixtures)
        print(f"  Enriched {len(enriched)} matches")
        
        # Step 3: Generate predictions
        print("\n[3/4] Generating predictions...")
        predictions = predict_all_matches(enriched)
        print(f"  Generated {len(predictions)} predictions")
        
        # Step 4: Save
        print("\n[4/4] Saving predictions...")
        save_predictions(predictions, target_date)
        
        print(f"\n✅ Done! {len(predictions)} predictions saved for {target_date}")
        
    except Exception as e:
        error_msg = f"Pipeline failed on {target_date}: {str(e)}"
        print(f"\n❌ {error_msg}")
        send_failure_alert(error_msg)  # Telegram notification
        raise  # Re-raise so GitHub Actions marks the run as failed

if __name__ == "__main__":
    main()
```

---

## Step 1: Fetch Fixtures (`scripts/fetch_fixtures.py`)

```python
"""
Fetches upcoming fixtures for all covered leagues.
Returns a flat list of fixture dicts ready for stat enrichment.
"""

from api_client import get_fixtures, LEAGUES
from datetime import date, timedelta

def get_all_fixtures(target_date: date, next_days: int = 1, league_filter=None) -> list:
    """
    Fetch all fixtures for the target date across all leagues.
    league_filter: optional league ID string for testing a single league.
    """
    all_fixtures = []
    leagues_to_check = LEAGUES.items()
    
    if league_filter:
        leagues_to_check = [(k, int(league_filter)) for k, v in LEAGUES.items() 
                            if v == int(league_filter)]
    
    for league_name, league_id in leagues_to_check:
        print(f"  Checking {league_name}...")
        fixtures = get_fixtures(league_id, next_days=next_days)
        
        # Filter to only target_date
        day_fixtures = [
            f for f in fixtures
            if f["fixture"]["date"].startswith(target_date.isoformat())
        ]
        
        all_fixtures.extend(day_fixtures)
    
    return all_fixtures
```

---

## Step 2: Enrich With Stats (`scripts/fetch_stats.py`)

```python
"""
Takes raw fixture objects and enriches them with team stats, H2H, and injuries.
Returns a list of match_data dicts ready for Claude.
"""

from api_client import get_team_stats, get_h2h, get_injuries, get_standings, safe_api_call
from cache import load_cache, save_cache

def enrich_with_stats(fixtures: list) -> list:
    """Enrich each fixture with all stats needed for prediction."""
    enriched = []
    
    for f in fixtures:
        try:
            match_data = build_match_data(f)
            enriched.append(match_data)
        except Exception as e:
            print(f"  Skipping {f['teams']['home']['name']} vs {f['teams']['away']['name']}: {e}")
            continue
    
    return enriched


def build_match_data(fixture: dict) -> dict:
    """Build the full match data dict from a single fixture."""
    home_id = fixture["teams"]["home"]["id"]
    away_id = fixture["teams"]["away"]["id"]
    league_id = fixture["league"]["id"]
    fixture_id = fixture["fixture"]["id"]
    
    # Get standings (use cache to save API calls)
    standings = load_cache(f"standings_{league_id}") or {}
    if not standings:
        raw = safe_api_call(get_standings, league_id)
        if raw:
            standings = {s["team"]["id"]: s for s in raw}
            save_cache(f"standings_{league_id}", standings)
    
    home_standing = standings.get(str(home_id), {})
    away_standing = standings.get(str(away_id), {})
    
    # Get team stats (cache per team per day)
    home_stats = load_cache(f"team_{home_id}") or safe_api_call(get_team_stats, home_id, league_id)
    away_stats = load_cache(f"team_{away_id}") or safe_api_call(get_team_stats, away_id, league_id)
    if home_stats: save_cache(f"team_{home_id}", home_stats)
    if away_stats: save_cache(f"team_{away_id}", away_stats)
    
    # H2H (not cached — always fresh)
    h2h = safe_api_call(get_h2h, home_id, away_id) or []
    
    # Injuries
    home_injuries = safe_api_call(get_injuries, home_id, fixture_id) or []
    away_injuries = safe_api_call(get_injuries, away_id, fixture_id) or []
    
    return {
        # Identifiers
        "match_id":      fixture_id,
        "home_team":     fixture["teams"]["home"]["name"],
        "home_team_id":  home_id,
        "home_logo":     fixture["teams"]["home"]["logo"],
        "away_team":     fixture["teams"]["away"]["name"],
        "away_team_id":  away_id,
        "away_logo":     fixture["teams"]["away"]["logo"],
        "league":        fixture["league"]["name"],
        "league_id":     league_id,
        "league_logo":   fixture["league"]["logo"],
        "country":       fixture["league"]["country"],
        "date":          fixture["fixture"]["date"],
        "venue":         fixture["fixture"].get("venue", {}).get("name", "Unknown"),
        "referee":       fixture["fixture"].get("referee", "Unknown"),
        
        # Standings
        "home_position": home_standing.get("rank", "N/A"),
        "home_points":   home_standing.get("points", "N/A"),
        "home_gd":       home_standing.get("goalsDiff", "N/A"),
        "away_position": away_standing.get("rank", "N/A"),
        "away_points":   away_standing.get("points", "N/A"),
        "away_gd":       away_standing.get("goalsDiff", "N/A"),
        
        # Form
        "home_form":         extract_form(home_stats),
        "home_scored_avg":   extract_avg_goals(home_stats, "for", "total"),
        "home_conceded_avg": extract_avg_goals(home_stats, "against", "total"),
        "home_home_w":       extract_record(home_stats, "wins", "home"),
        "home_home_d":       extract_record(home_stats, "draws", "home"),
        "home_home_l":       extract_record(home_stats, "loses", "home"),
        "home_xg":           extract_xg(home_stats, "home"),
        
        "away_form":         extract_form(away_stats),
        "away_scored_avg":   extract_avg_goals(away_stats, "for", "total"),
        "away_conceded_avg": extract_avg_goals(away_stats, "against", "total"),
        "away_away_w":       extract_record(away_stats, "wins", "away"),
        "away_away_d":       extract_record(away_stats, "draws", "away"),
        "away_away_l":       extract_record(away_stats, "loses", "away"),
        "away_xg":           extract_xg(away_stats, "away"),
        
        # H2H
        "h2h_summary": summarize_h2h(h2h, home_id, away_id),
        "h2h_detail":  format_h2h_detail(h2h),
        
        # Injuries
        "home_injuries": format_injuries(home_injuries),
        "away_injuries": format_injuries(away_injuries),
    }
```

---

## Step 3: Predict All Matches (`scripts/predictor.py`)

```python
"""
Sends each match to Claude and collects predictions.
Handles retries and skips failed matches gracefully.
"""

from predict_prompt import predict_with_retry
from datetime import datetime, timezone

def predict_all_matches(matches: list) -> list:
    """Generate predictions for all matches. Skip failures."""
    predictions = []
    
    for i, match in enumerate(matches, 1):
        print(f"  [{i}/{len(matches)}] {match['home_team']} vs {match['away_team']}...")
        
        prediction = predict_with_retry(match)
        
        if prediction is None:
            print(f"    ⚠️  Skipped (Claude failed)")
            continue
        
        # Build the full record to save
        record = {
            **{k: match[k] for k in [
                "match_id", "home_team", "home_team_id", "home_logo",
                "away_team", "away_team_id", "away_logo", "league",
                "league_id", "league_logo", "country", "date",
                "venue", "referee"
            ]},
            "prediction": prediction,
            "stats_snapshot": {
                "home_form":     match["home_form"],
                "away_form":     match["away_form"],
                "home_position": match["home_position"],
                "away_position": match["away_position"],
                "home_points":   match["home_points"],
                "away_points":   match["away_points"],
                "h2h_summary":   match["h2h_summary"],
                "home_injuries": match["home_injuries"],
                "away_injuries": match["away_injuries"],
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "result": None
        }
        
        predictions.append(record)
        print(f"    ✅ {prediction['outcome']} ({prediction['confidence']} confidence)")
    
    return predictions
```

---

## Helper Extraction Functions

```python
def extract_form(stats: dict) -> str:
    """Get form string like 'WWDLW'. Returns 'N/A' if missing."""
    if not stats:
        return "N/A"
    return stats.get("form", "N/A")[-5:] or "N/A"  # Last 5 chars

def extract_avg_goals(stats: dict, direction: str, venue: str) -> str:
    """direction: 'for' or 'against'. venue: 'home', 'away', 'total'"""
    try:
        return str(stats["goals"][direction]["average"][venue])
    except (KeyError, TypeError):
        return "N/A"

def extract_record(stats: dict, result: str, venue: str) -> str:
    """result: 'wins', 'draws', 'loses'. venue: 'home', 'away'"""
    try:
        return str(stats["fixtures"][result][venue]["total"])
    except (KeyError, TypeError):
        return "N/A"

def extract_xg(stats: dict, venue: str) -> str:
    """Extract expected goals average. Returns N/A if not available (common)."""
    try:
        return str(stats["expected_goals"][venue])
    except (KeyError, TypeError):
        return "N/A"

def format_injuries(injuries: list) -> str:
    """Format list of injury objects into readable string."""
    if not injuries:
        return "None reported"
    parts = []
    for p in injuries[:5]:  # Max 5
        name = p.get("player", {}).get("name", "Unknown")
        reason = p.get("player", {}).get("reason", "Unknown")
        parts.append(f"{name} ({reason})")
    return ", ".join(parts)

def summarize_h2h(fixtures: list, home_id: int, away_id: int) -> str:
    """Returns string like 'Last 5: 3W 1D 1L for Arsenal'"""
    if not fixtures:
        return "No H2H data available"
    hw = dr = aw = 0
    for f in fixtures:
        hg = f.get("goals", {}).get("home", 0) or 0
        ag = f.get("goals", {}).get("away", 0) or 0
        if hg > ag: hw += 1
        elif hg == ag: dr += 1
        else: aw += 1
    return f"Last {len(fixtures)}: {hw}W {dr}D {aw}L"
```

---

## Testing Individual Steps

Run these from VS Code terminal to test without GitHub Actions:

```bash
# Test just fixture fetching (Premier League)
python scripts/predict.py --league 39

# Test a specific date
python scripts/predict.py --date 2026-03-15

# Test full run
python scripts/predict.py

# Check output
cat data/predictions/$(date +%Y-%m-%d).json | python -m json.tool
```
