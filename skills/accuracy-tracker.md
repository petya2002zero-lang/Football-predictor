# Skill: Accuracy Tracker

## What This Skill Is For
This skill defines how we check predictions against real results and track
accuracy over time. Read this before editing `scripts/check_results.py`
or the stats page (`web/stats.html`, `web/stats.js`).

---

## Overview

After matches finish, a second GitHub Actions job runs to:
1. Fetch actual results from API-Football for yesterday's matches
2. Compare them against our predictions
3. Update the `result` field in each prediction record
4. Recalculate accuracy stats and save to `data/accuracy/accuracy-log.json`

---

## The Results Checker (`scripts/check_results.py`)

```python
"""
Fetch actual match results and compare against predictions.
Run this the morning AFTER matches have been played (e.g., 08:00 UTC).
"""

import json
import os
import requests
from datetime import date, timedelta
from pathlib import Path

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "x-rapidapi-key": os.environ.get("API_FOOTBALL_KEY"),
    "x-rapidapi-host": "v3.football.api-sports.io"
}

PREDICTIONS_DIR = Path("data/predictions")
ACCURACY_FILE = Path("data/accuracy/accuracy-log.json")


def get_results_for_date(target_date: date) -> dict:
    """
    Fetch finished match results for a given date.
    Returns dict keyed by fixture_id → {home_goals, away_goals, status}
    """
    response = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={"date": target_date.isoformat(), "status": "FT"}  # FT = Full Time
    )
    data = response.json()
    
    results = {}
    for fixture in data.get("response", []):
        fid = fixture["fixture"]["id"]
        results[fid] = {
            "home_goals": fixture["goals"]["home"],
            "away_goals": fixture["goals"]["away"],
            "status": fixture["fixture"]["status"]["short"]
        }
    return results


def determine_actual_outcome(home_goals: int, away_goals: int) -> str:
    """Convert score to outcome string."""
    if home_goals > away_goals:
        return "Home Win"
    elif home_goals == away_goals:
        return "Draw"
    else:
        return "Away Win"


def update_predictions_with_results(target_date: date):
    """
    Load predictions for target_date, fetch results, add result field.
    Saves updated predictions file back to disk.
    """
    pred_file = PREDICTIONS_DIR / f"{target_date.isoformat()}.json"
    
    if not pred_file.exists():
        print(f"No predictions file for {target_date}")
        return 0
    
    predictions = json.loads(pred_file.read_text())
    results = get_results_for_date(target_date)
    
    updated = 0
    for match in predictions:
        fid = match["match_id"]
        if fid in results and results[fid]["home_goals"] is not None:
            hg = results[fid]["home_goals"]
            ag = results[fid]["away_goals"]
            outcome = determine_actual_outcome(hg, ag)
            
            match["result"] = {
                "home_goals": hg,
                "away_goals": ag,
                "actual_score": f"{hg}-{ag}",
                "actual_outcome": outcome,
                "correct": match["prediction"]["outcome"] == outcome
            }
            updated += 1
    
    # Save updated predictions
    pred_file.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
    print(f"Updated {updated}/{len(predictions)} predictions with results")
    return updated


def rebuild_accuracy_log():
    """
    Recalculate accuracy stats from all prediction files.
    Saves to data/accuracy/accuracy-log.json
    """
    from datetime import datetime, timezone
    
    overall = {"total": 0, "correct": 0}
    by_confidence = {
        "High":   {"total": 0, "correct": 0},
        "Medium": {"total": 0, "correct": 0},
        "Low":    {"total": 0, "correct": 0}
    }
    by_league = {}
    by_outcome = {
        "Home Win": {"predicted": 0, "correct": 0},
        "Draw":     {"predicted": 0, "correct": 0},
        "Away Win": {"predicted": 0, "correct": 0}
    }
    recent_dates = []
    
    # Get all prediction files, sorted newest first
    all_files = sorted(PREDICTIONS_DIR.glob("*.json"), reverse=True)
    today = date.today()
    
    for pred_file in all_files:
        file_date = date.fromisoformat(pred_file.stem)
        predictions = json.loads(pred_file.read_text())
        
        is_recent = (today - file_date).days <= 30
        
        for match in predictions:
            result = match.get("result")
            if not result:
                continue  # Match hasn't been played yet
            
            pred_outcome = match["prediction"]["outcome"]
            confidence = match["prediction"]["confidence"]
            league = match["league"]
            correct = result["correct"]
            
            # Overall
            overall["total"] += 1
            if correct:
                overall["correct"] += 1
            
            # By confidence
            by_confidence[confidence]["total"] += 1
            if correct:
                by_confidence[confidence]["correct"] += 1
            
            # By league
            if league not in by_league:
                by_league[league] = {"total": 0, "correct": 0}
            by_league[league]["total"] += 1
            if correct:
                by_league[league]["correct"] += 1
            
            # By outcome type
            by_outcome[pred_outcome]["predicted"] += 1
            if correct:
                by_outcome[pred_outcome]["correct"] += 1
            
            # Recent 30 days
            if is_recent:
                recent_dates.append(correct)
    
    def pct(correct, total):
        return round(correct / total * 100, 1) if total > 0 else 0
    
    # Build accuracy log
    log = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "total_predictions": overall["total"],
            "correct_outcome":   overall["correct"],
            "accuracy_pct":      pct(overall["correct"], overall["total"])
        },
        "by_confidence": {
            k: {
                "total":        v["total"],
                "correct":      v["correct"],
                "accuracy_pct": pct(v["correct"], v["total"])
            }
            for k, v in by_confidence.items()
        },
        "by_league": {
            k: {
                "total":        v["total"],
                "correct":      v["correct"],
                "accuracy_pct": pct(v["correct"], v["total"])
            }
            for k, v in sorted(by_league.items(), key=lambda x: -x[1]["total"])
        },
        "by_outcome_type": {
            k: {
                "predicted":    v["predicted"],
                "correct":      v["correct"],
                "accuracy_pct": pct(v["correct"], v["predicted"])
            }
            for k, v in by_outcome.items()
        },
        "recent_30_days": {
            "total":        len(recent_dates),
            "correct":      sum(recent_dates),
            "accuracy_pct": pct(sum(recent_dates), len(recent_dates))
        }
    }
    
    ACCURACY_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACCURACY_FILE.write_text(json.dumps(log, indent=2))
    print(f"Accuracy log updated: {log['overall']['accuracy_pct']}% overall")
    return log


if __name__ == "__main__":
    yesterday = date.today() - timedelta(days=1)
    print(f"Checking results for {yesterday}...")
    update_predictions_with_results(yesterday)
    print("Rebuilding accuracy log...")
    rebuild_accuracy_log()
```

---

## Updated GitHub Actions Workflow

Add a second job to check results (runs at 08:00 UTC, after most matches finish):

```yaml
  check-results:
    runs-on: ubuntu-latest
    needs: predict          # Runs after predict job
    if: always()            # Run even if predict failed
    
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - run: pip install -r requirements.txt
      
      - name: Check results & update accuracy
        env:
          API_FOOTBALL_KEY: ${{ secrets.API_FOOTBALL_KEY }}
        run: python scripts/check_results.py
      
      - name: Commit updated results
        run: |
          git config user.name "football-predictor-bot"
          git config user.email "bot@football-predictor.com"
          git add data/predictions/ data/accuracy/
          git diff --staged --quiet || (git commit -m "📊 Results updated $(date -u +%Y-%m-%d)" && git push)
```

Or run it as a separate schedule at 08:00 UTC.

---

## Stats Page Display (`web/stats.js`)

```javascript
async function loadStats() {
  const response = await fetch('../data/accuracy/accuracy-log.json');
  const log = await response.json();
  
  // Overall accuracy
  document.getElementById('overall-pct').textContent = 
    `${log.overall.accuracy_pct}%`;
  document.getElementById('overall-count').textContent = 
    `${log.overall.correct_outcome} / ${log.overall.total_predictions} correct`;
  
  // By confidence table
  renderConfidenceTable(log.by_confidence);
  
  // By league table (sorted by total)
  renderLeagueTable(log.by_league);
  
  // Outcome accuracy bars
  renderOutcomeBars(log.by_outcome_type);
}
```
