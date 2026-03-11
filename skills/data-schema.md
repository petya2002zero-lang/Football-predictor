# Skill: Data Schema

## What This Skill Is For
This skill defines the exact structure of every JSON file in this project.
Read this before writing any code that creates, reads, or modifies data files.
Every file saved must match these schemas — no exceptions.

---

## File Locations

```
data/
├── predictions/
│   └── YYYY-MM-DD.json        ← One file per day, all leagues
├── results/
│   └── YYYY-MM-DD-results.json ← Actual scores after matches finish
├── accuracy/
│   └── accuracy-log.json       ← Running accuracy statistics
└── cache/
    └── YYYY-MM-DD_*.json       ← Temp API cache (gitignored)
```

---

## 1. Predictions File (`data/predictions/YYYY-MM-DD.json`)

This is an **array** of match prediction objects.
One file per day. All leagues combined in one file.

```json
[
  {
    "match_id": 1035274,
    "home_team": "Arsenal",
    "home_team_id": 42,
    "home_logo": "https://media.api-sports.io/football/teams/42.png",
    "away_team": "Chelsea",
    "away_team_id": 49,
    "away_logo": "https://media.api-sports.io/football/teams/49.png",
    "league": "Premier League",
    "league_id": 39,
    "league_logo": "https://media.api-sports.io/football/leagues/39.png",
    "country": "England",
    "date": "2026-03-10T20:00:00+00:00",
    "venue": "Emirates Stadium",
    "referee": "Michael Oliver",
    "prediction": {
      "outcome": "Home Win",
      "confidence": "High",
      "home_win_pct": 58,
      "draw_pct": 22,
      "away_win_pct": 20,
      "predicted_score": "2-1",
      "score_range": "1-0 to 3-1",
      "btts": true,
      "over_2_5": true,
      "key_factors": [
        "Arsenal unbeaten in last 7 home games (W5 D2)",
        "Chelsea lost 3 of last 5 away matches",
        "H2H: Arsenal won 3 of last 5 meetings",
        "Arsenal xG at home: 2.1 avg vs Chelsea xG away: 0.9 avg"
      ],
      "analysis": "Arsenal's strong home form and superior xG numbers make them clear favourites. Chelsea's away record this season has been inconsistent and the absence of their key midfielder compounds the challenge."
    },
    "stats_snapshot": {
      "home_form": "WWDWW",
      "away_form": "LDWLD",
      "home_position": 2,
      "away_position": 5,
      "home_points": 58,
      "away_points": 44,
      "h2h_summary": "Last 5: 3W 1D 1L for Arsenal",
      "home_injuries": "None reported",
      "away_injuries": "Enzo Fernandez (Knee), Romeo Lavia (Suspension)"
    },
    "generated_at": "2026-03-09T06:12:34+00:00",
    "result": null
  }
]
```

### Field Rules:
- `match_id` — integer, from API-Football fixture ID, must be unique
- `date` — ISO 8601 with timezone offset (what API-Football returns)
- `prediction.outcome` — exactly one of: `"Home Win"`, `"Draw"`, `"Away Win"`
- `prediction.confidence` — exactly one of: `"Low"`, `"Medium"`, `"High"`
- `prediction.home_win_pct + draw_pct + away_win_pct` — must equal 100
- `prediction.predicted_score` — format `"2-1"` (home-away)
- `stats_snapshot` — raw inputs used for the prediction (for auditing)
- `generated_at` — ISO 8601 UTC timestamp when Claude generated it
- `result` — null until the match is played, then filled by check_results.py

---

## 2. Results File (`data/results/YYYY-MM-DD-results.json`)

Filled in after matches finish. Same structure as predictions but with actual scores.

```json
[
  {
    "match_id": 1035274,
    "date": "2026-03-10T20:00:00+00:00",
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "league": "Premier League",
    "actual_score": "2-0",
    "actual_outcome": "Home Win",
    "fetched_at": "2026-03-11T00:05:00+00:00"
  }
]
```

---

## 3. Accuracy Log (`data/accuracy/accuracy-log.json`)

Updated daily after results come in. Used by the website stats page.

```json
{
  "last_updated": "2026-03-11T00:10:00+00:00",
  "overall": {
    "total_predictions": 312,
    "correct_outcome": 187,
    "accuracy_pct": 59.9
  },
  "by_confidence": {
    "High":   { "total": 89,  "correct": 67, "accuracy_pct": 75.3 },
    "Medium": { "total": 145, "correct": 86, "accuracy_pct": 59.3 },
    "Low":    { "total": 78,  "correct": 34, "accuracy_pct": 43.6 }
  },
  "by_league": {
    "Premier League": { "total": 52, "correct": 33, "accuracy_pct": 63.5 },
    "La Liga":        { "total": 48, "correct": 29, "accuracy_pct": 60.4 },
    "Champions League": { "total": 22, "correct": 15, "accuracy_pct": 68.2 }
  },
  "by_outcome_type": {
    "Home Win": { "predicted": 145, "correct": 98, "accuracy_pct": 67.6 },
    "Draw":     { "predicted": 72,  "correct": 21, "accuracy_pct": 29.2 },
    "Away Win": { "predicted": 95,  "correct": 68, "accuracy_pct": 71.6 }
  },
  "recent_30_days": {
    "total": 98,
    "correct": 61,
    "accuracy_pct": 62.2
  }
}
```

---

## Python Helper: Schema Validation

Use this to validate any prediction object before writing to disk:

```python
def validate_prediction_record(record: dict) -> list[str]:
    """
    Validate a full prediction record (not just the prediction sub-object).
    Returns a list of error strings. Empty list = valid.
    """
    errors = []
    
    # Required top-level fields
    required = [
        "match_id", "home_team", "home_team_id", "away_team", "away_team_id",
        "league", "league_id", "date", "prediction", "stats_snapshot", "generated_at"
    ]
    for field in required:
        if field not in record:
            errors.append(f"Missing top-level field: {field}")
    
    # Prediction sub-object
    pred = record.get("prediction", {})
    if pred.get("outcome") not in ["Home Win", "Draw", "Away Win"]:
        errors.append(f"Invalid outcome: {pred.get('outcome')}")
    
    if pred.get("confidence") not in ["Low", "Medium", "High"]:
        errors.append(f"Invalid confidence: {pred.get('confidence')}")
    
    total = pred.get("home_win_pct", 0) + pred.get("draw_pct", 0) + pred.get("away_win_pct", 0)
    if total != 100:
        errors.append(f"Percentages sum to {total} not 100")
    
    return errors
```

---

## Python Helper: File I/O

```python
import json
from pathlib import Path
from datetime import date

PREDICTIONS_DIR = Path("data/predictions")
RESULTS_DIR = Path("data/results")
ACCURACY_FILE = Path("data/accuracy/accuracy-log.json")

def save_predictions(predictions: list, target_date: date = None):
    """Save list of predictions to daily file."""
    target_date = target_date or date.today()
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"{target_date.isoformat()}.json"
    path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
    print(f"Saved {len(predictions)} predictions to {path}")

def load_predictions(target_date: date = None) -> list:
    """Load predictions for a given date. Returns empty list if not found."""
    target_date = target_date or date.today()
    path = PREDICTIONS_DIR / f"{target_date.isoformat()}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())

def load_accuracy() -> dict:
    """Load accuracy log. Returns empty structure if not found."""
    if not ACCURACY_FILE.exists():
        return {}
    return json.loads(ACCURACY_FILE.read_text())
```

---

## Important Notes

- All JSON files use **UTF-8 encoding** (for team/player names with accents)
- Use `ensure_ascii=False` in `json.dumps()` to preserve special characters
- `result` field in predictions starts as `null` and gets filled in later
- Never delete prediction files — keep the full history for accuracy tracking
- The `cache/` folder is gitignored — never commit it
