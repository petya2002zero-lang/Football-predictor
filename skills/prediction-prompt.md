# Skill: Prediction Prompt

## What This Skill Is For
This skill defines exactly how to call Claude API to generate match predictions.
Read this before editing `scripts/predict.py` or changing any prompt logic.

---

## The Golden Rule
The prompt is the most important part of this project.
Better structured data in = better predictions out.
Never simplify the prompt to save tokens — quality matters more.

---

## Claude API Setup

```python
import anthropic
import json
import os

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1000  # Enough for full JSON response
```

---

## The Prediction Function

```python
def predict_match(match_data: dict) -> dict | None:
    """
    Send match data to Claude and get a structured prediction back.
    Returns prediction dict or None if Claude fails.
    """
    prompt = build_prompt(match_data)
    
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        raw_text = message.content[0].text.strip()
        
        # Clean up in case Claude adds backticks
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        
        prediction = json.loads(raw_text)
        validate_prediction(prediction)
        return prediction
        
    except json.JSONDecodeError as e:
        print(f"Claude returned invalid JSON: {e}")
        print(f"Raw response: {raw_text}")
        return None
    except Exception as e:
        print(f"Claude API error: {e}")
        return None
```

---

## Building the Prompt

```python
def build_prompt(m: dict) -> str:
    """
    Build the full prediction prompt from match data dict.
    m must contain all keys shown below.
    """
    return f"""You are an expert football analyst with deep statistical knowledge.
Analyze this match and provide a precise, data-driven prediction.

═══════════════════════════════════════
MATCH: {m['home_team']} vs {m['away_team']}
Competition: {m['league']}
Date: {m['date']}
Venue: {m.get('venue', 'Unknown')}
Referee: {m.get('referee', 'Unknown')}
═══════════════════════════════════════

LEAGUE STANDINGS:
{m['home_team']}: Position {m['home_position']}, {m['home_points']} pts, GD {m['home_gd']}
{m['away_team']}: Position {m['away_position']}, {m['away_points']} pts, GD {m['away_gd']}

RECENT FORM — last 5 games (newest first):
{m['home_team']}: {m['home_form']}
  → Avg goals scored: {m['home_scored_avg']} | Avg conceded: {m['home_conceded_avg']}
{m['away_team']}: {m['away_form']}
  → Avg goals scored: {m['away_scored_avg']} | Avg conceded: {m['away_conceded_avg']}

HOME / AWAY SPLITS this season:
{m['home_team']} at HOME: {m['home_home_w']}W {m['home_home_d']}D {m['home_home_l']}L
  → xG at home: {m.get('home_xg', 'N/A')} avg
{m['away_team']} AWAY: {m['away_away_w']}W {m['away_away_d']}D {m['away_away_l']}L
  → xG away: {m.get('away_xg', 'N/A')} avg

HEAD-TO-HEAD — last 5 meetings:
{m['h2h_summary']}
{m.get('h2h_detail', '')}

INJURIES & SUSPENSIONS:
{m['home_team']}: {m.get('home_injuries', 'None reported')}
{m['away_team']}: {m.get('away_injuries', 'None reported')}

═══════════════════════════════════════

Respond ONLY with a valid JSON object. No explanation before or after.
Use exactly this structure:

{{
  "outcome": "Home Win" or "Draw" or "Away Win",
  "confidence": "Low" or "Medium" or "High",
  "home_win_pct": <integer>,
  "draw_pct": <integer>,
  "away_win_pct": <integer>,
  "predicted_score": "<like 2-1>",
  "score_range": "<like 1-0 to 3-1>",
  "btts": <true or false>,
  "over_2_5": <true or false>,
  "key_factors": [
    "<specific factor with data>",
    "<specific factor with data>",
    "<specific factor with data>",
    "<specific factor with data>"
  ],
  "analysis": "<2-3 sentence summary of why this outcome is predicted>"
}}

Rules:
- home_win_pct + draw_pct + away_win_pct must equal exactly 100
- outcome must match whichever percentage is highest
- key_factors must reference actual numbers from the data above
- If data is missing or conflicting, set confidence to "Low"
- Do not invent statistics
"""
```

---

## Confidence Level Rules

Use these criteria when reviewing or auditing predictions:

| Confidence | Criteria |
|---|---|
| **High** | Strong form advantage + H2H support + no key injuries + clear home/away pattern |
| **Medium** | Some indicators align but others are mixed or missing |
| **Low** | Conflicting data, missing key stats, cup match with unknown lineups, or very even teams |

Cup matches (FA Cup, DFB-Pokal etc.) should almost always be **Low** or **Medium** — rotated squads are common.

---

## Retry Logic

If Claude fails, retry once after a short wait:

```python
import time

def predict_with_retry(match_data: dict, retries: int = 1) -> dict | None:
    """Try prediction, retry once on failure."""
    result = predict_match(match_data)
    if result is None and retries > 0:
        print("Retrying after 10 seconds...")
        time.sleep(10)
        result = predict_match(match_data)
    return result
```

---

## Validating Claude's Output

Always validate before saving:

```python
def validate_prediction(p: dict) -> None:
    """
    Raise ValueError if prediction is malformed.
    Call this before saving any prediction to disk.
    """
    required_fields = [
        "outcome", "confidence", "home_win_pct", "draw_pct",
        "away_win_pct", "predicted_score", "score_range",
        "btts", "over_2_5", "key_factors", "analysis"
    ]
    
    for field in required_fields:
        if field not in p:
            raise ValueError(f"Missing field: {field}")
    
    # Percentages must sum to 100
    total = p["home_win_pct"] + p["draw_pct"] + p["away_win_pct"]
    if total != 100:
        raise ValueError(f"Percentages sum to {total}, not 100")
    
    # Outcome must match highest percentage
    pcts = {
        "Home Win": p["home_win_pct"],
        "Draw": p["draw_pct"],
        "Away Win": p["away_win_pct"]
    }
    expected = max(pcts, key=pcts.get)
    if p["outcome"] != expected:
        raise ValueError(f"Outcome '{p['outcome']}' doesn't match highest pct ({expected})")
    
    # Key factors must be a non-empty list
    if not isinstance(p["key_factors"], list) or len(p["key_factors"]) == 0:
        raise ValueError("key_factors must be a non-empty list")
```

---

## Cost Awareness

- claude-sonnet-4-20250514 input: ~$3 per million tokens
- Average prompt = ~600 tokens input, ~250 tokens output
- 50 matches/day = ~42,500 tokens = ~$0.20/day
- Well within Claude Pro API limits
