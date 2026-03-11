# Skill: Leagues Reference

## What This Skill Is For
Complete reference for all covered leagues, their API-Football IDs,
seasonal patterns, and special handling rules. Read this when working
with fixture fetching, filtering, or displaying league-specific data.

---

## All Covered Leagues

```python
LEAGUES = {
    # ── Tier 1: Top 5 European Leagues ─────────────────────────────
    "Premier League":       {"id": 39,  "country": "England",     "tier": 1},
    "La Liga":              {"id": 140, "country": "Spain",       "tier": 1},
    "Bundesliga":           {"id": 78,  "country": "Germany",     "tier": 1},
    "Serie A":              {"id": 135, "country": "Italy",       "tier": 1},
    "Ligue 1":              {"id": 61,  "country": "France",      "tier": 1},

    # ── Tier 2: European Club Competitions ─────────────────────────
    "Champions League":     {"id": 2,   "country": "Europe",      "tier": 2},
    "Europa League":        {"id": 3,   "country": "Europe",      "tier": 2},
    "Conference League":    {"id": 848, "country": "Europe",      "tier": 2},

    # ── Tier 2: Other Top Leagues ───────────────────────────────────
    "Eredivisie":           {"id": 88,  "country": "Netherlands", "tier": 2},
    "Belgian Pro League":   {"id": 144, "country": "Belgium",     "tier": 2},
    "Primeira Liga":        {"id": 94,  "country": "Portugal",    "tier": 2},
    "Championship":         {"id": 40,  "country": "England",     "tier": 2},
    "Saudi Pro League":     {"id": 307, "country": "Saudi Arabia","tier": 2},
    "AFC Champions League": {"id": 17,  "country": "Asia",        "tier": 2},

    # ── National Cups ───────────────────────────────────────────────
    "FA Cup":               {"id": 45,  "country": "England",     "tier": 3},
    "Copa del Rey":         {"id": 143, "country": "Spain",       "tier": 3},
    "DFB-Pokal":            {"id": 81,  "country": "Germany",     "tier": 3},
    "Coppa Italia":         {"id": 137, "country": "Italy",       "tier": 3},
    "Coupe de France":      {"id": 66,  "country": "France",      "tier": 3},
}

# Quick lookup: ID → name
LEAGUE_ID_TO_NAME = {v["id"]: k for k, v in LEAGUES.items()}

# Quick lookup: just the IDs for API calls
ALL_LEAGUE_IDS = [v["id"] for v in LEAGUES.values()]
TIER1_IDS = [v["id"] for v in LEAGUES.values() if v["tier"] == 1]
TIER2_IDS = [v["id"] for v in LEAGUES.values() if v["tier"] == 2]
CUP_IDS   = [v["id"] for v in LEAGUES.values() if v["tier"] == 3]
```

---

## Frontend Tab Groups

How leagues are grouped in the filter tabs on the website:

```javascript
const LEAGUE_TABS = [
  { label: "All",      ids: null },  // null = show everything
  { label: "PL",       ids: [39]  },
  { label: "UCL",      ids: [2]   },
  { label: "La Liga",  ids: [140] },
  { label: "BL",       ids: [78]  },
  { label: "Serie A",  ids: [135] },
  { label: "Ligue 1",  ids: [61]  },
  { label: "EL/ECL",   ids: [3, 848] },
  { label: "Other",    ids: [88, 144, 94, 40, 307, 17] },
  { label: "Cups",     ids: [45, 143, 81, 137, 66] },
];
```

---

## Special Handling Rules

### Cup Matches (Tier 3)
- **Always set confidence to "Low" or "Medium"** — managers rotate squads
- Add this note to the prompt: `"This is a cup match. Teams often rotate. Lower confidence accordingly."`
- FA Cup especially: giant-killing is common, don't over-favour big teams

### Champions League / Europa League Group Stage vs Knockouts
- **Group stage** (Sept–Dec): treat like league games, good data available
- **Knockout rounds** (Feb–May): H2H is more important, form matters more
- Both legs matter — first leg result affects second leg prediction

### AFC Champions League
- Time zones differ significantly — double-check kick-off times
- Less H2H data available (fewer meetings between clubs)
- Lower confidence is appropriate for most matches

### Saudi Pro League
- Season runs Feb–Nov (opposite to European leagues)
- Less historical data than European leagues
- Marquee signings can distort team statistics quickly

### Championship (England Tier 2)
- Very high volume — 24 teams, lots of matches
- Form is very important; league position less so (bunched table)
- Stats quality is lower — be cautious with High confidence

---

## Typical Fixture Volume Per Week

| League | Avg matches/week |
|---|---|
| Premier League | 10 |
| La Liga | 10 |
| Bundesliga | 9 |
| Serie A | 10 |
| Ligue 1 | 10 |
| Champions League | 8–16 (matchday weeks only) |
| Europa League | 8–16 (matchday weeks only) |
| Conference League | 8–16 (matchday weeks only) |
| Eredivisie | 9 |
| Belgian Pro League | 8 |
| Primeira Liga | 9 |
| Championship | 12 |
| Saudi Pro League | 8 |
| All cups | 4–32 (round dependent) |

**Total on a busy week: ~150 matches**
On a quiet week (mid-week, no Europe): ~50 matches

With 300 API calls/day and ~4 calls per match:
- Quiet week: well within limits ✅
- Busy week: ~600 calls needed → use caching aggressively ⚠️

---

## Season Dates Reference (2025/2026)

| League | Season Start | Season End | Winter Break |
|---|---|---|---|
| Premier League | Aug 2025 | May 2026 | None (continues Dec/Jan) |
| La Liga | Aug 2025 | May 2026 | ~2 weeks Jan |
| Bundesliga | Aug 2025 | May 2026 | ~6 weeks Dec–Jan |
| Serie A | Aug 2025 | May 2026 | ~2 weeks Jan |
| Ligue 1 | Aug 2025 | May 2026 | ~2 weeks Dec/Jan |
| Eredivisie | Aug 2025 | May 2026 | ~4 weeks Dec–Jan |
| Belgian Pro League | Jul 2025 | May 2026 | ~3 weeks Dec/Jan |
| Championship | Aug 2025 | May 2026 | None |
| Saudi Pro League | Feb 2026 | Nov 2026 | Ramadan break Mar/Apr |
| Champions League | Sep 2025 | May 2026 | No group stage Jan |

**During winter breaks:** Fewer matches. The script handles this naturally
(if no fixtures found for a league, it just skips it).

---

## API-Football Season Parameter

Most endpoints need a `season` parameter (4-digit year of season start):

```python
CURRENT_SEASON = 2025  # For 2025/2026 season

# Exception: Saudi Pro League 2026 season
SAUDI_SEASON = 2025  # Started Feb 2026 but API uses 2025
```

Always double-check if a league returns 0 results — season year might need updating.
