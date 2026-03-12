"""
train_master.py — V3 Elite Quant Data Engine
=============================================
Key upgrades vs V2:
  • API quota budgeting: counts live requests, protects 7500/day limit
  • Stores league + home_id/away_id in recent_results → train_ml.py uses them
  • home/away cs_pct and fts_pct stored separately (train_ml.py needs both)
  • Top-5 leagues + UEFA processed first for priority data quality
  • get_team_stats makes 1 API call per team per week (not per run)
  • get_true_xg reuses fixture/statistics calls already made in get_team_stats
  • Smarter H2H: stores league context; skips re-fetch if cached today
  • Quota-aware get_team_stats: skips fixture/statistics sub-calls when near limit
"""

import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    import penaltyblog as pb
    PENALTYBLOG_AVAILABLE = True
except ImportError:
    PENALTYBLOG_AVAILABLE = False
    log.warning("penaltyblog not installed — Pi Ratings use fallback. pip install penaltyblog")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    log.error("API_KEY env var not set. Aborting.")
    sys.exit(1)

SEASON = 2025
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {"x-rapidapi-host": "v3.football.api-sports.io", "x-rapidapi-key": API_KEY}

# Quota guard: stop making uncached requests at this threshold
QUOTA_SOFT_LIMIT = 7000   # leave 500 buffer below the 7500 hard limit
_quota_used = 0           # live request counter for this run

# Leagues ordered: priority (top 5 + UEFA) first, then secondary
PRIORITY_LEAGUES = [39, 140, 78, 135, 61, 2, 3, 4, 848]   # PL, LaLiga, BL, SA, L1, UCL, UEL, UECL, EL
SECONDARY_LEAGUES = [
    88, 144, 40, 141, 79, 136, 62, 45, 143, 529, 137, 66,
    71, 94, 203, 253, 307, 17, 98,
]
LEAGUES           = PRIORITY_LEAGUES + SECONDARY_LEAGUES
STANDINGS_LEAGUES = PRIORITY_LEAGUES + [88, 144, 40, 141, 71, 94, 203, 253, 307, 17, 98]

# League name → rho for Dixon-Coles (used in pro_preds)
LEAGUE_RHO = {
    "Premier League": -0.134, "La Liga": -0.127, "Bundesliga": -0.119,
    "Serie A": -0.131, "Ligue 1": -0.122,
    "Champions League": -0.115, "UEFA Champions League": -0.115,
    "Europa League": -0.110, "UEFA Europa League": -0.110,
    "Conference League": -0.108, "UEFA Europa Conference League": -0.108,
}

STRICT_REFS = [
    "Mateu Lahoz", "Anthony Taylor", "Gil Manzano", "Hernandez Hernandez",
    "Clement Turpin", "Felix Zwayer", "Artur Soares Dias", "Facundo Tello",
    "Wilton Sampaio", "Istvan Kovacs", "Michael Oliver", "Daniele Orsato",
    "Szymon Marciniak",
]


# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------

def _db():
    return sqlite3.connect("database.sqlite", check_same_thread=False)

def init_db():
    import os
    db_path = "database.sqlite"
    # Validate existing file — delete if corrupted so SQLite recreates it cleanly
    if os.path.exists(db_path):
        try:
            test_conn = sqlite3.connect(db_path)
            test_conn.execute("PRAGMA integrity_check")
            test_conn.close()
        except sqlite3.DatabaseError:
            log.warning("database.sqlite is corrupted — deleting and recreating.")
            os.remove(db_path)
    conn = _db(); c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS api_cache (endpoint TEXT PRIMARY KEY, last_modified TEXT, response TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS bet_log (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, match TEXT, pick TEXT, odds REAL, stake REAL, result TEXT, profit REAL)")
    try:
        c.execute("DELETE FROM api_cache WHERE response LIKE '%\"errors\": %'")
    except Exception:
        pass
    conn.commit(); conn.close()

def save_kv(key, data):
    conn = _db(); c = conn.cursor()
    c.execute("REPLACE INTO kv_store (key,value) VALUES (?,?)", (key, json.dumps(data)))
    conn.commit(); conn.close()

def load_kv(key, default=None):
    try:
        conn = _db(); c = conn.cursor()
        c.execute("SELECT value FROM kv_store WHERE key=?", (key,))
        row = c.fetchone(); conn.close()
        if row: return json.loads(row[0])
    except Exception as e:
        log.warning("load_kv(%s): %s", key, e)
    return default if default is not None else {}

init_db()


# ---------------------------------------------------------------------------
# API FETCHER with 304 cache + quota counter
# ---------------------------------------------------------------------------

def fetch_api(endpoint, params=None, allow_uncached=True):
    """
    Fetch from API-Football with If-Modified-Since caching.
    If quota is near the soft limit, returns cached data only (no live request).
    """
    global _quota_used
    url = f"{BASE_URL}/{endpoint}"
    param_str = json.dumps(params, sort_keys=True) if params else ""
    cache_key = f"{endpoint}||{param_str}"

    conn = _db(); c = conn.cursor()
    c.execute("SELECT last_modified, response FROM api_cache WHERE endpoint=?", (cache_key,))
    row = c.fetchone()

    # Quota guard: serve from cache if near limit
    if _quota_used >= QUOTA_SOFT_LIMIT:
        conn.close()
        if row:
            log.debug("Quota soft limit — serving cache for %s", endpoint)
            return json.loads(row[1])
        return {}

    if not allow_uncached and not row:
        conn.close()
        return {}

    hdrs = HEADERS.copy()
    if row and row[0]:
        hdrs["If-Modified-Since"] = row[0]

    try:
        r = requests.get(url, headers=hdrs, params=params, timeout=15)
        _quota_used += 1  # count every live request

        if r.status_code == 304 and row:
            conn.close()
            return json.loads(row[1])
        elif r.status_code == 200:
            resp = r.json()
            if not resp.get("errors"):
                lm = r.headers.get("Last-Modified",
                     datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT"))
                c.execute("REPLACE INTO api_cache (endpoint,last_modified,response) VALUES (?,?,?)",
                          (cache_key, lm, json.dumps(resp)))
                conn.commit()
            conn.close()
            return resp
        else:
            log.warning("API %s → HTTP %d", endpoint, r.status_code)
    except Exception as e:
        log.warning("fetch_api(%s): %s", endpoint, e)

    conn.close()
    return json.loads(row[1]) if row else {}


# ---------------------------------------------------------------------------
# 1. RECENT RESULTS (last 90 days) + Pi Ratings
# ---------------------------------------------------------------------------
log.info("📜 Fetching recent results...")
recent_results = []
past_start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
today      = datetime.now().strftime("%Y-%m-%d")

for league_id in LEAGUES:
    try:
        resp = fetch_api("fixtures",
                         {"league": league_id, "season": SEASON,
                          "from": past_start, "to": today})
        for item in resp.get("response", []):
            if item["fixture"]["status"]["short"] not in ("FT","AET","PEN"):
                continue
            hg = item["goals"]["home"]; ag = item["goals"]["away"]
            if hg is None or ag is None: continue
            recent_results.append({
                "fixture_id": item["fixture"]["id"],          # for historical odds lookup
                "date":       item["fixture"]["date"],
                "home":       item["teams"]["home"]["name"],
                "away":       item["teams"]["away"]["name"],
                "home_id":    item["teams"]["home"]["id"],   # ← NEW: train_ml.py needs this
                "away_id":    item["teams"]["away"]["id"],   # ← NEW
                "league":     item["league"]["name"],         # ← NEW: for per-league rho
                "league_id":  item["league"]["id"],
                "home_goals": int(hg), "away_goals": int(ag),
                "score":      f"{hg}-{ag}",
                "referee":    item["fixture"].get("referee") or "Unknown",
            })
    except Exception as e:
        log.warning("Recent results league %d: %s", league_id, e)

save_kv("recent", recent_results)
log.info("Saved %d recent results. API requests so far: %d", len(recent_results), _quota_used)

# Pi Ratings — computed via Elo (no external dependency)
# Falls back gracefully to penaltyblog if available and working
pi_ratings = load_kv("pi_ratings", {})

def compute_elo_ratings(results, k=32, home_adv=60.0, initial=1000.0):
    """
    Compute Elo ratings from a list of match result dicts.
    Each dict needs: home, away, home_goals, away_goals.
    Returns dict of {team_name: elo_rating}.
    """
    ratings = {}
    def get_r(t): return ratings.get(t, initial)
    def expected(ra, rb): return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    for r in sorted(results, key=lambda x: x.get("date", "")):
        h, a = r["home"], r["away"]
        hg, ag = int(r.get("home_goals", 0)), int(r.get("away_goals", 0))
        rh, ra_ = get_r(h) + home_adv, get_r(a)
        eh = expected(rh, ra_)
        score_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        # Goal-difference multiplier (0.5 GD → 1.0x, 3+ GD → 1.75x)
        gd = abs(hg - ag)
        mult = 1.0 + min(gd * 0.25, 0.75)
        delta = k * mult * (score_h - eh)
        ratings[h] = get_r(h) + delta
        ratings[a] = get_r(a) - delta
    return ratings

def _find_pi_ratings_class():
    """
    Dynamically locate PiRatings in whatever penaltyblog version is installed.
    penaltyblog has moved the class across multiple releases:
      v0.x  → pb.models.PiRatings
      v1.x  → pb.ratings.PiRatings
      later → may be at top-level or under different submodule
    Returns the class (not an instance), or None if not found.
    """
    import importlib, pkgutil, inspect
    # 1. Known explicit paths — fastest path for common versions
    for attr_path in [
        "ratings.PiRatingSystem",
        "ratings.pi.PiRatingSystem",
        "ratings.PiRatings",
        "models.PiRatings",
        "PiRatings",
        "ratings.pi_ratings.PiRatings",
        "models.pi_ratings.PiRatings",
    ]:
        try:
            obj = pb
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if inspect.isclass(obj):
                log.info("penaltyblog: found %s at pb.%s", obj.__name__, attr_path)
                return obj
        except AttributeError:
            continue
    # 2. Deep scan — walk every submodule looking for any class named PiRatings
    try:
        for _, modname, _ in pkgutil.walk_packages(pb.__path__, prefix="penaltyblog."):
            try:
                mod = importlib.import_module(modname)
                for cls_name in ("PiRatingSystem", "PiRatings"):
                    cls = getattr(mod, cls_name, None)
                    if cls and inspect.isclass(cls):
                        log.info("penaltyblog: found %s via deep scan in %s", cls_name, modname)
                        return cls
            except Exception:
                continue
    except Exception:
        pass
    return None


def _try_fit_pi_ratings(cls, df):
    """
    Try all known fit() signatures across penaltyblog versions.
    Returns a ratings dict {team: float} or None on failure.
    """
    import inspect
    # Attempt to build an instance — try known constructor signatures
    instance = None
    for kwargs in [
        {"c": 0.065, "lambda_": 0.1},
        {"c": 0.065},
        {},
    ]:
        try:
            instance = cls(**kwargs)
            break
        except TypeError:
            continue
    if instance is None:
        log.warning("penaltyblog PiRatings: no constructor signature matched.")
        return None

    # Try all known fit() call patterns
    fit_attempts = [
        # v0/v1 positional: fit(home_goals, away_goals, home_team, away_team)
        lambda m: m.fit(df["home_goals"], df["away_goals"], df["home"], df["away"]),
        # DataFrame style: fit(df)  or  fit(data=df)
        lambda m: m.fit(df),
        # Named columns variant
        lambda m: m.fit(
            home_goals=df["home_goals"], away_goals=df["away_goals"],
            home_team=df["home"],        away_team=df["away"]
        ),
    ]
    for attempt in fit_attempts:
        try:
            attempt(instance)
            break
        except (TypeError, Exception):
            continue
    else:
        log.warning("penaltyblog PiRatings: no fit() signature matched.")
        return None

    # Try all known ways to extract ratings dict
    for getter in [
        lambda m: m.get_ratings(),
        lambda m: dict(m.ratings),
        lambda m: dict(m.ratings_),
        lambda m: {t: float(r) for t, r in zip(m.teams_, m.ratings_)},
        lambda m: m.to_dict(),
    ]:
        try:
            result = getter(instance)
            if isinstance(result, dict) and result:
                return result
        except Exception:
            continue
    log.warning("penaltyblog PiRatings: could not extract ratings dict.")
    return None


_elo_computed = False
if recent_results:
    if PENALTYBLOG_AVAILABLE:
        try:
            df_pi = pd.DataFrame(recent_results)
            df_pi["date"] = pd.to_datetime(df_pi["date"])
            df_pi = df_pi.dropna(subset=["home_goals", "away_goals"])
            df_pi["home_goals"] = df_pi["home_goals"].astype(int)
            df_pi["away_goals"] = df_pi["away_goals"].astype(int)

            pi_cls = _find_pi_ratings_class()
            if pi_cls is not None:
                ratings_dict = _try_fit_pi_ratings(pi_cls, df_pi)
                if ratings_dict:
                    for team, rating in ratings_dict.items():
                        pi_ratings[team] = float(rating)
                    log.info("✅ Pi Ratings (penaltyblog) computed for %d teams.", len(ratings_dict))
                    _elo_computed = True
                else:
                    log.warning("penaltyblog PiRatings fit/extract failed — using Elo fallback.")
            else:
                # Log what IS available to help diagnose future version changes
                import pkgutil, importlib
                available = []
                try:
                    for _, modname, _ in pkgutil.walk_packages(pb.__path__, prefix="penaltyblog."):
                        try:
                            mod = importlib.import_module(modname)
                            cls_names = [x for x in dir(mod)
                                         if not x.startswith("_") and isinstance(getattr(mod, x, None), type)]
                            if cls_names:
                                available.append(f"{modname}: {cls_names}")
                        except Exception:
                            pass
                except Exception:
                    pass
                log.warning("penaltyblog PiRatings class not found. Available classes: %s — using Elo fallback.",
                             available[:5] if available else "none discovered")
        except Exception as e:
            log.warning("penaltyblog Pi Ratings error (%s) — using Elo fallback.", e)
    else:
        log.info("penaltyblog not installed — using Elo ratings.")

if not _elo_computed and recent_results:
    elo = compute_elo_ratings(recent_results)
    # Normalise to roughly the same scale as Pi Ratings (centre at 0)
    mean_elo = sum(elo.values()) / max(1, len(elo))
    for team, rating in elo.items():
        pi_ratings[team] = round((rating - mean_elo) / 400.0, 4)
    log.info("✅ Elo ratings computed for %d teams.", len(elo))


# ---------------------------------------------------------------------------
# 2. UPCOMING FIXTURES (next 10 days)
# ---------------------------------------------------------------------------
log.info("📅 Fetching upcoming fixtures...")
upcoming_matches = []; seen_matches = set(); team_future_fixtures = {}
window_end = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")

for league_id in LEAGUES:
    try:
        resp = fetch_api("fixtures",
                         {"league": league_id, "season": SEASON,
                          "from": today, "to": window_end})
        for item in resp.get("response", []):
            mid  = item["fixture"]["id"]
            h_id = item["teams"]["home"]["id"]
            a_id = item["teams"]["away"]["id"]
            m_dt = item["fixture"]["date"]
            team_future_fixtures.setdefault(h_id, []).append(m_dt)
            team_future_fixtures.setdefault(a_id, []).append(m_dt)
            if mid in seen_matches: continue
            seen_matches.add(mid)
            upcoming_matches.append({
                "id":          mid,
                "date":        m_dt,
                "home":        item["teams"]["home"]["name"],
                "away":        item["teams"]["away"]["name"],
                "league":      item["league"]["name"],
                "league_id":   item["league"]["id"],
                "home_id":     h_id,
                "away_id":     a_id,
                "home_logo":   item["teams"]["home"]["logo"],
                "away_logo":   item["teams"]["away"]["logo"],
                "league_logo": item["league"]["logo"],
                "referee":     item["fixture"].get("referee") or "Unknown",
            })
    except Exception as e:
        log.warning("Upcoming league %d: %s", league_id, e)

upcoming_matches.sort(key=lambda x: x["date"])
save_kv("upcoming", upcoming_matches)
log.info("Saved %d upcoming matches. API requests: %d", len(upcoming_matches), _quota_used)


# ---------------------------------------------------------------------------
# 3. STANDINGS, FORMS, KEY PLAYERS, LEAGUE AVERAGES
# ---------------------------------------------------------------------------
log.info("📊 Fetching standings...")
standings_data = {}; team_forms = {}; key_players = {}; league_averages = {}

def add_key_player(tid, name, role):
    key_players.setdefault(tid, [])
    if not any(p["name"]==name for p in key_players[tid]):
        key_players[tid].append({"name": name, "role": role})

for league_id in STANDINGS_LEAGUES:
    if _quota_used >= QUOTA_SOFT_LIMIT:
        break
    try:
        r = fetch_api("standings", {"league": league_id, "season": SEASON})
        if not r.get("response"): continue
        ln = r["response"][0]["league"]
        standings_data[ln["name"]] = ln
        h_scored = h_played = a_scored = a_played = 0
        if ln.get("standings"):
            table = ln["standings"][0]
            max_pts  = table[0]["points"]  if table else 0
            rel_pts  = table[-3]["points"] if len(table) > 3 else 0
            for row in table:
                tid   = row["team"]["id"]
                tname = row["team"]["name"]
                team_forms[tid] = str(row.get("form") or "?????")[-5:]
                h_scored += row["home"]["goals"]["for"]  or 0
                h_played += row["home"]["played"]        or 0
                a_scored += row["away"]["goals"]["for"]  or 0
                a_played += row["away"]["played"]        or 0
                pts  = row["points"] or 0
                xpts = (row["all"]["win"]*3 + row["all"]["draw"]
                        + (row.get("goalsDiff", 0)*0.15))
                pi_ratings[f"{tname}_safety"]     = float(pts - rel_pts)
                pi_ratings[f"{tname}_title"]       = float(pts - max_pts)
                pi_ratings[f"{tname}_xpts_delta"]  = float(pts - xpts)
            # Track result/market rates alongside goal averages
            h_wins = a_wins = draws = total_played = 0
            o25 = btts_count = 0
            for row2 in table:
                hw = row2["home"]["win"]  or 0
                aw = row2["away"]["win"]  or 0
                hd = row2["home"]["draw"] or 0
                tp = row2["all"]["played"] or 0
                hgf_h = row2["home"]["goals"]["for"]     or 0
                hga_h = row2["home"]["goals"]["against"] or 0
                agf_a = row2["away"]["goals"]["for"]     or 0
                aga_a = row2["away"]["goals"]["against"] or 0
                h_wins += hw; a_wins += aw
                draws  += hd
                total_played += tp
                # approximate o25 / btts from season totals
                total_goals_h = hgf_h + hga_h
                total_goals_a = agf_a + aga_a
                games_h = max(1, row2["home"]["played"] or 1)
                games_a = max(1, row2["away"]["played"] or 1)
                o25 += (total_goals_h / games_h) + (total_goals_a / games_a)
            n_teams = max(1, len(table))
            # Each actual match appears once in h_wins/a_wins/draws but twice in
            # total_played (counted for both home and away team). Use the sum of
            # outcomes as the correct denominator so rates sum to 1.0.
            n_matches = max(1, h_wins + a_wins + draws)
            league_averages[league_id] = {
                "h_avg":       (h_scored/h_played)   if h_played else 1.45,
                "a_avg":       (a_scored/a_played)    if a_played else 1.20,
                "home_win_rate": h_wins / n_matches,
                "away_win_rate": a_wins / n_matches,
                "draw_rate":     draws  / n_matches,
                "avg_goals_pg":  (h_scored + a_scored) / n_matches,
            }
        for ep, role in [("topscorers","Scorer"), ("topassists","Playmaker")]:
            if _quota_used >= QUOTA_SOFT_LIMIT:
                break
            try:
                rd = fetch_api(f"players/{ep}", {"league": league_id, "season": SEASON})
                for p in rd.get("response",[])[:8]:
                    add_key_player(p["statistics"][0]["team"]["id"], p["player"]["name"], role)
            except Exception: pass
        time.sleep(0.4)
    except Exception as e:
        log.warning("Standings league %d: %s", league_id, e)

save_kv("standings", standings_data)
save_kv("team_forms", team_forms)
save_kv("top_scorers", key_players)
save_kv("league_averages", league_averages)
save_kv("pi_ratings", pi_ratings)
log.info("Standings done. API requests: %d", _quota_used)


# ---------------------------------------------------------------------------
# 4. TEAM STATS (weekly-cached; separate home & away cs/fts stored)
# ---------------------------------------------------------------------------
player_cache     = load_kv("player_cache", {})
coach_cache      = load_kv("coach_cache", {})
team_stats_cache = load_kv("team_stats_cache", {})
events_cache     = load_kv("events_cache", {})
transfer_cache   = load_kv("transfer_cache", {})
sidelined_cache  = load_kv("sidelined_cache", {})

def get_team_stats(team_id, league_id, fixture_date=None):
    """
    Returns rich stats dict.  Cached weekly per team+league.
    Key change: stores cs_h, cs_a, fts_h, fts_a separately
    so train_ml.py can use them as independent features.

    fixture_date: optional timezone-aware datetime of the upcoming match.
    When provided, the weekly cache is invalidated if the team's last_date
    is within 3 days of the fixture (recent match may have updated their stats).
    """
    cycle     = int(datetime.now().timestamp() // 604800)
    cache_key = f"{team_id}_{league_id}_w{cycle}_v3"
    if cache_key in team_stats_cache:
        cached = team_stats_cache[cache_key]
        # Smart invalidation: bypass stale cache when team played very recently
        if fixture_date is not None and cached.get("last_date"):
            try:
                last_dt = datetime.fromisoformat(cached["last_date"])
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                fix_dt = fixture_date if fixture_date.tzinfo else fixture_date.replace(tzinfo=timezone.utc)
                days_gap = (fix_dt - last_dt).days
                if 0 <= days_gap <= 3:
                    # Team played within 3 days of this fixture — refresh stats
                    log.debug("Cache invalidated for team %d: last_match %d days before fixture", team_id, days_gap)
                    del team_stats_cache[cache_key]
                else:
                    return cached
            except (ValueError, TypeError):
                return cached
        else:
            return cached

    if _quota_used >= QUOTA_SOFT_LIMIT:
        return None
    try:
        r = fetch_api("teams/statistics",
                      {"league": league_id, "season": SEASON, "team": team_id})
        if not r or not r.get("response"): return None
        s = r["response"]

        # Recent fixtures — skip sub-calls if quota is tight
        rec_s = rec_c = count = 0
        last_date = "2020-01-01T00:00:00+00:00"
        dominance_score = 0
        false_form = False
        xg_total = 0.0; xg_count = 0

        if _quota_used < QUOTA_SOFT_LIMIT - 200:
            r2 = fetch_api("fixtures", {"team": team_id, "last": 5})
            if r2 and r2.get("response"):
                fixtures_sorted = sorted(r2["response"], key=lambda x: x["fixture"]["date"])
                last_date = fixtures_sorted[-1]["fixture"]["date"]
                for m in fixtures_sorted:
                    home = (m["teams"]["home"]["id"] == team_id)
                    gf = (m["goals"]["home"] if home else m["goals"]["away"]) or 0
                    ga = (m["goals"]["away"] if home else m["goals"]["home"]) or 0
                    rec_s += gf; rec_c += ga; count += 1

                # Only check last 3 for false_form + dominance + true xG
                for m in fixtures_sorted[-3:]:
                    fid  = m["fixture"]["id"]
                    home = (m["teams"]["home"]["id"] == team_id)
                    gf   = (m["goals"]["home"] if home else m["goals"]["away"]) or 0
                    ga   = (m["goals"]["away"] if home else m["goals"]["home"]) or 0

                    if gf <= ga:
                        if str(fid) not in events_cache:
                            rv = fetch_api("fixtures/events", {"fixture": fid, "type": "Card"})
                            events_cache[str(fid)] = rv.get("response", [])
                            time.sleep(0.1)
                        for ev in events_cache.get(str(fid), []):
                            if (ev["team"]["id"]==team_id
                                    and ev.get("detail")=="Red Card"):
                                try:
                                    if int(ev["time"]["elapsed"]) <= 45:
                                        false_form = True
                                except Exception: pass

                    if _quota_used < QUOTA_SOFT_LIMIT - 100:
                        rs = fetch_api("fixtures/statistics",
                                       {"fixture": fid, "team": team_id})
                        if rs.get("response"):
                            sd = {i["type"]: i["value"]
                                  for i in rs["response"][0]["statistics"]}
                            poss = int(str(sd.get("Ball Possession","50%")).replace("%","") or 50)
                            sot  = int(sd.get("Shots on Goal", 0) or 0)
                            if poss >= 55 and sot >= 5:  dominance_score += 1
                            elif poss <= 45 and sot <= 2: dominance_score -= 1
                            raw_xg = sd.get("expected_goals")
                            if raw_xg:
                                try: xg_total += float(raw_xg); xg_count += 1
                                except Exception: pass
                        time.sleep(0.12)

        def sf(v):
            try: return float(v) if v is not None else 0.0
            except: return 0.0

        played_h     = max(1, s.get("fixtures",{}).get("played",{}).get("home")     or 1)
        played_a     = max(1, s.get("fixtures",{}).get("played",{}).get("away")     or 1)
        played_total = max(1, s.get("fixtures",{}).get("played",{}).get("total")    or 1)

        # Separate home and away cs/fts — both needed by train_ml.py
        cs_h  = (s.get("clean_sheet",{}).get("home",  0) or 0) / played_h
        cs_a  = (s.get("clean_sheet",{}).get("away",  0) or 0) / played_a
        fts_h = (s.get("failed_to_score",{}).get("home", 0) or 0) / played_h
        fts_a = (s.get("failed_to_score",{}).get("away", 0) or 0) / played_a

        def get_pct(md, keys):
            total = 0.0
            if not isinstance(md, dict): return total
            for k in keys:
                v = md.get(k)
                if isinstance(v, dict) and v.get("percentage") is not None:
                    try: total += float(str(v["percentage"]).replace("%",""))
                    except: pass
            return total

        def parse_cards(ct):
            total = 0
            try:
                for v in ((s.get("cards",{}) or {}).get(ct,{}) or {}).values():
                    if isinstance(v,dict) and v.get("total"): total += int(v["total"])
            except: pass
            return total

        stats = {
            "s_h":       (sf(s.get("goals",{}).get("for",{}).get("average",{}).get("home"))
                           + (rec_s/count if count else 0)) / 2,
            "s_a":       (sf(s.get("goals",{}).get("for",{}).get("average",{}).get("away"))
                           + (rec_s/count if count else 0)) / 2,
            "c_h":       (sf(s.get("goals",{}).get("against",{}).get("average",{}).get("home"))
                           + (rec_c/count if count else 0)) / 2,
            "c_a":       (sf(s.get("goals",{}).get("against",{}).get("average",{}).get("away"))
                           + (rec_c/count if count else 0)) / 2,
            "last_date": last_date,
            "dominance": dominance_score,
            "home_win_pct": (s.get("fixtures",{}).get("wins",{}).get("home",0) or 0) / played_h,
            "away_win_pct": (s.get("fixtures",{}).get("wins",{}).get("away",0) or 0) / played_a,
            "fast_start": get_pct(s.get("goals",{}).get("for",{}).get("minute",{}),
                                  ["0-15","16-30"]),
            "late_leak":  get_pct(s.get("goals",{}).get("against",{}).get("minute",{}),
                                  ["76-90","91-105","106-120"]),
            # Both home and away cs/fts stored separately
            "cs_h":  cs_h,   "cs_a":  cs_a,
            "fts_h": fts_h,  "fts_a": fts_a,
            "cs_pct":  cs_h,   # legacy alias used by dashboard badges
            "fts_pct": fts_a,  # legacy alias
            "false_form": false_form,
            "yellow_avg": parse_cards("yellow") / played_total,
            "red_avg":    parse_cards("red")    / played_total,
            "true_xg":       (xg_total/xg_count) if xg_count > 0 else None,
            "true_xg_count": xg_count,
        }
        team_stats_cache[cache_key] = stats
        time.sleep(0.1)
        return stats

    except Exception as e:
        log.warning("get_team_stats(%d): %s", team_id, e)
        return None


def get_last_match_date(team_id):
    if _quota_used >= QUOTA_SOFT_LIMIT:
        return datetime.now(timezone.utc) - timedelta(days=7)
    try:
        r = fetch_api("fixtures", {"team": team_id, "last": 1})
        if r.get("response"):
            raw = r["response"][0]["fixture"]["date"].replace("Z","+00:00")
            return datetime.fromisoformat(raw)
    except Exception: pass
    return datetime.now(timezone.utc) - timedelta(days=7)


def is_new_manager(team_id):
    tid = str(team_id)
    if tid not in coach_cache:
        if _quota_used >= QUOTA_SOFT_LIMIT:
            coach_cache[tid] = "2000-01-01"
        else:
            try:
                rc = fetch_api("coachs", {"team": team_id})
                join_date = "2000-01-01"
                for c in rc.get("response",[]):
                    for car in c.get("career",[]):
                        if car["team"]["id"]==team_id and car["end"] is None:
                            join_date = car.get("start") or "2000-01-01"; break
                coach_cache[tid] = join_date; time.sleep(0.1)
            except Exception:
                coach_cache[tid] = "2000-01-01"
    try:
        return (datetime.now() - datetime.strptime(coach_cache[tid], "%Y-%m-%d")).days <= 21
    except: return False


def has_exodus(team_id):
    tid = str(team_id)
    if tid not in transfer_cache:
        if _quota_used >= QUOTA_SOFT_LIMIT:
            transfer_cache[tid] = False
        else:
            try:
                rt = fetch_api("transfers", {"team": team_id})
                exodus = False
                for t_obj in rt.get("response",[]):
                    for tr in t_obj.get("transfers",[]):
                        if tr.get("type") == "Out":
                            try:
                                if (datetime.now() - datetime.strptime(tr["date"],"%Y-%m-%d")).days <= 30:
                                    exodus = True; break
                            except: pass
                transfer_cache[tid] = exodus; time.sleep(0.1)
            except Exception:
                transfer_cache[tid] = False
    return bool(transfer_cache.get(tid, False))


# ---------------------------------------------------------------------------
# 5. MATCH INSIGHTS (H2H, Injuries, Suspensions, Odds)
# ---------------------------------------------------------------------------
log.info("⚔️  Fetching match insights (%d matches)...", min(len(upcoming_matches), 150))
h2h_data = {}; match_insights = {}

for i, match in enumerate(upcoming_matches[:150]):
    if _quota_used >= QUOTA_SOFT_LIMIT:
        log.warning("Quota soft limit reached at match %d/%d — stopping insight fetch.",
                    i, min(len(upcoming_matches), 150))
        break

    m_id    = match["id"]
    h2h_key = f"{match['home_id']}-{match['away_id']}"
    if i % 10 == 0:
        log.info("  Insights %d/%d  API requests: %d",
                 i, min(len(upcoming_matches), 150), _quota_used)

    # H2H (cache for 24h by storing with date in key)
    today_str = datetime.now().strftime("%Y-%m-%d")
    h2h_cache_key = f"h2h_{h2h_key}_{today_str}"
    if h2h_cache_key not in h2h_data:
        try:
            rh = fetch_api("fixtures/headtohead", {"h2h": h2h_key, "last": 5})
            if rh.get("response"):
                h2h_list = [
                    {
                        "date":       x["fixture"]["date"][:10],
                        "score":      f"{x['goals']['home']}-{x['goals']['away']}",
                        "winner":     ("Home" if x["teams"]["home"]["winner"]
                                       else "Away" if x["teams"]["away"]["winner"]
                                       else "Draw"),
                        "home_logo":  x["teams"]["home"]["logo"],
                        "away_logo":  x["teams"]["away"]["logo"],
                    }
                    for x in rh["response"]
                ]
                h2h_hw = sum(1 for x in h2h_list if x["winner"]=="Home")
                h2h_dw = sum(1 for x in h2h_list if x["winner"]=="Draw")
                h2h_aw = sum(1 for x in h2h_list if x["winner"]=="Away")
                h2h_n  = max(1, len(h2h_list))
                h2h_data[h2h_key] = h2h_list
                h2h_data[h2h_cache_key] = True   # mark as fetched today
                h2h_data[f"{h2h_key}_stats"] = {
                    "n": h2h_n,
                    "h_win_rate": round(h2h_hw / h2h_n, 3),
                    "draw_rate":  round(h2h_dw / h2h_n, 3),
                    "a_win_rate": round(h2h_aw / h2h_n, 3),
                }
        except Exception as e:
            log.debug("H2H %d: %s", m_id, e)

    insight = {"injuries": [], "odds": {}, "api_pred": {}, "sharp_action": None, "suspensions": []}

    # Injuries
    try:
        ri = fetch_api("injuries", {"fixture": m_id})
        for item in ri.get("response", []):
            p_id   = item["player"].get("id")
            p_name = item["player"]["name"]
            t_id   = item["team"]["id"]
            t_name = item["team"]["name"]
            reason = item["player"].get("reason","")
            impact_pct = 0.0; stats_str = ""; is_key = False; is_reg = False

            if p_id and _quota_used < QUOTA_SOFT_LIMIT - 50:
                pck = f"p_{p_id}_{SEASON}"
                if pck not in player_cache:
                    rp = fetch_api("players", {"id": p_id, "season": SEASON})
                    player_cache[pck] = rp.get("response", []); time.sleep(0.1)
                pd_data = player_cache.get(pck, [])
                if pd_data and pd_data[0].get("statistics"):
                    sn = pd_data[0]["statistics"][0]
                    minutes  = int(sn.get("games",{}).get("minutes")    or 0)
                    g_total  = int(sn.get("goals",{}).get("total")      or 0)
                    a_total  = int(sn.get("goals",{}).get("assists")    or 0)
                    k_pass   = int(sn.get("passes",{}).get("key")       or 0)
                    intercep = int(sn.get("tackles",{}).get("interceptions") or 0)
                    rating   = float(sn.get("games",{}).get("rating")   or 0.0)
                    # --- Impact: player's share of team goals ---
                    # Use goals+assists relative to minutes played as rate metric
                    # A player needs real volume to qualify for high-impact tiers
                    apps        = max(1, int(sn.get("games",{}).get("appearences") or 0))
                    g_per_game  = g_total / apps
                    a_per_game  = a_total / apps
                    ga_per_game = g_per_game + a_per_game * 0.7
                    ga_total    = g_total + a_total * 0.7

                    # Tier thresholds — require BOTH volume AND rate
                    # Superstar: 15+ G+A AND averaging 0.6+ per game AND 7.5+ rating
                    if ga_total >= 15 and ga_per_game >= 0.60 and rating >= 7.5:
                        impact_pct = round(min(0.28, ga_total / 60.0), 2)
                        is_key = True
                        stats_str = (f"Superstar: {g_total}G {a_total}A in {apps} apps "
                                     f"(Impact: -{int(impact_pct*100)}% xG)")
                    # Key Attacker: 8+ G+A AND 0.4+ per game
                    elif ga_total >= 8 and ga_per_game >= 0.40:
                        impact_pct = round(min(0.18, ga_total / 60.0), 2)
                        is_key = True
                        stats_str = (f"Key Attacker: {g_total}G {a_total}A in {apps} apps "
                                     f"(Impact: -{int(impact_pct*100)}% xG)")
                    # Contributing Attacker: 4+ G+A
                    elif ga_total >= 4:
                        impact_pct = round(min(0.10, ga_total / 80.0), 2)
                        is_key = True
                        stats_str = (f"Contributor: {g_total}G {a_total}A in {apps} apps "
                                     f"(Impact: -{int(impact_pct*100)}% xG)")
                    # Defensive Anchor: 1800+ mins AND strong interceptions AND rating
                    elif minutes >= 1800 and intercep >= 25 and rating >= 7.0:
                        impact_pct = 0.12; is_key = True
                        stats_str = f"Defensive Anchor: {intercep} INT, {minutes} mins (Impact: -12% xG)"
                    # Key Passer: heavy creative output
                    elif k_pass >= 40 and minutes >= 1200:
                        impact_pct = 0.10; is_key = True
                        stats_str = f"Key Passer: {k_pass} KP, {minutes} mins (Impact: -10% xG)"
                    # Regular Starter: meaningful minutes, no standout stats
                    elif minutes >= 1200:
                        impact_pct = 0.05; is_reg = True
                        stats_str = f"Regular Starter: {minutes} mins (Impact: -5% xG)"
                    elif minutes >= 400:
                        impact_pct = 0.02
                        stats_str = f"Rotational: {minutes} mins (Impact: -2% xG)"
                    else:
                        stats_str = f"Benchwarmer: {minutes} mins (No xG impact)"
            insight["injuries"].append({
                "player_name": p_name, "team_name": t_name, "team_id": t_id,
                "reason": reason, "impact_pct": impact_pct,
                "stats_str": stats_str, "is_key": is_key, "is_regular": is_reg,
            })
    except Exception as e:
        log.warning("Injuries match %d: %s", m_id, e)

    # Suspensions via sidelined
    try:
        for t_id in (match["home_id"], match["away_id"]):
            if str(t_id) not in sidelined_cache:
                rs = fetch_api("sidelined", {"team": t_id})
                sidelined_cache[str(t_id)] = rs.get("response", []); time.sleep(0.1)
            for s in sidelined_cache.get(str(t_id), []):
                if s.get("type") == "Suspended":
                    insight["suspensions"].append({
                        "player_name": s["player"]["name"], "team_id": t_id, "reason": "Suspended",
                    })
    except Exception as e:
        log.warning("Suspensions match %d: %s", m_id, e)

    # API predictions
    try:
        rp = fetch_api("predictions", {"fixture": m_id})
        if rp.get("response"):
            insight["api_pred"] = rp["response"][0]["predictions"]
    except Exception: pass

    # Odds + sharp money
    try:
        ro = fetch_api("odds", {"fixture": m_id})
        if ro.get("response") and ro["response"][0].get("bookmakers"):
            bookies = ro["response"][0]["bookmakers"]
            pin   = next((b for b in bookies if b["id"]==4), None)
            b365  = next((b for b in bookies if b["id"]==8), bookies[0])
            def ext_odds(bn):
                if not bn: return {}
                for bet in bn.get("bets",[]):
                    if bet["name"]=="Match Winner":
                        return {v["value"]: float(v["odd"]) for v in bet["values"]}
                return {}
            p_odds = ext_odds(pin); b_odds = ext_odds(b365)
            insight["odds"]  = b_odds                # B365 for display/value calc
            insight["p_odds"] = p_odds               # Pinnacle for sharp implied probs
            if p_odds and b_odds:
                if b_odds.get("Home",0) - p_odds.get("Home",0) >= 0.07:
                    insight["sharp_action"] = f"{match['home']} (Pinnacle Drop)"
                if b_odds.get("Away",0) - p_odds.get("Away",0) >= 0.07:
                    insight["sharp_action"] = f"{match['away']} (Pinnacle Drop)"
    except Exception: pass

    match_insights[m_id] = insight
    time.sleep(0.1)

save_kv("h2h", h2h_data)
save_kv("sidelined_cache", sidelined_cache)
save_kv("insights", match_insights)
log.info("Insights done. API requests: %d", _quota_used)


# ---------------------------------------------------------------------------
# 5.5. HISTORICAL ODDS — enrich recent results with Pinnacle probs for training
# ---------------------------------------------------------------------------
# The market features (pin_implied_h/a, market_edge) are the sharpest predictive
# signals but were previously all-zero in training data because pro_preds only
# covers upcoming matches. This section fetches and caches odds for completed
# fixtures so train_ml.py can use real market signals in every training row.
# Budget: HIST_ODDS_PER_RUN per run; across runs the full recent window fills in.
HIST_ODDS_PER_RUN = 120

hist_odds_cache = load_kv("hist_odds_cache", {})

def _extract_odds(bookmakers):
    """Return (pin_implied_h, pin_implied_a) from a bookmakers list."""
    def _to_prob(o):
        try: return round(1.0 / float(o), 4) if float(o) > 1 else 0.0
        except: return 0.0
    def _match_winner(bookie):
        for bet in bookie.get("bets", []):
            if bet["name"] == "Match Winner":
                return {v["value"]: float(v["odd"]) for v in bet["values"]}
        return {}
    pin  = next((b for b in bookmakers if b["id"] == 4), None)
    b365 = next((b for b in bookmakers if b["id"] == 8),
                bookmakers[0] if bookmakers else None)
    p = _match_winner(pin)  if pin  else {}
    b = _match_winner(b365) if b365 else {}
    src = p or b  # prefer Pinnacle
    return _to_prob(src.get("Home", 0)), _to_prob(src.get("Away", 0))

# Sort newest-first so quota fills the most time-decay-weighted records first
unfetched = [
    r for r in sorted(recent_results, key=lambda x: x["date"], reverse=True)
    if str(r.get("fixture_id", "")) not in hist_odds_cache
][:HIST_ODDS_PER_RUN]

log.info("Historical odds: %d cached, fetching up to %d more...",
         len(hist_odds_cache), len(unfetched))

for r in unfetched:
    if _quota_used >= QUOTA_SOFT_LIMIT - 100:
        log.warning("Quota near limit — stopping historical odds fetch.")
        break
    fid = str(r["fixture_id"])
    try:
        ro = fetch_api("odds", {"fixture": r["fixture_id"]})
        pin_h = pin_a = 0.0
        if ro.get("response") and ro["response"][0].get("bookmakers"):
            pin_h, pin_a = _extract_odds(ro["response"][0]["bookmakers"])
        hist_odds_cache[fid] = {"pin_implied_h": pin_h, "pin_implied_a": pin_a}
    except Exception as e:
        log.debug("Hist odds fixture %s: %s", fid, e)
        hist_odds_cache[fid] = {"pin_implied_h": 0.0, "pin_implied_a": 0.0}
    time.sleep(0.05)

save_kv("hist_odds_cache", hist_odds_cache)
log.info("Historical odds cache: %d total entries. API requests: %d",
         len(hist_odds_cache), _quota_used)


# ---------------------------------------------------------------------------
# 6. ADVANCED QUANT ENGINE — compile pro_predictions per match
# ---------------------------------------------------------------------------
log.info("💎 Compiling V3 pro predictions...")
pro_predictions = {}

for match in upcoming_matches[:150]:
    match_key = f"{match['home']} vs {match['away']}"
    try:
        fix_dt = datetime.fromisoformat(match["date"].replace("Z", "+00:00"))
    except Exception:
        fix_dt = None
    h_stats = get_team_stats(match["home_id"], match["league_id"], fixture_date=fix_dt) or {}
    a_stats = get_team_stats(match["away_id"], match["league_id"], fixture_date=fix_dt) or {}

    h_pi = float(pi_ratings.get(match["home"], 0.0))
    a_pi = float(pi_ratings.get(match["away"],  0.0))
    l_avg = league_averages.get(match["league_id"], {"h_avg": 1.45, "a_avg": 1.20})

    # Venue-specific xG — use home team's HOME scoring avg and away team's AWAY scoring avg
    # Falls back to season-wide average when venue-split is unavailable
    h_scored_home = h_stats.get("s_h")   # home team's goals-per-game AT HOME
    a_scored_away = a_stats.get("s_a")   # away team's goals-per-game AWAY
    h_conceded_home = h_stats.get("c_h") # home team goals conceded AT HOME
    a_conceded_away = a_stats.get("c_a") # away team goals conceded AWAY

    if h_scored_home is not None and a_conceded_away is not None and h_scored_home > 0:
        # Geometric blend: team's venue attack × opponent's venue defence ÷ league avg
        l_h_avg = l_avg.get("h_avg", 1.45)
        h_xg = max(0.2, min(5.0, (h_scored_home * a_conceded_away) / max(0.5, l_h_avg)))
    else:
        pi_adj_h = max(-1.5, min(1.5, (h_pi - a_pi) / 2.0))
        h_xg = max(0.2, l_avg["h_avg"] + pi_adj_h)

    if a_scored_away is not None and h_conceded_home is not None and a_scored_away > 0:
        l_a_avg = l_avg.get("a_avg", 1.20)
        a_xg = max(0.2, min(5.0, (a_scored_away * h_conceded_home) / max(0.5, l_a_avg)))
    else:
        pi_adj_a = max(-1.5, min(1.5, (h_pi - a_pi) / 2.0))
        a_xg = max(0.2, l_avg["a_avg"] - pi_adj_a)

    # Pi rating differential still applied as tiebreaker
    pi_boost = (h_pi - a_pi) * 0.15
    h_xg = max(0.2, h_xg + pi_boost)
    a_xg = max(0.2, a_xg - pi_boost)

    # Form & dominance adjustment — exponential decay (most recent = highest weight)
    # Weights for last-5 results oldest→newest: [0.5, 0.7, 1.0, 1.4, 2.0]  (sum≈5.6)
    _FORM_WEIGHTS = [0.5, 0.7, 1.0, 1.4, 2.0]
    def _weighted_form(form_str):
        pts = [3 if c=="W" else 1 if c=="D" else 0
               for c in str(form_str or "?????").upper()[-5:]]
        if not pts: return 5.0
        # Pad with neutral (1 pt) if fewer than 5 results
        while len(pts) < 5: pts.insert(0, 1)
        return sum(w * p for w, p in zip(_FORM_WEIGHTS, pts))
    h_form = _weighted_form(team_forms.get(match["home_id"], "?????"))
    a_form = _weighted_form(team_forms.get(match["away_id"], "?????"))
    if h_form >= 10 and h_stats.get("dominance",0) < 0: h_xg *= 0.92
    if h_form <= 4  and h_stats.get("dominance",0) > 0: h_xg *= 1.08
    if a_form >= 10 and a_stats.get("dominance",0) < 0: a_xg *= 0.92
    if a_form <= 4  and a_stats.get("dominance",0) > 0: a_xg *= 1.08

    # True xG overlay — use the xg stored in get_team_stats (no extra API call)
    h_true_xg = h_stats.get("true_xg")
    a_true_xg = a_stats.get("true_xg")
    # True xG blend — weight scales with sample size (0→0% trust, 5+→30% trust)
    h_xg_count = h_stats.get("true_xg_count", 0)
    a_xg_count = a_stats.get("true_xg_count", 0)
    if h_true_xg is not None and h_xg_count > 0:
        blend = min(h_xg_count / 5.0, 1.0) * 0.30
        h_xg = h_xg * (1 - blend) + h_true_xg * blend
    if a_true_xg is not None and a_xg_count > 0:
        blend = min(a_xg_count / 5.0, 1.0) * 0.30
        a_xg = a_xg * (1 - blend) + a_true_xg * blend

    # Fatigue
    h_fatigue = a_fatigue = False
    try:
        m_dt = datetime.fromisoformat(match["date"].replace("Z","+00:00"))
        if m_dt.tzinfo is None: m_dt = m_dt.replace(tzinfo=timezone.utc)
        def _tz(d):
            if d.tzinfo is None: return d.replace(tzinfo=timezone.utc)
            return d
        h_last = _tz(get_last_match_date(match["home_id"]))
        a_last = _tz(get_last_match_date(match["away_id"]))
        h_rest = (m_dt - h_last).days; a_rest = (m_dt - a_last).days
        diff   = h_rest - a_rest
        if diff <= -3 and h_rest <= 3:
            h_xg *= 0.85; a_xg *= 1.10; h_fatigue = True
        elif diff >= 3 and a_rest <= 3:
            a_xg *= 0.85; h_xg *= 1.10; a_fatigue = True
    except Exception as e:
        log.debug("Fatigue %s: %s", match_key, e)

    # Lookahead
    h_look = a_look = False
    try:
        m_dt2 = datetime.fromisoformat(match["date"].replace("Z","+00:00"))
        if m_dt2.tzinfo is None: m_dt2 = m_dt2.replace(tzinfo=timezone.utc)
        def _has_soon(tid):
            for d in team_future_fixtures.get(tid,[]):
                try:
                    fd = datetime.fromisoformat(d.replace("Z","+00:00"))
                    if fd.tzinfo is None: fd = fd.replace(tzinfo=timezone.utc)
                    if 1 <= (fd - m_dt2).days <= 4: return True
                except: pass
            return False
        h_look = _has_soon(match["home_id"])
        a_look = _has_soon(match["away_id"])
    except Exception: pass

    # Bogey team
    h2h_key  = f"{match['home_id']}-{match['away_id']}"
    h2h_hist = h2h_data.get(h2h_key, [])
    h2h_stats = h2h_data.get(f"{h2h_key}_stats", {"n":0,"h_win_rate":0.33,"draw_rate":0.25,"a_win_rate":0.33})
    bogey = None
    if len(h2h_hist) >= 3:
        if h_xg > a_xg*1.3 and sum(1 for x in h2h_hist if x["winner"]=="Home") == 0:
            bogey = match["away"]
        elif a_xg > h_xg*1.3 and sum(1 for x in h2h_hist if x["winner"]=="Away") == 0:
            bogey = match["home"]

    # Pinnacle implied probabilities (sharp signal)
    raw_odds = insight.get("odds", {})         # B365
    pin_odds  = insight.get("p_odds", {})      # Pinnacle (sharper signal)
    def odds_to_prob(o):
        try: return round(1.0 / float(o), 4) if float(o) > 1 else 0.0
        except: return 0.0
    # Prefer Pinnacle; fall back to B365 if Pinnacle not available
    pin_implied_h = odds_to_prob(pin_odds.get("Home", 0)) or odds_to_prob(raw_odds.get("Home", 0))
    pin_implied_a = odds_to_prob(pin_odds.get("Away", 0)) or odds_to_prob(raw_odds.get("Away", 0))

    # Market edge: model implied prob vs Pinnacle (positive = model sees more value)
    # Only meaningful when Pinnacle odds exist; otherwise 0.0
    import math as _math
    def _dc_home_prob(hx, ax, rho=-0.130):
        """Quick Dixon-Coles home win probability for edge computation."""
        try:
            n = 8
            m = [[0.0]*n for _ in range(n)]
            for i in range(n):
                ph = _math.exp(-hx) * (hx**i) / _math.factorial(i)
                for j in range(n):
                    pa = _math.exp(-ax) * (ax**j) / _math.factorial(j)
                    tau = (1.0 - hx*ax*rho if i==0 and j==0 else
                           1.0 + ax*rho    if i==1 and j==0 else
                           1.0 + hx*rho    if i==0 and j==1 else
                           1.0 - rho       if i==1 and j==1 else 1.0)
                    m[i][j] = ph * pa * max(0.0, tau)
            total = sum(m[i][j] for i in range(n) for j in range(n))
            if total <= 0: return 0.45, 0.27, 0.28
            p_h = sum(m[i][j] for i in range(n) for j in range(n) if i > j) / total
            p_d = sum(m[i][i] for i in range(n)) / total
            p_a = sum(m[i][j] for i in range(n) for j in range(n) if j > i) / total
            return round(p_h, 4), round(p_d, 4), round(p_a, 4)
        except Exception:
            return 0.45, 0.27, 0.28

    # Floor xG before DC call — Poisson requires λ > 0
    h_xg = max(0.10, h_xg); a_xg = max(0.10, a_xg)
    _rho = LEAGUE_RHO.get(match["league"], -0.130)
    _model_h, _model_d, _model_a = _dc_home_prob(h_xg, a_xg, _rho)
    # Edge = model probability − market implied probability (in percentage points)
    market_edge_h = round((_model_h - pin_implied_h) * 100, 2) if pin_implied_h > 0 else 0.0
    market_edge_a = round((_model_a - pin_implied_a) * 100, 2) if pin_implied_a > 0 else 0.0
    # Draw edge using remaining probability
    pin_implied_d = max(0.0, 1.0 - pin_implied_h - pin_implied_a) if pin_implied_h > 0 else 0.0
    market_edge_d = round((_model_d - pin_implied_d) * 100, 2) if pin_implied_d > 0 else 0.0

    ref_name = (match.get("referee") or "").lower()
    strict_ref = any(r.lower() in ref_name for r in STRICT_REFS)
    h_bounce = is_new_manager(match["home_id"])
    a_bounce = is_new_manager(match["away_id"])
    h_exodus = has_exodus(match["home_id"])
    a_exodus = has_exodus(match["away_id"])

    if h_bounce:                        h_xg *= 1.15
    if a_bounce:                        a_xg *= 1.15
    if h_stats.get("false_form"):       h_xg *= 1.10
    if a_stats.get("false_form"):       a_xg *= 1.10
    if h_exodus:                        h_xg *= 0.85
    if a_exodus:                        a_xg *= 0.85

    insight = match_insights.get(match["id"], {})
    h_susp = any(s["team_id"]==match["home_id"] for s in insight.get("suspensions",[]))
    a_susp = any(s["team_id"]==match["away_id"] for s in insight.get("suspensions",[]))
    if h_susp: h_xg *= 0.92
    if a_susp: a_xg *= 0.92

    # Injury differential — sum of impact_pct for each side (0.0 when no data)
    injuries = insight.get("injuries", [])
    h_inj_impact = sum(float(inj.get("impact_pct", 0.0))
                       for inj in injuries if inj.get("team_id") == match["home_id"])
    a_inj_impact = sum(float(inj.get("impact_pct", 0.0))
                       for inj in injuries if inj.get("team_id") == match["away_id"])
    injury_diff = round(h_inj_impact - a_inj_impact, 4)  # positive = home more injured

    xpts_delta = (float(pi_ratings.get(f"{match['home']}_xpts_delta", 0.0))
                  - float(pi_ratings.get(f"{match['away']}_xpts_delta", 0.0)))
    h_safety   = float(pi_ratings.get(f"{match['home']}_safety", 10.0))
    h_title    = float(pi_ratings.get(f"{match['home']}_title",  -15.0))
    a_safety   = float(pi_ratings.get(f"{match['away']}_safety", 10.0))
    a_title    = float(pi_ratings.get(f"{match['away']}_title",  -15.0))

    h_xg = max(0.20, h_xg); a_xg = max(0.20, a_xg)

    pro_predictions[match_key] = {
        # Core xG
        "h_xg": round(h_xg, 3), "a_xg": round(a_xg, 3),
        # Cards
        "h_yellow": h_stats.get("yellow_avg", 1.8), "a_yellow": a_stats.get("yellow_avg", 1.8),
        "h_red":    h_stats.get("red_avg",    0.1),  "a_red":    a_stats.get("red_avg",    0.1),
        # Tactical
        "h_fast":  h_stats.get("fast_start", 0),  "a_leak":  a_stats.get("late_leak",  0),
        "a_fast":  a_stats.get("fast_start", 0),  "h_leak":  h_stats.get("late_leak",  0),
        # Clean sheet / fail-to-score — home AND away stored separately (V3 key change)
        "cs_h":   h_stats.get("cs_h",  0.30),   "cs_a":   a_stats.get("cs_a",  0.25),
        "fts_h":  h_stats.get("fts_h", 0.25),   "fts_a":  a_stats.get("fts_a", 0.28),
        # Syndicate intel
        "h_lookahead": h_look,   "a_lookahead": a_look,
        "bogey_team":  bogey,    "strict_ref":  strict_ref,
        "h_bounce":    h_bounce, "a_bounce":    a_bounce,
        "h_false_form": bool(h_stats.get("false_form")),
        "a_false_form": bool(a_stats.get("false_form")),
        "h_exodus":    h_exodus, "a_exodus":    a_exodus,
        "h_fatigue":   h_fatigue,"a_fatigue":   a_fatigue,
        "h_susp":      h_susp,   "a_susp":      a_susp,
        # ML features — expanded V3.3 (venue xG + weighted form + market edge)
        "expected_points_delta": round(xpts_delta, 4),
        "pts_from_safety":       round(h_safety,   2),
        "pts_from_title":        round(h_title,    2),
        "a_pts_from_safety":     round(a_safety,   2),
        "a_pts_from_title":      round(a_title,    2),
        "injury_diff":           injury_diff,
        # H2H ML features
        "h2h_h_win_rate":  h2h_stats.get("h_win_rate", 0.33),
        "h2h_draw_rate":   h2h_stats.get("draw_rate",  0.25),
        "h2h_a_win_rate":  h2h_stats.get("a_win_rate", 0.33),
        "h2h_n":           h2h_stats.get("n", 0),
        # League market rates (draw tendency, scoring environment)
        "league_home_win_rate": l_avg.get("home_win_rate", 0.45),
        "league_draw_rate":     l_avg.get("draw_rate",     0.25),
        "league_avg_goals":     l_avg.get("avg_goals_pg",  2.65),
        # Pinnacle implied probability (sharp money signal)
        "pin_implied_h": pin_implied_h,
        "pin_implied_a": pin_implied_a,
        # Model vs Pinnacle edge (percentage points; positive = model disagrees with market upward)
        "market_edge_h": market_edge_h,
        "market_edge_a": market_edge_a,
        "market_edge_d": market_edge_d,
        # Card features (ref + team tendencies)
        "h_yellow": h_stats.get("yellow_avg", 1.8),
        "a_yellow": a_stats.get("yellow_avg", 1.8),
        # True xG flags for dashboard badge
        "h_true_xg": h_true_xg is not None,
        "a_true_xg": a_true_xg is not None,
        # League rho for dashboard DC matrix
        "rho": LEAGUE_RHO.get(match["league"], -0.130),
        # Form (for display)
        "h_form_pts": h_form, "a_form_pts": a_form,
    }

save_kv("pro_preds", pro_predictions)
save_kv("team_stats_cache", team_stats_cache)
save_kv("coach_cache",      coach_cache)
save_kv("events_cache",     events_cache)
save_kv("transfer_cache",   transfer_cache)
save_kv("player_cache",     player_cache)

log.info("✅ V3 pro predictions saved for %d matches.", len(pro_predictions))
log.info("✅ Total API requests this run: %d / 7500", _quota_used)
log.info("✅ train_master.py V3 complete.")