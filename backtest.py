"""
backtest.py — Offline backtesting against recent results.

Loads completed fixtures from database.sqlite, reconstructs predictions using
current model state, and outputs ROI tables by tier and pick type.

IMPORTANT LIMITATION: Uses *current* pi_ratings, team_forms, and league_averages
as proxies for match-time values.  This introduces look-ahead bias for older
records; results are most reliable for the most recent 4–6 weeks.
Matches with Pinnacle odds (from hist_odds_cache) produce meaningful ROI numbers;
those without use fixed fallback odds (shown separately).

Usage:
    python backtest.py [--min-tier {emerald,diamond+,diamond,gold}]
                       [--days N]   # restrict to last N days (default: all)
                       [--no-color]
"""

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DB_PATH      = "database.sqlite"
HF_JSON_PATH = "hf_data.json"
MODEL_FILES  = {
    "xgb_match":  "xgb_match_model.joblib",
    "lgb_match":  "lgb_match_model.joblib",
    "meta_match": "meta_match_model.joblib",
    "meta_scaler":"meta_scaler.joblib",
    "xgb_o25":    "xgb_o25_model.joblib",
    "lgb_o25":    "lgb_o25_model.joblib",
    "meta_o25":   "meta_o25_model.joblib",
    "meta_temp":  "meta_temperature.joblib",
}
TIER_THRESHOLDS = {"emerald": 85.0, "diamond+": 75.0, "diamond": 65.0, "gold": 0.0}
DEFAULT_RHO = -0.130
FALLBACK_ODDS = {"Home Win": 2.0, "Away Win": 2.7, "Draw": 3.3, "Over 2.5": 1.90}

# Feature lists — must match train_ml.py exactly after Task 1 + Task 2.
FEATURE_COLS = [
    "h_xg", "a_xg",
    "pi_diff", "expected_points_delta",
    "pts_from_safety", "pts_from_title",
    "a_pts_from_safety", "a_pts_from_title",
    "injury_diff",
    "cs_h", "cs_a", "fts_h", "fts_a",
    "cs_diff", "defensive_dominance", "attacking_vulnerability",
    "form_diff", "h_fast", "a_leak",
    "h2h_h_win_rate", "h2h_draw_rate", "h2h_a_win_rate",
    "league_home_win_rate", "league_draw_rate", "league_avg_goals",
    "pin_implied_h", "pin_implied_a",
    "market_edge_h", "market_edge_a",
]
FEATURE_COLS_O25 = [
    "h_xg", "a_xg",
    "cs_h", "cs_a", "fts_h", "fts_a",
    "defensive_dominance", "attacking_vulnerability",
    "h_fast", "a_leak",
    "league_avg_goals",
    "pin_implied_h", "pin_implied_a",
    "market_edge_h", "market_edge_a",
    "pi_diff",
    "injury_diff",
    "h2h_draw_rate",
    "form_diff",
]
META_RAW_COLS = [
    "h_xg", "a_xg", "pi_diff", "form_diff", "pin_implied_h", "pin_implied_a",
    "cs_h", "cs_a", "injury_diff", "league_avg_goals",
]

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
_hf_cache: dict = {}

def _load_hf_json() -> dict:
    global _hf_cache
    if _hf_cache:
        return _hf_cache
    try:
        with open(HF_JSON_PATH, encoding="utf-8") as f:
            _hf_cache = json.load(f)
    except Exception:
        _hf_cache = {}
    return _hf_cache


def load_kv(key: str, default=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
        row = c.fetchone()
        conn.close()
        if row:
            return json.loads(row["value"])
    except Exception as e:
        log.debug("SQLite load_kv(%s): %s", key, e)
    hf = _load_hf_json()
    if key in hf:
        return hf[key]
    return default if default is not None else {}


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
models: dict = {}

try:
    import joblib
    for name, path in MODEL_FILES.items():
        try:
            models[name] = joblib.load(path)
        except Exception:
            log.debug("Model not loaded: %s", path)
    if "xgb_match" in models:
        log.info("Models loaded: %s", list(models.keys()))
    else:
        log.warning("XGB match model missing — predictions will be DC-only.")
except ImportError:
    log.warning("joblib not available — DC-only mode.")


# ---------------------------------------------------------------------------
# DIXON-COLES HELPERS
# ---------------------------------------------------------------------------
LEAGUE_RHO = {
    "Premier League": -0.130, "La Liga": -0.120, "Bundesliga": -0.115,
    "Serie A": -0.125, "Ligue 1": -0.118, "Eredivisie": -0.100,
    "Pro League": -0.095, "Jupiler Pro League": -0.095,
    "Belgian Pro League": -0.095, "Champions League": -0.110,
    "UEFA Champions League": -0.110,
}

def _tau(i, j, lam, mu, rho):
    if   i == 0 and j == 0: return max(0.0, 1.0 - lam * mu * rho)
    elif i == 0 and j == 1: return max(0.0, 1.0 + lam * rho)
    elif i == 1 and j == 0: return max(0.0, 1.0 + mu * rho)
    elif i == 1 and j == 1: return max(0.0, 1.0 - rho)
    return 1.0

def build_dc_matrix(h_xg: float, a_xg: float, rho: float = DEFAULT_RHO, n: int = 10):
    m = np.zeros((n, n))
    for i in range(n):
        ph = math.exp(-h_xg) * (h_xg ** i) / math.factorial(i)
        for j in range(n):
            pa = math.exp(-a_xg) * (a_xg ** j) / math.factorial(j)
            m[i, j] = ph * pa * _tau(i, j, h_xg, a_xg, rho)
    t = m.sum()
    return m / t if t > 0 else m


def dc_probs(h_xg: float, a_xg: float, rho: float = DEFAULT_RHO):
    """Returns (p_away, p_draw, p_home) — class order 0=Away,1=Draw,2=Home."""
    m = build_dc_matrix(h_xg, a_xg, rho)
    p_h = float(np.sum(np.tril(m, -1)))
    p_d = float(np.sum(np.diag(m)))
    p_a = float(np.sum(np.triu(m, 1)))
    p_o25 = float(sum(m[i, j] for i in range(10) for j in range(10) if i + j > 2))
    return p_a, p_d, p_h, p_o25


# ---------------------------------------------------------------------------
# FEATURE CONSTRUCTION
# ---------------------------------------------------------------------------
def _form_pts(form_str) -> float:
    if not form_str or str(form_str) == "?????":
        return 5.0
    return float(sum(3 if c == "W" else 1 if c == "D" else 0 for c in str(form_str).upper()))


def build_feat(rec: dict, pro_preds: dict, pi_ratings: dict,
               team_forms: dict, league_avgs: dict, hist_odds: dict) -> dict:
    """Build a feature dict for one recent result record."""
    home  = rec.get("home", "")
    away  = rec.get("away", "")
    league = rec.get("league", "")
    fid   = str(rec.get("fixture_id", ""))
    mk    = f"{home} vs {away}"

    pp = pro_preds.get(mk, {})

    # xG — pro_preds first, then pi-based estimate
    h_xg = float(pp.get("h_xg", 0.0))
    a_xg = float(pp.get("a_xg", 0.0))
    if h_xg == 0.0 or a_xg == 0.0:
        lg = league_avgs.get(league, {})
        lg_avg = float(lg.get("avg_goals_pg", 2.65))
        h_pi = float(pi_ratings.get(home, 1000))
        a_pi = float(pi_ratings.get(away, 1000))
        pi_adj = max(-1.5, min(1.5, (h_pi - a_pi) / 200.0))
        h_xg = max(0.2, min(5.0, lg_avg * 0.53 + pi_adj * 0.3))
        a_xg = max(0.2, min(5.0, lg_avg * 0.47 - pi_adj * 0.3))

    # Pi difference
    h_pi = float(pi_ratings.get(home, 1000))
    a_pi = float(pi_ratings.get(away, 1000))
    pi_diff = h_pi - a_pi

    # Form
    h_form_raw = team_forms.get(home, {})
    a_form_raw = team_forms.get(away, {})
    if isinstance(h_form_raw, dict):
        h_form_str = h_form_raw.get("form", "?????")
    else:
        h_form_str = str(h_form_raw)
    if isinstance(a_form_raw, dict):
        a_form_str = a_form_raw.get("form", "?????")
    else:
        a_form_str = str(a_form_raw)
    form_diff = _form_pts(h_form_str) - _form_pts(a_form_str)

    # Defensive stats
    cs_h  = float(pp.get("cs_h",  0.30))
    cs_a  = float(pp.get("cs_a",  0.25))
    fts_h = float(pp.get("fts_h", 0.25))
    fts_a = float(pp.get("fts_a", 0.28))
    cs_diff              = cs_h - cs_a
    defensive_dominance  = (cs_h + fts_a) / 2.0
    attacking_vulnerability = (cs_a + fts_h) / 2.0

    # League averages
    lg = league_avgs.get(league, {})
    league_avg_goals     = float(lg.get("avg_goals_pg",      2.65))
    league_home_win_rate = float(lg.get("home_win_rate",     0.45))
    league_draw_rate     = float(lg.get("draw_rate",         0.25))

    # Pinnacle odds from hist_odds_cache
    odds_entry = hist_odds.get(fid, {})
    pin_h = float(odds_entry.get("pin_implied_h", pp.get("pin_implied_h", 0.0)))
    pin_a = float(odds_entry.get("pin_implied_a", pp.get("pin_implied_a", 0.0)))
    has_odds = pin_h > 0.01 and pin_a > 0.01

    # DC fair probs for market edge
    dc_pa, dc_pd, dc_ph, _ = dc_probs(h_xg, a_xg,
                                      LEAGUE_RHO.get(league, DEFAULT_RHO))
    market_edge_h = dc_ph - pin_h if has_odds else 0.0
    market_edge_a = dc_pa - pin_a if has_odds else 0.0

    return {
        "h_xg":  h_xg,  "a_xg":  a_xg,
        "pi_diff": pi_diff,
        "expected_points_delta": float(pp.get("expected_points_delta", 0.0)),
        "pts_from_safety":       float(pp.get("pts_from_safety",       10.0)),
        "pts_from_title":        float(pp.get("pts_from_title",       -15.0)),
        "a_pts_from_safety":     float(pp.get("a_pts_from_safety",     10.0)),
        "a_pts_from_title":      float(pp.get("a_pts_from_title",     -15.0)),
        "injury_diff":           float(pp.get("injury_diff",            0.0)),
        "cs_h": cs_h, "cs_a": cs_a, "fts_h": fts_h, "fts_a": fts_a,
        "cs_diff": cs_diff,
        "defensive_dominance": defensive_dominance,
        "attacking_vulnerability": attacking_vulnerability,
        "form_diff": form_diff,
        "h_fast": float(pp.get("h_fast", 22.0)),
        "a_leak": float(pp.get("a_leak", 16.0)),
        "h2h_h_win_rate": float(pp.get("h2h_h_win_rate", 0.33)),
        "h2h_draw_rate":  float(pp.get("h2h_draw_rate",  0.25)),
        "h2h_a_win_rate": float(pp.get("h2h_a_win_rate", 0.33)),
        "league_home_win_rate": league_home_win_rate,
        "league_draw_rate":     league_draw_rate,
        "league_avg_goals":     league_avg_goals,
        "pin_implied_h": pin_h,
        "pin_implied_a": pin_a,
        "market_edge_h": market_edge_h,
        "market_edge_a": market_edge_a,
        "_has_odds":   has_odds,
        "_pin_h": pin_h, "_pin_a": pin_a,
        "_dc_ph": dc_ph, "_dc_pd": dc_pd, "_dc_pa": dc_pa,
    }


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
def clamp(p: float, lo: float = 1.0, hi: float = 99.0) -> float:
    return max(lo, min(hi, p))


def predict(feat: dict, league: str = "") -> dict:
    """
    Returns a dict with p_h, p_d, p_a, p_o25 (all 0-100) and tier/top_pick.
    Falls back gracefully if models are missing.
    """
    rho = LEAGUE_RHO.get(league, DEFAULT_RHO)
    h_xg, a_xg = feat["h_xg"], feat["a_xg"]

    # DC baseline
    dc_pa, dc_pd, dc_ph, dc_o25 = dc_probs(h_xg, a_xg, rho)
    p_h, p_d, p_a = dc_ph * 100, dc_pd * 100, dc_pa * 100
    p_o25 = dc_o25 * 100

    if "xgb_match" in models and "meta_match" in models:
        try:
            row = pd.DataFrame([[feat[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
            row_o25 = pd.DataFrame([[feat[c] for c in FEATURE_COLS_O25]], columns=FEATURE_COLS_O25)

            xgb_p = models["xgb_match"].predict_proba(row)[0]
            ml_pa, ml_pd, ml_ph = float(xgb_p[0]), float(xgb_p[1]), float(xgb_p[2])

            # O2.5 via separate feature set
            if "xgb_o25" in models:
                o25_p = models["xgb_o25"].predict_proba(row_o25)[0]
                ml_p_o25 = float(o25_p[1]) * 100
            else:
                ml_p_o25 = p_o25  # DC fallback

            # Meta stack
            if "lgb_match" in models:
                lgb_p = models["lgb_match"].predict_proba(row)[0]
                meta_in = np.array([[
                    dc_pa, dc_pd, dc_ph,
                    ml_pa, ml_pd, ml_ph,
                    float(lgb_p[0]), float(lgb_p[1]), float(lgb_p[2]),
                ]])
            else:
                meta_in = np.array([[dc_pa, dc_pd, dc_ph, ml_pa, ml_pd, ml_ph]])

            # Scale raw meta features
            if "meta_scaler" in models:
                raw_meta = np.array([[feat[c] for c in META_RAW_COLS]], dtype=np.float64)
                scaled = models["meta_scaler"].transform(raw_meta)
                meta_in = np.hstack((meta_in, scaled))

            final = models["meta_match"].predict_proba(meta_in)[0]
            _mt = models.get("meta_temp", 1.0)
            if _mt is not None and abs(_mt - 1.0) > 0.01:
                _lg = np.log(np.clip(final, 1e-15, 1.0)) / _mt
                _lg -= _lg.max()
                _ex = np.exp(_lg)
                final = _ex / _ex.sum()
            p_a_r, p_d_r, p_h_r = float(final[0]) * 100, float(final[1]) * 100, float(final[2]) * 100
            s = p_h_r + p_d_r + p_a_r
            p_h, p_d, p_a = (p_h_r / s * 100, p_d_r / s * 100, p_a_r / s * 100)

            # O2.5 meta blend
            if "meta_o25" in models and "lgb_o25" in models:
                lgb_o25_p = models["lgb_o25"].predict_proba(row_o25)[0]
                dc_o25_col = np.array([[dc_o25]])
                xgb_o25_col = np.array([[float(o25_p[1])]])
                lgb_o25_col = np.array([[float(lgb_o25_p[1])]])
                meta_o25_in = np.hstack((dc_o25_col, xgb_o25_col, lgb_o25_col))
                o25_meta = models["meta_o25"].predict_proba(meta_o25_in)[0]
                p_o25 = float(o25_meta[1]) * 100
            else:
                p_o25 = clamp(p_o25 * 0.5 + ml_p_o25 * 0.5)

        except Exception as e:
            log.debug("ML prediction failed: %s — DC only.", e)

    p_h, p_d, p_a = clamp(p_h), clamp(p_d), clamp(p_a)
    p_o25 = clamp(p_o25)

    # Tier assignment (same logic as render_match_card)
    picks = {"Home Win": p_h, "Draw": p_d, "Away Win": p_a, "Over 2.5": p_o25}
    top_pick = max(picks, key=picks.get)
    top_p = picks[top_pick]

    if top_p >= 85.0:
        tier = "EMERALD"
    elif top_p >= 75.0:
        tier = "DIAMOND+"
    elif top_p >= 65.0:
        tier = "DIAMOND"
    else:
        tier = "GOLD"

    return {
        "p_h": p_h, "p_d": p_d, "p_a": p_a, "p_o25": p_o25,
        "tier": tier, "top_pick": top_pick, "top_p": top_p,
    }


# ---------------------------------------------------------------------------
# RESULT EVALUATION
# ---------------------------------------------------------------------------
def parse_score(score_str: str):
    """Returns (h_goals, a_goals) or (None, None) on failure."""
    try:
        parts = str(score_str).strip().split("-")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None, None


def is_correct(pick: str, h: int, a: int) -> bool:
    if pick == "Home Win":  return h > a
    if pick == "Away Win":  return a > h
    if pick == "Draw":      return h == a
    if pick == "Over 2.5":  return (h + a) > 2
    if pick == "Under 3.5": return (h + a) < 4
    return False


def implied_decimal_odds(pick: str, feat: dict) -> float:
    """Best available decimal odds for a pick (Pinnacle implied → decimal)."""
    pin_h, pin_a = feat["_pin_h"], feat["_pin_a"]
    if feat["_has_odds"]:
        if pick == "Home Win" and pin_h > 0.01:
            return round(1.0 / pin_h, 3)
        if pick == "Away Win" and pin_a > 0.01:
            return round(1.0 / pin_a, 3)
        if pick == "Draw":
            pin_d = max(0.01, 1.0 - pin_h - pin_a)
            return round(1.0 / pin_d, 3)
    return FALLBACK_ODDS.get(pick, 2.0)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Football AI Backtester")
    parser.add_argument("--min-tier", choices=["emerald", "diamond+", "diamond", "gold"],
                        default="gold", help="Minimum tier to include (default: gold = all)")
    parser.add_argument("--days", type=int, default=0,
                        help="Restrict to last N days (0 = all records)")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    log.info("Loading data from SQLite / hf_data.json…")
    recent      = load_kv("recent",       [])
    pro_preds   = load_kv("pro_preds",    {})
    pi_ratings  = load_kv("pi_ratings",   {})
    team_forms  = load_kv("team_forms",   {})
    league_avgs = load_kv("league_averages", {})
    hist_odds   = load_kv("hist_odds_cache", {})

    log.info("recent=%d  pro_preds=%d  hist_odds=%d  pi_ratings=%d",
             len(recent), len(pro_preds), len(hist_odds), len(pi_ratings))

    if not recent:
        log.error("No recent records found — run train_master.py first.")
        sys.exit(1)

    # Date cutoff
    cutoff = datetime.min
    if args.days > 0:
        cutoff = datetime.now() - timedelta(days=args.days)

    tier_order = ["EMERALD", "DIAMOND+", "DIAMOND", "GOLD"]
    min_tier_idx = tier_order.index(args.min_tier.upper().replace("DIAMOND-PLUS", "DIAMOND+"))

    rows = []
    skipped = 0

    for rec in recent:
        if not isinstance(rec, dict):
            continue
        score_str = rec.get("score", "")
        h_goals, a_goals = parse_score(score_str)
        if h_goals is None:
            skipped += 1
            continue

        # Date filter
        date_str = rec.get("date", "")
        if args.days > 0 and date_str:
            try:
                rec_date = datetime.fromisoformat(date_str[:19])
                if rec_date < cutoff:
                    continue
            except Exception:
                pass

        home  = rec.get("home", "")
        away  = rec.get("away", "")
        league = rec.get("league", "")

        feat = build_feat(rec, pro_preds, pi_ratings, team_forms, league_avgs, hist_odds)
        pred = predict(feat, league=league)

        tier_idx = tier_order.index(pred["tier"])
        if tier_idx > min_tier_idx:
            continue  # below requested minimum tier

        pick      = pred["top_pick"]
        correct   = is_correct(pick, h_goals, a_goals)
        dec_odds  = implied_decimal_odds(pick, feat)
        has_odds  = feat["_has_odds"]
        profit    = (dec_odds - 1.0) if correct else -1.0

        rows.append({
            "home":      home, "away": away, "league": league,
            "score":     score_str,
            "tier":      pred["tier"],
            "pick":      pick,
            "confidence": round(pred["top_p"], 1),
            "correct":   correct,
            "odds":      dec_odds,
            "has_odds":  has_odds,
            "profit":    profit,
        })

    if not rows:
        log.warning("No rows after filtering (skipped %d records without scores).", skipped)
        sys.exit(0)

    df = pd.DataFrame(rows)
    log.info("Scored %d matches (%d skipped — missing score).", len(df), skipped)

    # -----------------------------------------------------------------------
    # OUTPUT TABLE helpers
    # -----------------------------------------------------------------------
    SEP = "─" * 72

    def roi_str(p: float) -> str:
        sign = "+" if p >= 0 else ""
        return f"{sign}{p:.1f}%"

    def print_table(title: str, group_col: str, order: list = None):
        print(f"\n{'═'*72}")
        print(f"  {title}")
        print(f"{'═'*72}")
        hdr = f"{'Group':<18}  {'Picks':>6}  {'Correct':>8}  {'Accuracy':>9}  {'ROI (pin)':>10}  {'ROI (all)':>10}"
        print(hdr)
        print(SEP)

        groups = order if order else sorted(df[group_col].unique())
        totals = {"picks": 0, "correct": 0, "profit_pin": 0.0,
                  "n_pin": 0, "profit_all": 0.0}

        for g in groups:
            sub = df[df[group_col] == g]
            if sub.empty:
                continue
            n     = len(sub)
            n_ok  = sub["correct"].sum()
            acc   = n_ok / n * 100

            sub_pin  = sub[sub["has_odds"]]
            roi_pin  = (sub_pin["profit"].sum() / len(sub_pin) * 100) if len(sub_pin) > 0 else float("nan")
            roi_all  = (sub["profit"].sum() / n * 100)

            totals["picks"]      += n
            totals["correct"]    += n_ok
            totals["profit_all"] += sub["profit"].sum()
            if len(sub_pin) > 0:
                totals["profit_pin"] += sub_pin["profit"].sum()
                totals["n_pin"]      += len(sub_pin)

            pin_str = roi_str(roi_pin) if not math.isnan(roi_pin) else "  n/a"
            print(f"{g:<18}  {n:>6}  {n_ok:>8}  {acc:>8.1f}%  {pin_str:>10}  {roi_str(roi_all):>10}")

        print(SEP)
        t_acc     = totals["correct"] / totals["picks"] * 100 if totals["picks"] else 0
        t_roi_all = totals["profit_all"] / totals["picks"] * 100 if totals["picks"] else 0
        t_roi_pin = (totals["profit_pin"] / totals["n_pin"] * 100) if totals["n_pin"] > 0 else float("nan")
        pin_str   = roi_str(t_roi_pin) if not math.isnan(t_roi_pin) else "  n/a"
        print(f"{'TOTAL':<18}  {totals['picks']:>6}  {totals['correct']:>8}  "
              f"{t_acc:>8.1f}%  {pin_str:>10}  {roi_str(t_roi_all):>10}")

    # -----------------------------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------------------------
    print(f"\n  Football AI Backtest — {len(df)} matches")
    print(f"  Models: {list(models.keys()) or 'DC-only'}")
    print(f"  ⚠  NOTE: Uses current ratings/forms — look-ahead bias applies to older records.")
    print(f"  ⚠  Pinnacle ROI column covers {df['has_odds'].sum()} / {len(df)} matches with odds data.")

    print_table("ROI by Tier", "tier", order=["EMERALD", "DIAMOND+", "DIAMOND", "GOLD"])
    print_table("ROI by Pick Type", "pick",
                order=["Home Win", "Away Win", "Draw", "Over 2.5"])

    # Per-tier × per-pick breakdown
    print(f"\n{'═'*72}")
    print("  Accuracy by Tier × Pick")
    print(f"{'═'*72}")
    pivot = df.groupby(["tier", "pick"])["correct"].agg(["sum", "count"])
    pivot["accuracy"] = (pivot["sum"] / pivot["count"] * 100).round(1)
    pivot.columns = ["Correct", "Picks", "Accuracy%"]
    pivot = pivot.reindex(pd.MultiIndex.from_product(
        [["EMERALD", "DIAMOND+", "DIAMOND", "GOLD"],
         ["Home Win", "Away Win", "Draw", "Over 2.5"]],
        names=["tier", "pick"]
    )).dropna(subset=["Picks"])
    print(pivot.to_string())
    print()


if __name__ == "__main__":
    main()
