"""
data_quality_check.py — Inspect the quality of data in database.sqlite / hf_data.json.

Checks:
  1. Pro-preds feature coverage for upcoming matches
  2. Pinnacle odds coverage for upcoming matches
  3. hist_odds_cache coverage for recent results

Usage: python data_quality_check.py
"""

import json
import os
import sqlite3
from datetime import datetime

DB_PATH = "database.sqlite"
HF_PATH = "hf_data.json"

# Raw keys that exist directly in pro_preds (excludes 3 derived:
# cs_diff, defensive_dominance, attacking_vulnerability — computed inline).
# Also excludes form_diff (computed from h_form_pts - a_form_pts).
# We check h_form_pts / a_form_pts instead.
RAW_FEATURE_KEYS = [
    "h_xg", "a_xg",
    "pi_diff", "expected_points_delta",
    "pts_from_safety", "pts_from_title",
    "a_pts_from_safety", "a_pts_from_title",
    "injury_diff",
    "cs_h", "cs_a", "fts_h", "fts_a",
    "h_form_pts", "a_form_pts",   # proxies for form_diff
    "h_fast", "a_leak",
    "h2h_h_win_rate", "h2h_draw_rate", "h2h_a_win_rate",
    "league_home_win_rate", "league_draw_rate", "league_avg_goals",
    "pin_implied_h", "pin_implied_a",
    "market_edge_h", "market_edge_a",
]
N_RAW = len(RAW_FEATURE_KEYS)  # 27


def load_kv(key, default=None):
    """Read from SQLite kv_store; fall back to hf_data.json."""
    if os.path.exists(DB_PATH):
        try:
            with sqlite3.connect(DB_PATH) as con:
                row = con.execute(
                    "SELECT value FROM kv_store WHERE key=?", (key,)
                ).fetchone()
                if row:
                    return json.loads(row[0])
        except Exception:
            pass
    if os.path.exists(HF_PATH):
        try:
            with open(HF_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return data.get(key, default)
        except Exception:
            pass
    return default


def pct(n, total):
    return f"{n / total * 100:.1f}%" if total else "N/A"


def bar(n, total, width=20):
    filled = int(n / total * width) if total else 0
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main():
    print("=" * 60)
    print("  DATA QUALITY REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    upcoming    = load_kv("upcoming", [])
    pro_preds   = load_kv("pro_preds", {})
    recent      = load_kv("recent", [])
    hist_odds   = load_kv("hist_odds_cache", {})

    # ------------------------------------------------------------------ #
    # 1. Pro-preds feature coverage for upcoming matches
    # ------------------------------------------------------------------ #
    print(f"\n{'─'*60}")
    print("  1. UPCOMING MATCHES — PRO-PREDS COVERAGE")
    print(f"{'─'*60}")

    n_upcoming = len(upcoming)
    n_has_preds = 0
    n_fully_populated = 0
    n_partial = 0
    missing_keys_counter: dict[str, int] = {}

    for match in upcoming:
        key = f"{match.get('home', '?')} vs {match.get('away', '?')}"
        pp = pro_preds.get(key)
        if pp is None:
            continue
        n_has_preds += 1
        missing = [k for k in RAW_FEATURE_KEYS if pp.get(k) is None]
        for k in missing:
            missing_keys_counter[k] = missing_keys_counter.get(k, 0) + 1
        if not missing:
            n_fully_populated += 1
        else:
            n_partial += 1

    n_missing_preds = n_upcoming - n_has_preds

    print(f"  Total upcoming matches  : {n_upcoming:>5}")
    print(f"  Has pro_preds entry     : {n_has_preds:>5}  {bar(n_has_preds, n_upcoming)}  {pct(n_has_preds, n_upcoming)}")
    if n_has_preds:
        print(f"  Fully populated ({N_RAW:2d}/27) : {n_fully_populated:>5}  {bar(n_fully_populated, n_has_preds)}  {pct(n_fully_populated, n_has_preds)}")
        print(f"  Partially populated     : {n_partial:>5}  {bar(n_partial, n_has_preds)}  {pct(n_partial, n_has_preds)}")
    print(f"  Missing pro_preds entry : {n_missing_preds:>5}  {bar(n_missing_preds, n_upcoming)}  {pct(n_missing_preds, n_upcoming)}")

    if missing_keys_counter:
        print("\n  Most-missing raw features:")
        for k, cnt in sorted(missing_keys_counter.items(), key=lambda x: -x[1])[:10]:
            print(f"    {k:<30} missing in {cnt:>3} match(es)")

    # ------------------------------------------------------------------ #
    # 2. Pinnacle odds coverage for upcoming matches
    # ------------------------------------------------------------------ #
    print(f"\n{'─'*60}")
    print("  2. UPCOMING MATCHES — PINNACLE ODDS COVERAGE")
    print(f"{'─'*60}")

    n_with_pin = 0
    n_no_pin = 0
    no_pin_matches = []

    for match in upcoming:
        key = f"{match.get('home', '?')} vs {match.get('away', '?')}"
        pp = pro_preds.get(key, {})
        pin_h = pp.get("pin_implied_h", 0.0) or 0.0
        if pin_h > 0.0:
            n_with_pin += 1
        else:
            n_no_pin += 1
            no_pin_matches.append(key)

    print(f"  Has Pinnacle data       : {n_with_pin:>5}  {bar(n_with_pin, n_upcoming)}  {pct(n_with_pin, n_upcoming)}")
    print(f"  Missing (pin_implied=0) : {n_no_pin:>5}  {bar(n_no_pin, n_upcoming)}  {pct(n_no_pin, n_upcoming)}")

    if no_pin_matches:
        print("\n  Matches without Pinnacle odds:")
        for m in no_pin_matches[:20]:
            print(f"    • {m}")
        if len(no_pin_matches) > 20:
            print(f"    ... and {len(no_pin_matches) - 20} more")

    # ------------------------------------------------------------------ #
    # 3. hist_odds_cache coverage for recent results
    # ------------------------------------------------------------------ #
    print(f"\n{'─'*60}")
    print("  3. RECENT RESULTS — HIST ODDS CACHE COVERAGE")
    print(f"{'─'*60}")

    n_recent = len(recent)
    n_with_odds = 0
    n_no_fid = 0

    for r in recent:
        fid = str(r.get("fixture_id", ""))
        if not fid or fid == "None":
            n_no_fid += 1
            continue
        if fid in hist_odds:
            n_with_odds += 1

    n_missing_odds = n_recent - n_with_odds - n_no_fid

    print(f"  Total recent results    : {n_recent:>5}")
    print(f"  Has fixture_id          : {n_recent - n_no_fid:>5}  {bar(n_recent - n_no_fid, n_recent)}  {pct(n_recent - n_no_fid, n_recent)}")
    print(f"  Has hist_odds entry     : {n_with_odds:>5}  {bar(n_with_odds, n_recent)}  {pct(n_with_odds, n_recent)}")
    print(f"  Missing odds            : {n_missing_odds:>5}  {bar(n_missing_odds, n_recent)}  {pct(n_missing_odds, n_recent)}")
    print(f"  Missing fixture_id      : {n_no_fid:>5}  {bar(n_no_fid, n_recent)}  {pct(n_no_fid, n_recent)}")

    fills_needed = max(0, (n_recent - n_with_odds - n_no_fid))
    runs_est = (fills_needed // 120) + 1
    print(f"\n  hist_odds fills at 120/run → ~{runs_est} more train_master.py run(s) for full coverage")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{'─'*60}")
    print("  SUMMARY")
    print(f"{'─'*60}")
    print(f"  Upcoming: {n_has_preds}/{n_upcoming} have pro_preds  |  "
          f"{n_with_pin}/{n_upcoming} have Pinnacle odds")
    print(f"  Recent:   {n_with_odds}/{n_recent} have hist_odds_cache entries  ({pct(n_with_odds, n_recent)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
