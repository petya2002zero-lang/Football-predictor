import streamlit as st
import streamlit.components.v1 as _stc
import pandas as pd
import numpy as np
import os
import re
import logging
import difflib
import sqlite3
import json
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XGBoost / LightGBM wrappers for sklearn calibration float32 compat
# ---------------------------------------------------------------------------
import xgboost as xgb
try:
    from sklearn.utils._tags import ClassifierTags, TargetTags

    def _xgb_sklearn_tags(self):
        from sklearn.base import BaseEstimator
        tags = BaseEstimator.__sklearn_tags__(self)
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags(multi_class=True)
        tags.target_tags.required = True
        return tags

    xgb.XGBClassifier.__sklearn_tags__ = _xgb_sklearn_tags
except Exception:
    xgb.XGBClassifier._estimator_type = "classifier"


class _XGBFloat32(xgb.XGBClassifier):
    """Wraps predict_proba to return float32 — matches X dtype for sklearn calibration."""
    def predict_proba(self, X, **kwargs):
        return super().predict_proba(X, **kwargs).astype(np.float32)

# Register in a stub "train_ml" module so joblib can deserialize models
# without importing the real train_ml.py (which runs the full training pipeline).
import types as _types
import sys as _sys
_stub = _types.ModuleType("train_ml")
_stub._XGBFloat32 = _XGBFloat32
_sys.modules.setdefault("train_ml", _stub)

try:
    import lightgbm as lgb

    class _LGBMFloat32(lgb.LGBMClassifier):
        """Wraps predict_proba to return float32 — matches X dtype for sklearn calibration."""
        def predict_proba(self, X, **kwargs):
            return super().predict_proba(X, **kwargs).astype(np.float32)

    _stub._LGBMFloat32 = _LGBMFloat32
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    log.warning("joblib not installed — ML models disabled. pip install joblib")

# ---------------------------------------------------------------------------
# ML ENGINE  (V5.0 Meta-Ensemble — 29 features, XGB + LGB + scaled meta)
# ---------------------------------------------------------------------------
ML_ENABLED = False
ml_model_match   = None
ml_model_o25     = None
lgb_model_match  = None
lgb_model_o25    = None
meta_model_match = None
meta_model_o25   = None
meta_scaler      = None
meta_temperature = 1.0

# V3.3: 26 features. Must match train_ml.py FEATURE_COLS exactly.
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

# Goals-market feature set — must match FEATURE_COLS_O25 in train_ml.py exactly.
# Separate from FEATURE_COLS: drops table-position and 1X2-distribution features.
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

# V5.0: 10 raw features passed through StandardScaler into meta-learner.
# Must match train_ml.py META_RAW_COLS exactly.
META_RAW_COLS = [
    "h_xg", "a_xg", "pi_diff", "form_diff", "pin_implied_h", "pin_implied_a",
    "cs_h", "cs_a", "injury_diff", "league_avg_goals",
]

if _JOBLIB_AVAILABLE:
    try:
        ml_model_match   = joblib.load("xgb_match_model.joblib")
        ml_model_o25     = joblib.load("xgb_o25_model.joblib")
        meta_model_match = joblib.load("meta_match_model.joblib")
        ML_ENABLED = True
        log.info("ML models loaded (V5.0 — XGB + meta).")
        try:
            lgb_model_match = joblib.load("lgb_match_model.joblib")
            log.info("LightGBM match model loaded.")
        except Exception:
            log.info("lgb_match_model.joblib not found — XGB-only match stack.")
        try:
            meta_scaler = joblib.load("meta_scaler.joblib")
            log.info("Meta scaler loaded.")
        except Exception:
            log.info("meta_scaler.joblib not found — raw meta features (unscaled).")
        try:
            lgb_model_o25 = joblib.load("lgb_o25_model.joblib")
            log.info("LightGBM O2.5 model loaded.")
        except Exception:
            log.info("lgb_o25_model.joblib not found — XGB-only O2.5 stack.")
        try:
            meta_model_o25 = joblib.load("meta_o25_model.joblib")
            log.info("Meta O2.5 model loaded.")
        except Exception:
            log.info("meta_o25_model.joblib not found — blended O2.5.")
        try:
            meta_temperature = joblib.load("meta_temperature.joblib")
            log.info("Meta temperature loaded (T=%.2f).", meta_temperature)
        except Exception:
            log.info("meta_temperature.joblib not found — no temperature scaling.")
    except Exception as e:
        log.warning("ML model loading failed: %s — running in math-only mode.", e)

# ---------------------------------------------------------------------------
# LEAGUE-SPECIFIC Dixon-Coles rho  (V3 upgrade — was hardcoded -0.13 for all)
# ---------------------------------------------------------------------------
LEAGUE_RHO: dict = {
    "Premier League":                -0.134,
    "La Liga":                       -0.127,
    "Bundesliga":                    -0.119,
    "Serie A":                       -0.131,
    "Ligue 1":                       -0.122,
    "Champions League":              -0.115,
    "UEFA Champions League":         -0.115,
    "Europa League":                 -0.110,
    "UEFA Europa League":            -0.110,
    "Conference League":             -0.108,
    "UEFA Europa Conference League": -0.108,
}
DEFAULT_RHO = -0.130

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Elite Quant Predictor", page_icon="🏆", layout="wide")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
DEFAULT_LEAGUES = [
    "UEFA Champions League", "Champions League",
    "UEFA Europa League", "Europa League",
    "UEFA Europa Conference League", "Europa Conference League",
    "UEFA Conference League", "Conference League",
    "AFC Champions League Elite", "Premier League", "La Liga", "Bundesliga",
    "Serie A", "Ligue 1", "Eredivisie", "Pro League", "Jupiler Pro League",
    "Belgian Pro League", "Süper Lig", "Primeira Liga", "Championship",
    "Saudi Pro League", "Major League Soccer",
]
CUP_NAMES = [
    "FA Cup", "Copa del Rey", "DFB Pokal", "Coppa Italia",
    "Coupe de France", "Carabao Cup", "KNVB Beker",
]


def format_time(iso_date_str: str) -> str:
    try:
        dt = datetime.fromisoformat(str(iso_date_str).replace("Z", "+00:00"))
        try:
            dt = dt.astimezone(ZoneInfo("Europe/Budapest"))
        except Exception:
            pass
        return dt.strftime("%Y.%m.%d %H:%M")
    except Exception:
        return str(iso_date_str)


# ---------------------------------------------------------------------------
# CSS STYLES
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .match-card { background-color: #262730; padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #444; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .team-logo { width: 30px; height: 30px; margin-right: 12px; vertical-align: middle; object-fit: contain; }
    .league-logo { width: 20px; height: 20px; margin-right: 8px; vertical-align: middle; object-fit: contain; filter: grayscale(100%); opacity: 0.7; }
    .tier-badge { padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; display: inline-block; margin-bottom: 10px;}
    .tier-emerald { background: linear-gradient(135deg, #006266, #00b894, #55efc4); color: #1e1e1e; border: 2px solid #00ffcc; box-shadow: 0 0 18px rgba(0, 255, 204, 0.8), inset 0 0 10px rgba(255, 255, 255, 0.6); font-weight: 900; letter-spacing: 1px; text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8); }
    .tier-diamond-plus { background: linear-gradient(135deg, #0abde3, #5f27cd); color: white; border: 1px solid #00d2d3; box-shadow: 0 0 8px rgba(10, 189, 227, 0.4); }
    .tier-diamond { background-color: #0abde3; color: white; border: 1px solid #00d2d3; }
    .tier-gold { background-color: #feca57; color: #2d3436; border: 1px solid #ff9f43; }
    .league-header { color: #b2bec3; font-size: 13px; margin-bottom: 12px; display: flex; align-items: center; }
    .team-row { display: flex; align-items: center; margin-bottom: 8px; }
    .team-name { font-size: 19px; font-weight: 700; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
    .strategy-headline { color: #fab1a0; font-size: 13px; font-weight: 600; margin-top: 15px; font-style: italic; }
    .form-dots { display: flex; gap: 3px; justify-content: center; }
    .dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
    .dot-w { background-color: #00b894; } .dot-d { background-color: #636e72; } .dot-l { background-color: #d63031; }
    .xg-container-new { background-color: #1e272e; padding: 15px; border-radius: 10px; border: 1px solid #444; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    .xg-title-new { font-size: 11px; color: #b2bec3; text-transform: uppercase; font-weight: 700; margin-bottom: 10px; letter-spacing: 1px; }
    .xg-bar-wrapper { display: flex; height: 28px; width: 100%; background-color: #2d3436; border-radius: 14px; overflow: hidden; border: 1px solid #333; }
    .xg-bar-home { background: linear-gradient(90deg, #00b894, #0984e3); display: flex; align-items: center; padding-left: 12px; }
    .xg-bar-away { background: linear-gradient(90deg, #d63031, #fab1a0); display: flex; align-items: center; justify-content: flex-end; padding-right: 12px; }
    .xg-val-text { color: white; font-weight: 800; font-size: 13px; text-shadow: 0 1px 2px rgba(0,0,0,0.8); }
    .prob-header { font-size: 12px; color: #aaa; margin-bottom: 10px; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; }
    .prog-label { font-size: 11px; color: #ccc; margin-bottom: 4px; display: flex; justify-content: space-between; font-weight: 600; }
    .custom-bar-bg { background-color: #2d3436; height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 14px; border: 1px solid #444; }
    .custom-bar-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
    .stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 20px; }
    .stat-box { background-color: #1e1e1e; padding: 10px 5px; border-radius: 8px; text-align: center; border: 1px solid #333; }
    .stat-label { font-size: 9px; color: #888; text-transform: uppercase; font-weight: 700; }
    .stat-value { font-size: 15px; font-weight: 800; color: #fff; }
    .stExpander { border: none !important; background-color: transparent !important; margin-top: 20px !important; }
    .stExpander div[role="button"] { background-color: #222 !important; border-radius: 8px !important; border: 1px solid #444 !important; padding: 10px 15px !important; color: #ddd !important; font-weight: 600 !important; }
    .h2h-container { background: #181b21; border-radius: 12px; overflow: hidden; border: 1px solid #333; }
    .h2h-header-row { display: flex; justify-content: space-between; background: #2d3436; padding: 10px 15px; font-size: 11px; font-weight: 700; color: #b2bec3; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #444; }
    .h2h-row { display: flex; justify-content: space-between; padding: 8px 15px; border-bottom: 1px solid #2d3436; font-size: 13px; align-items: center; transition: background 0.2s; }
    .h2h-row:hover { background: #262a30; }
    .h2h-row:last-child { border-bottom: none; }
    .h2h-logo { width: 18px; height: 18px; object-fit: contain; vertical-align: middle; margin: 0 5px; }
    .badge-win { background: rgba(0, 184, 148, 0.2); color: #00b894; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; border: 1px solid rgba(0, 184, 148, 0.4); }
    .badge-loss { background: rgba(214, 48, 49, 0.2); color: #ff7675; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; border: 1px solid rgba(214, 48, 49, 0.4); }
    .badge-draw { background: rgba(178, 190, 195, 0.2); color: #dfe6e9; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; border: 1px solid rgba(178, 190, 195, 0.4); }
    .value-card { background: linear-gradient(145deg, #1e272e, #13171a); border: 1px solid #444; border-radius: 12px; padding: 20px; text-align: center; position: relative; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .value-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #00b894, #0984e3); border-radius: 12px 12px 0 0; }
    .value-title { color: #b2bec3; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 5px; }
    .value-score { font-size: 38px; font-weight: 900; color: white; margin: 5px 0 15px 0; text-shadow: 0 2px 10px rgba(0,0,0,0.5); font-family: 'Arial Black', sans-serif; }
    .kelly-divider { border: 0; border-top: 1px dashed #444; margin: 15px 0; }
    .value-stake { font-size: 28px; font-weight: 800; color: #55efc4; margin-top: 5px; text-shadow: 0 0 10px rgba(85, 239, 196, 0.3); }
    .bankroll-sub { color: #636e72; font-size: 10px; margin-top: 8px; font-style: italic; }
    .parlay-card { background: #2d3436; border: 1px solid #fab1a0; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .parlay-header { font-size: 20px; font-weight: bold; color: #fab1a0; margin-bottom: 10px; border-bottom: 1px solid #444; padding-bottom: 5px; }
    .value-box-green { background-color: rgba(0, 184, 148, 0.2); border: 1px solid #00b894; padding: 15px; border-radius: 8px; text-align: center; }
    .value-box-red { background-color: rgba(214, 48, 49, 0.2); border: 1px solid #d63031; padding: 15px; border-radius: 8px; text-align: center; }
    .insight-badge { background: #333; color: #ddd; font-size: 10px; padding: 2px 6px; border-radius: 4px; margin-right: 5px; border: 1px solid #555; display: inline-block; margin-bottom: 4px; }
    .insight-val { background: #1a472a; color: #4ade80; border-color: #22c55e; }
    .insight-consensus { background: #2c244d; color: #a78bfa; border-color: #8b5cf6; }
    .div-table-header { display: flex; background-color: #1e1e1e; color: #888; font-size: 11px; font-weight: 700; text-transform: uppercase; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; letter-spacing: 1px; }
    .div-table-row { display: flex; align-items: center; background-color: #262730; padding: 10px 15px; border-radius: 8px; margin-bottom: 6px; transition: transform 0.2s, background-color 0.2s; border: 1px solid #333; }
    .div-table-row:hover { transform: translateX(5px); background-color: #2d3436; border-color: #555; }
    .col-rank { flex: 0 0 40px; text-align: center; }
    .col-team { flex: 1; display: flex; align-items: center; gap: 10px; overflow: hidden; white-space: nowrap; }
    .col-stats { flex: 0 0 60px; text-align: center; color: #aaa; font-size: 12px; }
    .col-gd { flex: 0 0 50px; text-align: center; font-weight: bold; font-size: 12px; }
    .col-pts { flex: 0 0 50px; text-align: center; font-weight: 800; color: #00cec9; font-size: 14px; }
    .col-form { flex: 0 0 80px; display: flex; justify-content: center; }
    .rank-badge { width: 24px; height: 24px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; color: white; }
    .rk-ucl { background: #00b894; } .rk-uel { background: #e17055; } .rk-mid { background: #636e72; } .rk-rel { background: #d63031; }
    .table-logo { width: 22px; height: 22px; object-fit: contain; }
    .team-txt { font-weight: 600; color: #eee; font-size: 13px; text-overflow: ellipsis; overflow: hidden; }
    .metric-card { background-color: #1e272e; border: 1px solid #444; border-radius: 8px; padding: 15px; text-align: center; }
    .metric-title { font-size: 10px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #fff; }
    .ev-positive { color: #00b894; font-weight: bold; }
    .ev-negative { color: #d63031; font-weight: bold; }
    .sharp-box { border-left: 3px solid #00d2d3; background: rgba(0, 210, 211, 0.1); padding: 15px; margin-bottom: 15px; }
    .key-missing-badge { background: rgba(214, 48, 49, 0.2); border: 1px solid #d63031; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: bold; color: #ff7675; display: inline-block; margin-top: 5px; }
    .regular-missing-badge { background: rgba(254, 202, 87, 0.1); border: 1px solid #feca57; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: bold; color: #feca57; display: inline-block; margin-top: 5px; }
    .market-box { background: #1e272e; border: 1px solid #333; padding: 12px; border-radius: 8px; text-align: center; margin-bottom: 10px; }
    .market-title { font-size: 11px; color: #b2bec3; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 1px;}
    .market-row { display: flex; justify-content: space-between; font-size: 13px; padding: 5px 0; border-bottom: 1px solid #2d3436; }
    .market-row:last-child { border-bottom: none; }
    .market-val { font-weight: 800; color: #fff; }
    .odds-fair { color: #00cec9; font-weight: bold; }
    .model-metric-box { background: #1e272e; border: 1px solid #333; padding: 12px; border-radius: 8px; text-align: center; }
    .model-metric-label { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .model-metric-value { font-size: 22px; font-weight: 800; color: #55efc4; }
    .shap-bar-bg { background: #1e272e; border-radius: 4px; height: 10px; margin-top: 3px; overflow: hidden; }
    .shap-bar-fill { height: 100%; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DATABASE UTILS
# ---------------------------------------------------------------------------

def get_db_connection() -> sqlite3.Connection:
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.sqlite")
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value TEXT)")
        c.execute(
            "CREATE TABLE IF NOT EXISTS bet_log ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "date TEXT, match TEXT, pick TEXT, odds REAL, "
            "stake REAL, result TEXT, profit REAL)"
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("init_db: %s — running in read-only mode (hf_data.json fallback active).", e)


def save_kv(key: str, val):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("REPLACE INTO kv_store (key, value) VALUES (?, ?)", (key, json.dumps(val)))
    conn.commit()
    conn.close()


# hf_data.json — exported by the GitHub Actions HF sync step.
# On HF Spaces database.sqlite is absent (>10 MB HF limit);
# this file carries the kv_store contents instead.
_HF_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_data.json")
_hf_data_cache: dict = {}
_hf_data_loaded = False

def _load_hf_json() -> dict:
    """Load hf_data.json into memory (cached for process lifetime)."""
    global _hf_data_cache, _hf_data_loaded
    if _hf_data_loaded:
        return _hf_data_cache
    _hf_data_loaded = True
    try:
        with open(_HF_DATA_PATH, "r") as f:
            _hf_data_cache = json.load(f)
        log.info("hf_data.json loaded (%d keys).", len(_hf_data_cache))
    except Exception:
        _hf_data_cache = {}
    return _hf_data_cache


def load_kv(key: str, default=None):
    # Primary: SQLite (GitHub runner, local dev)
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
        row = c.fetchone()
        conn.close()
        if row:
            return json.loads(row["value"])
    except Exception as e:
        log.warning("load_kv SQLite (%s): %s — trying hf_data.json fallback.", key, e)
    # Fallback: hf_data.json (HF Spaces — database.sqlite not present)
    hf = _load_hf_json()
    if key in hf:
        return hf[key]
    return default if default is not None else {}


def load_bet_log_db() -> list:
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM bet_log ORDER BY id ASC")
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        log.warning("load_bet_log_db: %s", e)
        return []


def save_bet_log_full(bets: list):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM bet_log")
    for b in bets:
        c.execute(
            "INSERT INTO bet_log (date, match, pick, odds, stake, result, profit) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (b.get("date"), b.get("match"), b.get("pick"),
             b.get("odds"), b.get("stake"), b.get("result"), b.get("profit")),
        )
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_data() -> dict:
    d: dict = {}
    for k in ["upcoming", "pro_preds", "standings", "recent",
              "team_forms", "h2h", "top_scorers", "insights",
              "league_averages", "pi_ratings"]:
        d[k] = load_kv(k, [] if k in ("upcoming", "recent") else {})

    # Legacy .pkl migration
    if not d["upcoming"]:
        legacy = {
            "upcoming": "upcoming_matches.pkl", "pro_preds": "pro_predictions.pkl",
            "standings": "standings.pkl", "recent": "recent_results.pkl",
            "team_forms": "team_forms.pkl", "h2h": "h2h.pkl",
            "top_scorers": "top_scorers.pkl", "insights": "match_insights.pkl",
            "league_averages": "league_averages.pkl",
        }
        for key, fn in legacy.items():
            if os.path.exists(fn) and _JOBLIB_AVAILABLE:
                try:
                    obj = joblib.load(fn)
                    if isinstance(obj, dict) and key == "recent":
                        obj = []
                    save_kv(key, obj)
                    d[key] = obj
                except Exception as e:
                    log.warning("Legacy pkl %s: %s", fn, e)

    raw_bet_log = load_bet_log_db()
    existing_keys = {f"{b.get('date')}_{b.get('match')}" for b in raw_bet_log}
    added_new = False

    if os.path.exists("bet_log.pkl") and _JOBLIB_AVAILABLE:
        try:
            for b in joblib.load("bet_log.pkl"):
                k = f"{b.get('date')}_{b.get('match')}"
                if k not in existing_keys:
                    raw_bet_log.append(b); existing_keys.add(k); added_new = True
            os.remove("bet_log.pkl")
        except Exception as e:
            log.warning("Legacy bet_log.pkl: %s", e)

    for fn in ["data.csv", "data (2).csv"]:
        if os.path.exists(fn):
            try:
                try:
                    df_csv = pd.read_csv(fn)
                except Exception:
                    df_csv = pd.read_csv(fn, sep=None, engine="python")
                df_csv.columns = df_csv.columns.str.lower().str.strip()
                for bet in df_csv.to_dict("records"):
                    k = f"{bet.get('date')}_{bet.get('match')}"
                    if k not in existing_keys:
                        raw_bet_log.append(bet); existing_keys.add(k); added_new = True
            except Exception as e:
                log.warning("CSV import %s: %s", fn, e)

    if added_new:
        save_bet_log_full(raw_bet_log)
    d["bet_log"] = raw_bet_log
    return d


data = load_data()

# ---------------------------------------------------------------------------
# MATCH-MATH CACHE  (session-scoped; invalidated when data TTL reloads)
# ---------------------------------------------------------------------------
_MATH_VER = id(data["upcoming"])
if st.session_state.get("_math_ver") != _MATH_VER:
    st.session_state["_math_ver"]   = _MATH_VER
    st.session_state["_math_cache"] = {}

def _cached_match_math(m: dict):
    key = str(m.get("id") or f"{m.get('home')} vs {m.get('away')}")
    if key not in st.session_state["_math_cache"]:
        st.session_state["_math_cache"][key] = get_match_math(m)
    return st.session_state["_math_cache"][key]

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏆 AI Sports")
    page = st.selectbox("Navigate", [
        "🔮 League Predictions", "🏆 National Cup Predictions",
        "🟢 Emerald/Diamond Picks", "🟢 Emerald/Diamond Results",
        "🏗️ Parlay Builder", "📊 Tables", "📈 Profit Tracker",
        "📊 Sharp EV Dashboard", "🔎 Tactical Filters",
        "📉 CLV & Value Tracker", "🏦 Bankroll Management",
        "🧠 Model Health", "📡 Data Control Center",
    ])
    st.divider()

    version_label = "V3.2 XGB+LGB Active" if (ML_ENABLED and lgb_model_match) else ("V3.2 XGB Active" if ML_ENABLED else "Math Engine Active")
    version_color = "#00b894" if ML_ENABLED else "#d63031"
    feature_hint  = "24 features + XGB+LGB+DC Stack" if (ML_ENABLED and lgb_model_match) else ("24 features + XGB+DC Stack" if ML_ENABLED else "Offline — run train_ml.py")
    st.markdown(
        f"<div style='background-color:rgba(0,0,0,0.3); border:1px solid {version_color}; "
        f"padding:10px; border-radius:8px; text-align:center; margin-bottom:15px;'>"
        f"<b style='color:{version_color};'>🧠 {version_label}</b><br>"
        f"<span style='font-size:11px; color:#ddd;'>{feature_hint}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 🧠 AI Strategy Guide")
    st.markdown("""
    <div style='background-color: #1e272e; padding: 15px; border-radius: 10px; border: 1px solid #444; font-size: 13px; color: #ccc;'>
        <ul style='padding-left: 20px; margin: 0;'>
            <li style='margin-bottom: 8px;'><b>1. Value &gt; Winners:</b> Look for odds that are too high.</li>
            <li style='margin-bottom: 8px;'><b>2. The 2% Rule:</b> Never bet more than 2-3% of funds.</li>
            <li><b>3. Trust the Math:</b> Long-term statistics prevail.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🕵️ Syndicate Intel Glossary"):
        st.markdown("""
        <div style='font-size: 12px; color: #ddd;'>
            <p><b>🤖 SUPER CONSENSUS:</b> Both our AI and the bookmaker ML fully agree on an edge.</p>
            <p><b>🛑 SUSPENDED:</b> Key player missing due to card accumulation.</p>
            <p><b>🪫 FATIGUED:</b> Team has a severe rest disadvantage.</p>
            <p><b>⚖️ TRUE xG EDGE:</b> Model uses actual pitch xG data.</p>
            <p><b>⚠️ TRAP (Lookahead):</b> Team has a major match in 3-4 days.</p>
            <p><b>👻 HEX (Bogey Team):</b> Favorite historically struggles vs this underdog.</p>
            <p><b>👔 NEW MANAGER:</b> Recent coaching change — short-term overperformance likely.</p>
            <p><b>🚨 FALSE FORM:</b> Recent bad result was caused by an early red card.</p>
            <p><b>📉 EXODUS:</b> Team recently sold key players.</p>
            <p><b>🦈 SHARP MONEY:</b> Pinnacle odds movement signals professional betting.</p>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.caption(
        f"DB: {len(data['upcoming'])} upcoming | {len(data['recent'])} recent"
    )

if not data["upcoming"] and not data["recent"]:
    hf_hint = " (hf_data.json also empty)" if not _load_hf_json() else ""
    st.error(f"⚠️ No data. Run train_master.py to populate the database.{hf_hint}")
    st.stop()


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def fuzzy_match_name(name1: str, name2: str, threshold: float = 0.6) -> bool:
    def clean(n):
        n = n.lower()
        for w in [" vs ", "fc", "cf", "sc", "city", "united", "utd",
                  "town", "borussia", "real ", "sporting ", "sv ", "as "]:
            n = n.replace(w, "")
        return n.replace(" ", "").strip()
    n1, n2 = clean(name1), clean(name2)
    if n1 in n2 or n2 in n1:
        return True
    return difflib.SequenceMatcher(None, n1, n2).ratio() > threshold


def get_form_pts(form_str) -> int:
    if not form_str or str(form_str) == "?????":
        return 5
    return sum(3 if c == "W" else 1 if c == "D" else 0 for c in str(form_str).upper())


def clamp_prob(p: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, p))


def normalise_1x2(p_h: float, p_d: float, p_a: float):
    total = p_h + p_d + p_a
    if total <= 0:
        return 33.33, 33.33, 33.33
    f = 100.0 / total
    return p_h * f, p_d * f, p_a * f


def calculate_bet_result(bet: dict, results_map: dict) -> dict:
    match_desc = bet.get("match", "")
    is_parlay = "Parlay:" in match_desc
    legs = []
    if is_parlay:
        raw = match_desc.replace("Parlay:", "").strip()
        if not raw:
            return bet
        for p in raw.split(" | "):
            try:
                m_str, mkt_raw = p.rsplit(" (", 1)
                legs.append((m_str.strip(), mkt_raw.rstrip(")").strip()))
            except Exception:
                pass
    else:
        legs.append((match_desc, bet.get("pick")))

    if not legs:
        return bet

    leg_outcomes = []
    for m_name, market in legs:
        score = None
        for r_key, r_score in results_map.items():
            if fuzzy_match_name(m_name, r_key):
                score = r_score; break
        if not score:
            leg_outcomes.append("Pending"); continue
        try:
            h_s, a_s = map(int, score.split("-"))
            total_g = h_s + a_s
            clean_market = str(market).replace(" ", "").lower()
            sub_picks = re.split(r"[&+]|and", clean_market)
            sub_results = []
            for sub in sub_picks:
                if not sub: continue
                res = None
                ou = re.search(r"(over|under|o|u)\s*(\d+\.?\d*)", sub)
                if ou:
                    t_str = ou.group(1); val = float(ou.group(2))
                    res = total_g > val if t_str in ("over","o") else total_g < val
                elif "homewin" in sub or sub == "1": res = h_s > a_s
                elif "awaywin" in sub or sub == "2": res = a_s > h_s
                elif "draw" in sub or sub == "x":    res = h_s == a_s
                elif "btts" in sub:                  res = (h_s > 0) and (a_s > 0)
                sub_results.append(res if res is not None else "Error")
            if "Error" in sub_results:     leg_outcomes.append("Error")
            elif all(sub_results):         leg_outcomes.append("Win")
            else:                          leg_outcomes.append("Loss")
        except Exception:
            leg_outcomes.append("Error")

    if "Error" in leg_outcomes:
        bet["result"] = "Pending"
    elif "Loss" in leg_outcomes:
        bet["result"] = "Lost"
        try:    bet["profit"] = -float(bet.get("stake", 0))
        except: bet["profit"] = 0.0
    elif all(x == "Win" for x in leg_outcomes) and leg_outcomes:
        bet["result"] = "Won"
        try:
            stake = float(bet.get("stake", 0)); odds = float(bet.get("odds", 1.0))
            bet["profit"] = (stake * odds) - stake
        except: bet["profit"] = 0.0
    return bet


def calculate_ev(prob_pct: float, decimal_odds: float) -> float:
    return ((prob_pct / 100.0) * decimal_odds - 1) * 100


def calculate_kelly(prob_pct: float, decimal_odds: float,
                    bankroll: float = 1000, fractional: float = 0.25) -> float:
    if decimal_odds <= 1: return 0.0
    p = prob_pct / 100.0; q = 1 - p; b = decimal_odds - 1
    kelly_f = (b * p - q) / b
    return max(0.0, kelly_f * fractional * bankroll)


# ---------------------------------------------------------------------------
# DIXON-COLES PROBABILITY MATRIX  (fixed τ, league-specific rho)
# ---------------------------------------------------------------------------

def dc_tau(i: int, j: int, mu: float, nu: float, rho: float) -> float:
    if   i==0 and j==0: return max(0.0, 1.0 - mu*nu*rho)
    elif i==1 and j==0: return max(0.0, 1.0 + nu*rho)
    elif i==0 and j==1: return max(0.0, 1.0 + mu*rho)
    elif i==1 and j==1: return max(0.0, 1.0 - rho)
    return 1.0


def build_dc_matrix(h_xg: float, a_xg: float, rho: float = DEFAULT_RHO, n: int = 10) -> np.ndarray:
    m = np.zeros((n, n))
    for i in range(n):
        ph = math.exp(-h_xg) * (h_xg**i) / math.factorial(i)
        for j in range(n):
            pa = math.exp(-a_xg) * (a_xg**j) / math.factorial(j)
            m[i, j] = ph * pa * dc_tau(i, j, h_xg, a_xg, rho)
    t = m.sum()
    return m / t if t > 0 else m


# ---------------------------------------------------------------------------
# CORE MATCH MATH ENGINE  (V3: league-specific rho + 14-feature ML blend)
# ---------------------------------------------------------------------------

def get_match_math(match_data: dict):
    home_id   = match_data.get("home_id", 0)
    away_id   = match_data.get("away_id", 0)
    home_name = match_data.get("home", "Home")
    away_name = match_data.get("away", "Away")
    league    = match_data.get("league", "")
    match_key = f"{home_name} vs {away_name}"

    # League-specific rho (V3 upgrade)
    rho = LEAGUE_RHO.get(league, DEFAULT_RHO)

    # Base xG from pro_preds (train_master.py populates these)
    pp = data["pro_preds"].get(match_key, {})
    # Pro_preds may also carry a per-match rho computed in train_master
    rho = pp.get("rho", rho)

    h_xg     = float(pp.get("h_xg", 1.45))
    a_xg     = float(pp.get("a_xg", 1.20))
    h_yellow = float(pp.get("h_yellow", 1.8))
    a_yellow = float(pp.get("a_yellow", 1.8))
    h_red    = float(pp.get("h_red",    0.1))
    a_red    = float(pp.get("a_red",    0.1))

    # H2H goal average adjustment
    h2h_key = f"{home_id}-{away_id}"
    h2h_goals = []
    for hm in data["h2h"].get(str(h2h_key), [])[:3]:
        try:
            s = hm["score"].split("-")
            h2h_goals.append(int(s[0]) + int(s[1]))
        except Exception:
            pass
    if h2h_goals:
        avg = sum(h2h_goals) / len(h2h_goals)
        if avg >= 3.5:   h_xg *= 1.35; a_xg *= 1.35
        elif avg >= 2.5: h_xg *= 1.15; a_xg *= 1.15

    # Pi rating diff
    h_pi    = float(data["pi_ratings"].get(str(home_name), 0.0))
    a_pi    = float(data["pi_ratings"].get(str(away_name), 0.0))
    pi_diff = h_pi - a_pi

    # Form points
    h_form_pts = get_form_pts(data["team_forms"].get(str(home_id)))
    a_form_pts = get_form_pts(data["team_forms"].get(str(away_id)))
    form_diff  = float(h_form_pts - a_form_pts)

    # Injury penalties
    insight      = data["insights"].get(str(match_data.get("id")), {})
    injuries     = insight.get("injuries", [])
    ts_h_data    = data["top_scorers"].get(str(home_id), [])
    ts_a_data    = data["top_scorers"].get(str(away_id), [])
    if isinstance(ts_h_data, str):
        ts_h_data = [{"name": ts_h_data.split("(")[0].strip(), "role": "Key Player"}]
    if isinstance(ts_a_data, str):
        ts_a_data = [{"name": ts_a_data.split("(")[0].strip(), "role": "Key Player"}]

    h_penalty = 0.0; a_penalty = 0.0
    h_injury_alerts = []; a_injury_alerts = []

    for inj in injuries:
        if isinstance(inj, str):
            team_part = ""; player_part = inj
            if ":" in inj:
                team_part, player_part = inj.split(":", 1)
                player_part = player_part.split("(")[0].strip()
            is_key_h = any(fuzzy_match_name(player_part, ts["name"], 0.5)
                           for ts in ts_h_data) if (home_name in team_part or not team_part) else False
            is_key_a = any(fuzzy_match_name(player_part, ts["name"], 0.5)
                           for ts in ts_a_data) if (away_name in team_part or not team_part) else False
            if is_key_h:
                h_penalty += 0.15
                h_injury_alerts.append(f"⬇️ -15% xG | {home_name} missing {player_part}")
            elif home_name in team_part:
                h_penalty += 0.02
            if is_key_a:
                a_penalty += 0.15
                a_injury_alerts.append(f"⬇️ -15% xG | {away_name} missing {player_part}")
            elif away_name in team_part:
                a_penalty += 0.02
        elif isinstance(inj, dict):
            t_name  = inj.get("team_name", "")
            p_name  = inj.get("player_name", "")
            stats_s = inj.get("stats_str", "")
            impact  = float(inj.get("impact_pct", 0.0))
            if impact == 0.0 and (inj.get("is_key") or inj.get("is_regular")):
                impact = 0.15 if inj.get("is_key") else 0.05
                if not stats_s: stats_s = f"Old Cache (-{int(impact*100)}% xG)"
            if fuzzy_match_name(home_name, t_name, 0.6) or home_name in t_name:
                h_penalty += impact
                if impact >= 0.10: h_injury_alerts.append(f"⬇️ {p_name} | {stats_s}")
                elif impact > 0.0: h_injury_alerts.append(f"📉 {p_name} | {stats_s}")
            elif fuzzy_match_name(away_name, t_name, 0.6) or away_name in t_name:
                a_penalty += impact
                if impact >= 0.10: a_injury_alerts.append(f"⬇️ {p_name} | {stats_s}")
                elif impact > 0.0: a_injury_alerts.append(f"📉 {p_name} | {stats_s}")

    h_xg = max(0.1, h_xg * (1 - h_penalty))
    a_xg = max(0.1, a_xg * (1 - a_penalty))
    injury_diff = h_penalty - a_penalty

    # -----------------------------------------------------------------------
    # Dixon-Coles matrix  (V3: league-specific rho)
    # -----------------------------------------------------------------------
    prob_matrix = build_dc_matrix(h_xg, a_xg, rho=rho)

    p_h   = clamp_prob(float(np.sum(np.tril(prob_matrix, -1))) * 100)
    p_a   = clamp_prob(float(np.sum(np.triu(prob_matrix, 1)))  * 100)
    p_d   = clamp_prob(float(np.sum(np.diag(prob_matrix)))     * 100)
    p_o25 = clamp_prob(sum(prob_matrix[i,j] for i in range(10) for j in range(10) if i+j>2)*100)
    p_u35 = clamp_prob(sum(prob_matrix[i,j] for i in range(10) for j in range(10) if i+j<4)*100)
    p_btts = clamp_prob(sum(prob_matrix[i,j] for i in range(1,10) for j in range(1,10))*100)
    p_o15  = clamp_prob(sum(prob_matrix[i,j] for i in range(10) for j in range(10) if i+j>1)*100)
    p_u45  = clamp_prob(sum(prob_matrix[i,j] for i in range(10) for j in range(10) if i+j<5)*100)

    p_h, p_d, p_a = normalise_1x2(p_h, p_d, p_a)

    # Park-the-bus BTTS adjuster
    if (pp.get("cs_h", 0) > 0.45 and pp.get("fts_a", 0) > 0.40) or \
       (pp.get("cs_a", 0) > 0.45 and pp.get("fts_h", 0) > 0.40):
        p_btts = clamp_prob(p_btts * 0.70)
        p_u35  = clamp_prob(p_u35  * 1.15)

    # Asian handicap
    p_h_m15 = clamp_prob(sum(prob_matrix[i,j] for i in range(10) for j in range(10) if i-j>=2)*100)
    p_h_p15 = clamp_prob(sum(prob_matrix[i,j] for i in range(10) for j in range(10) if i-j>=-1)*100)
    p_a_m15 = clamp_prob(sum(prob_matrix[i,j] for i in range(10) for j in range(10) if j-i>=2)*100)
    p_a_p15 = clamp_prob(sum(prob_matrix[i,j] for i in range(10) for j in range(10) if j-i>=-1)*100)

    exp_h_corners = 3.2 + (h_xg * 1.6)
    exp_a_corners = 3.2 + (a_xg * 1.6)

    best_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    pred_h, pred_a = int(best_idx[0]), int(best_idx[1])
    if pred_h == 1 and pred_a == 1 and p_o25 > 65:
        score = "2-1" if p_h > p_a else "1-2"
    else:
        score = f"{pred_h}-{pred_a}"

    # -----------------------------------------------------------------------
    # V3.1 Meta-Ensemble stacking  (DC + XGB + LGB blend)
    # -----------------------------------------------------------------------
    if ML_ENABLED and ml_model_match and meta_model_match:
        try:
            xpts_delta = float(pp.get("expected_points_delta", 0.0))
            pts_safety = float(pp.get("pts_from_safety",       10.0))
            pts_title  = float(pp.get("pts_from_title",       -15.0))
            cs_h       = float(pp.get("cs_h",   0.30))
            cs_a       = float(pp.get("cs_a",   0.25))
            fts_h      = float(pp.get("fts_h",  0.25))
            fts_a      = float(pp.get("fts_a",  0.28))
            h_fast     = float(pp.get("h_fast", 22.0))
            a_leak     = float(pp.get("a_leak", 16.0))

            a_pts_safety = float(pp.get("a_pts_from_safety", 10.0))
            a_pts_title  = float(pp.get("a_pts_from_title", -15.0))
            h2h_hw  = float(pp.get("h2h_h_win_rate",  0.33))
            h2h_dr  = float(pp.get("h2h_draw_rate",   0.25))
            h2h_aw  = float(pp.get("h2h_a_win_rate",  0.33))
            lg_hw   = float(pp.get("league_home_win_rate", 0.45))
            lg_dr   = float(pp.get("league_draw_rate",     0.25))
            lg_ag   = float(pp.get("league_avg_goals",     2.65))
            pin_h   = float(pp.get("pin_implied_h", 0.0))
            pin_a   = float(pp.get("pin_implied_a", 0.0))

            mkt_edge_h = float(pp.get("market_edge_h", 0.0))
            mkt_edge_a = float(pp.get("market_edge_a", 0.0))
            cs_diff              = cs_h - cs_a
            defensive_dominance  = (cs_h + fts_a) / 2.0
            attacking_vulnerability = (cs_a + fts_h) / 2.0
            features = pd.DataFrame([[
                h_xg, a_xg, pi_diff, xpts_delta, pts_safety, pts_title,
                a_pts_safety, a_pts_title, injury_diff,
                cs_h, cs_a, fts_h, fts_a,
                cs_diff, defensive_dominance, attacking_vulnerability,
                form_diff, h_fast, a_leak,
                h2h_hw, h2h_dr, h2h_aw,
                lg_hw, lg_dr, lg_ag,
                pin_h, pin_a,
                mkt_edge_h, mkt_edge_a,
            ]], columns=FEATURE_COLS)

            # Base XGBoost (classes: 0=Away, 1=Draw, 2=Home)
            xgb_probs = ml_model_match.predict_proba(features)[0]
            ml_p_a, ml_p_d, ml_p_h = float(xgb_probs[0]), float(xgb_probs[1]), float(xgb_probs[2])

            features_o25 = pd.DataFrame([[
                h_xg, a_xg,
                cs_h, cs_a, fts_h, fts_a,
                defensive_dominance, attacking_vulnerability,
                h_fast, a_leak,
                lg_ag,
                pin_h, pin_a,
                mkt_edge_h, mkt_edge_a,
                pi_diff,
                injury_diff,
                h2h_dr,
                form_diff,
            ]], columns=FEATURE_COLS_O25)
            o25_probs = ml_model_o25.predict_proba(features_o25)[0]
            ml_p_o25  = float(o25_probs[1]) * 100

            # Build V5.0 meta stack: DC(3) + XGB(3) + LGB(3) + scaled_raw(10) = 19
            if lgb_model_match is not None:
                lgb_probs = lgb_model_match.predict_proba(features)[0]
                lgb_p_a, lgb_p_d, lgb_p_h = float(lgb_probs[0]), float(lgb_probs[1]), float(lgb_probs[2])
                raw_meta = np.array([[
                    h_xg, a_xg, pi_diff, form_diff, pin_h, pin_a,
                    cs_h, cs_a, injury_diff, lg_ag,
                ]], dtype=np.float32)
                raw_meta = np.clip(raw_meta, -10.0, 10.0)   # guard against scale drift
                if meta_scaler is not None:
                    raw_meta = meta_scaler.transform(raw_meta)
                    raw_meta = np.clip(raw_meta, -5.0, 5.0)  # bound scaled output
                meta_features = np.hstack([
                    np.array([[
                        p_a/100.0, p_d/100.0, p_h/100.0,
                        ml_p_a,    ml_p_d,    ml_p_h,
                        lgb_p_a,   lgb_p_d,   lgb_p_h,
                    ]]),
                    raw_meta,
                ])
            else:
                meta_features = np.array([[
                    p_a/100.0, p_d/100.0, p_h/100.0,
                    ml_p_a,    ml_p_d,    ml_p_h,
                ]])
            final_probs = meta_model_match.predict_proba(meta_features)[0]
            if meta_temperature is not None and abs(meta_temperature - 1.0) > 0.01:
                logits = np.log(np.clip(final_probs, 1e-15, 1.0))
                scaled = logits / meta_temperature
                scaled -= scaled.max()
                exp_s = np.exp(scaled)
                final_probs = exp_s / exp_s.sum()
            final_probs = np.clip(final_probs, 0.03, 0.97)
            final_probs /= final_probs.sum()
            p_a_raw = float(final_probs[0]) * 100
            p_d_raw = float(final_probs[1]) * 100
            p_h_raw = float(final_probs[2]) * 100
            p_h, p_d, p_a = normalise_1x2(p_h_raw, p_d_raw, p_a_raw)

            # O2.5: V5.0 meta stack (DC + XGB + LGB → meta) or simple blend
            if lgb_model_o25 is not None and meta_model_o25 is not None:
                lgb_o25_probs = lgb_model_o25.predict_proba(features_o25)[0]
                dc_o25 = p_o25 / 100.0
                meta_o25_features = np.array([[
                    dc_o25, float(o25_probs[1]), float(lgb_o25_probs[1]),
                ]])
                o25_meta_probs = meta_model_o25.predict_proba(meta_o25_features)[0]
                p_o25 = clamp_prob(float(o25_meta_probs[1]) * 100)
            else:
                p_o25 = clamp_prob(p_o25 * 0.5 + ml_p_o25 * 0.5)
        except Exception as e:
            log.warning("ML blend failed for %s: %s — DC only.", match_key, e)

    details = {
        "h_injury_alerts": h_injury_alerts,
        "a_injury_alerts": a_injury_alerts,
        "h_inj": len(injuries), "a_inj": 0,
        "h_minus_1_5": p_h_m15, "a_plus_1_5":  p_a_p15,
        "a_minus_1_5": p_a_m15, "h_plus_1_5":  p_h_p15,
        "h_corners": exp_h_corners, "a_corners": exp_a_corners,
        "h_yellow": h_yellow, "a_yellow": a_yellow,
        "h_red": h_red, "a_red": a_red,
        "p_o15": p_o15, "p_u45": p_u45,
    }
    return p_h, p_d, p_a, h_xg, a_xg, p_o25, p_btts, p_u35, score, details


def get_form_html(form_str) -> str:
    if not form_str or str(form_str) == "?????":
        return ""
    dots = "".join(
        f"<span class='dot dot-{c.lower()}'></span>"
        for c in str(form_str)
    )
    return f"<div class='form-dots'>{dots}</div>"


# ---------------------------------------------------------------------------
# RENDER MATCH CARD
# ---------------------------------------------------------------------------

def render_match_card(m: dict, pre_calc=None):
    if pre_calc:
        p_h, p_d, p_a, h_xg, a_xg, p_o25, p_btts, p_u35, score, details = pre_calc
    else:
        p_h, p_d, p_a, h_xg, a_xg, p_o25, p_btts, p_u35, score, details = get_match_math(m)

    match_key = f"{m['home']} vs {m['away']}"
    pp = data["pro_preds"].get(match_key, {})
    has_true_xg   = pp.get("h_true_xg") or pp.get("a_true_xg")
    emerald_thresh = 85.0 if has_true_xg else 90.0

    core_probs = {
        "Home Win": p_h, "Draw": p_d, "Away Win": p_a,
        "Over 2.5": p_o25, "Under 3.5": p_u35, "BTTS": p_btts,
        "Over 1.5": details.get("p_o15", 0), "Under 4.5": details.get("p_u45", 0),
    }
    best_pick_name = max(core_probs, key=core_probs.get)
    top_p = core_probs[best_pick_name]

    safe_probs = {
        f"{m['home']} +1.5": details.get("h_plus_1_5", 0),
        f"{m['away']} +1.5": details.get("a_plus_1_5", 0),
        f"{m['home']} -1.5": details.get("h_minus_1_5", 0),
        f"{m['away']} -1.5": details.get("a_minus_1_5", 0),
    }
    best_safe_name = max(safe_probs, key=safe_probs.get)
    safe_p = safe_probs[best_safe_name]

    if safe_p > top_p and top_p < emerald_thresh:
        if safe_p >= emerald_thresh: top_p = emerald_thresh - 0.1; best_pick_name = best_safe_name
        elif safe_p >= 65.0:         top_p = safe_p;               best_pick_name = best_safe_name

    if top_p >= emerald_thresh:   t_lab, t_cls = "🟢 EMERALD TIER",  "tier-emerald"
    elif top_p >= 75.0:           t_lab, t_cls = "💎 DIAMOND+ TIER", "tier-diamond-plus"
    elif top_p >= 65.0:           t_lab, t_cls = "💎 DIAMOND TIER",  "tier-diamond"
    else:                         t_lab, t_cls = "🥇 GOLD TIER",      "tier-gold"

    insight = data["insights"].get(
        str(m.get("id")), {"injuries": [], "odds": {}, "api_pred": {}, "suspensions": []}
    )
    ts_h_data = data["top_scorers"].get(str(m.get("home_id")), [])
    ts_a_data = data["top_scorers"].get(str(m.get("away_id")), [])
    if isinstance(ts_h_data, str):
        ts_h_data = [{"name": ts_h_data.split("(")[0].strip(), "role": "Key Player"}]
    if isinstance(ts_a_data, str):
        ts_a_data = [{"name": ts_a_data.split("(")[0].strip(), "role": "Key Player"}]

    top_scorer_h = ", ".join(
        f"{x['name']} ({x.get('role', x.get('goals', 'Key Player'))})" for x in ts_h_data[:2]
    ) if ts_h_data else ""
    top_scorer_a = ", ".join(
        f"{x['name']} ({x.get('role', x.get('goals', 'Key Player'))})" for x in ts_a_data[:2]
    ) if ts_a_data else ""

    # Value tag
    value_tag = ""
    if insight.get("odds"):
        odds_map = insight["odds"]
        win_pick = "home" if p_h > p_a else "away"
        win_prob = p_h if p_h > p_a else p_a
        win_odd  = odds_map.get(win_pick.capitalize(), odds_map.get(win_pick, 0))
        if win_odd > 0 and (win_prob - (1/win_odd)*100) > 5.0:
            value_tag = (
                f"<span class='insight-badge insight-val'>"
                f"💰 VALUE: {win_pick.upper()} WIN @ {win_odd}</span>"
            )

    # Consensus tag
    api_advice = insight.get("api_pred", {}).get("advice", "")
    supports = []
    if p_o25 > 70 and ("Over" in api_advice or "+2.5" in api_advice): supports.append("Over 2.5")
    if p_btts > 70 and ("Both teams" in api_advice or "BTTS" in api_advice):  supports.append("BTTS")
    if p_h > 70 and "Home" in api_advice: supports.append("Home Win")
    if p_a > 70 and "Away" in api_advice: supports.append("Away Win")

    if len(supports) >= 2 or (supports and p_h > 85):
        consensus_tag = (
            f"<span class='insight-badge' style='background:#ff9f43; color:#1e1e1e; "
            f"font-weight:800; border:1px solid #e17055;'>"
            f"🤖 SUPER CONSENSUS: {' & '.join(supports)}</span>"
        )
    elif supports:
        consensus_tag = (
            f"<span class='insight-badge insight-consensus'>"
            f"🤖 Supports: {' & '.join(supports)}</span>"
        )
    else:
        consensus_tag = ""

    # Intel badges
    intel_tags = ""
    if pp.get("h_susp"):       intel_tags += "<span class='insight-badge' style='background:#e74c3c; color:#fff;'>🛑 SUSPENDED: Home Player</span> "
    if pp.get("a_susp"):       intel_tags += "<span class='insight-badge' style='background:#e74c3c; color:#fff;'>🛑 SUSPENDED: Away Player</span> "
    if pp.get("h_fatigue"):    intel_tags += "<span class='insight-badge' style='background:#7f8c8d; color:#fff;'>🪫 FATIGUED: Home</span> "
    if pp.get("a_fatigue"):    intel_tags += "<span class='insight-badge' style='background:#7f8c8d; color:#fff;'>🪫 FATIGUED: Away</span> "
    if has_true_xg:            intel_tags += "<span class='insight-badge' style='background:#27ae60; color:#fff;'>⚖️ TRUE xG EDGE</span> "
    if insight.get("sharp_action"):
        intel_tags += (
            f"<span class='insight-badge' style='background:#d35400; color:#fff;'>"
            f"🦈 SHARP MONEY: {insight['sharp_action']}</span> "
        )
    if pp.get("h_lookahead"):  intel_tags += "<span class='insight-badge' style='background:#f39c12; color:#fff;'>⚠️ TRAP: Home Lookahead</span> "
    if pp.get("a_lookahead"):  intel_tags += "<span class='insight-badge' style='background:#f39c12; color:#fff;'>⚠️ TRAP: Away Lookahead</span> "
    if pp.get("bogey_team"):
        intel_tags += (
            f"<span class='insight-badge' style='background:#c0392b; color:#fff;'>"
            f"👻 HEX: {pp['bogey_team']} Bogey Team</span> "
        )
    if pp.get("strict_ref"):   intel_tags += "<span class='insight-badge' style='background:#c0392b; color:#fff;'>🟥 VOLATILITY: Strict Ref</span> "
    if (pp.get("cs_h",0) > 0.45 and pp.get("fts_a",0) > 0.40) or \
       (pp.get("cs_a",0) > 0.45 and pp.get("fts_h",0) > 0.40):
        intel_tags += "<span class='insight-badge' style='background:#2c3e50; color:#fff;'>🚌 PARK THE BUS: BTTS-No Alert</span> "
    if pp.get("h_bounce"):     intel_tags += "<span class='insight-badge' style='background:#2980b9; color:#fff;'>👔 NEW MANAGER: Home</span> "
    if pp.get("a_bounce"):     intel_tags += "<span class='insight-badge' style='background:#2980b9; color:#fff;'>👔 NEW MANAGER: Away</span> "
    if pp.get("h_false_form"): intel_tags += "<span class='insight-badge' style='background:#16a085; color:#fff;'>🚨 FALSE FORM: Home</span> "
    if pp.get("a_false_form"): intel_tags += "<span class='insight-badge' style='background:#16a085; color:#fff;'>🚨 FALSE FORM: Away</span> "
    if pp.get("h_exodus"):     intel_tags += "<span class='insight-badge' style='background:#8e44ad; color:#fff;'>📉 EXODUS: Home</span> "
    if pp.get("a_exodus"):     intel_tags += "<span class='insight-badge' style='background:#8e44ad; color:#fff;'>📉 EXODUS: Away</span> "
    if pp.get("h_fast", 0) > 28 and pp.get("a_leak", 0) > 28:
        intel_tags += "<span class='insight-badge' style='background:#8e44ad; color:#fff;'>⏱️ TACTICAL: Home Fast Start</span> "
    elif pp.get("a_fast", 0) > 28 and pp.get("h_leak", 0) > 28:
        intel_tags += "<span class='insight-badge' style='background:#8e44ad; color:#fff;'>⏱️ TACTICAL: Away Fast Start</span> "
    # Market edge badges — only shown when Pinnacle data exists (edge != 0)
    _edge_h = float(pp.get("market_edge_h", 0.0))
    _edge_a = float(pp.get("market_edge_a", 0.0))
    if _edge_h >= 5.0:
        intel_tags += (f"<span class='insight-badge insight-val'>"
                       f"📈 EDGE +{_edge_h:.1f}pp: {m['home']} vs market</span> ")
    elif _edge_h <= -5.0:
        intel_tags += (f"<span class='insight-badge' style='background:#c0392b; color:#fff;'>"
                       f"📉 FADE {abs(_edge_h):.1f}pp: Market fancies {m['home']}</span> ")
    if _edge_a >= 5.0:
        intel_tags += (f"<span class='insight-badge insight-val'>"
                       f"📈 EDGE +{_edge_a:.1f}pp: {m['away']} vs market</span> ")
    elif _edge_a <= -5.0:
        intel_tags += (f"<span class='insight-badge' style='background:#c0392b; color:#fff;'>"
                       f"📉 FADE {abs(_edge_a):.1f}pp: Market fancies {m['away']}</span> ")

    strat_pts = []
    if p_o25 > 70:                                            strat_pts.append(f"O2.5 ({p_o25:.0f}%)")
    if p_btts > 70:                                           strat_pts.append(f"BTTS ({p_btts:.0f}%)")
    if p_h > 70:                                              strat_pts.append(f"Home Win ({p_h:.0f}%)")
    if p_u35 > 70:                                            strat_pts.append(f"U3.5 ({p_u35:.0f}%)")
    if details.get("p_o15", 0) > 80:                          strat_pts.append(f"O1.5 ({details.get('p_o15',0):.0f}%)")
    if details.get("p_u45", 0) > 80:                          strat_pts.append(f"U4.5 ({details.get('p_u45',0):.0f}%)")
    if details.get("h_minus_1_5", 0) > 45:                   strat_pts.append(f"🔥 {m['home']} -1.5 ({details.get('h_minus_1_5',0):.0f}%)")
    if details.get("a_minus_1_5", 0) > 45:                   strat_pts.append(f"🔥 {m['away']} -1.5 ({details.get('a_minus_1_5',0):.0f}%)")
    if details.get("a_plus_1_5", 0) > 80:                    strat_pts.append(f"🛡️ {m['away']} +1.5 ({details.get('a_plus_1_5',0):.0f}%)")
    if details.get("h_plus_1_5", 0) > 80:                    strat_pts.append(f"🛡️ {m['home']} +1.5 ({details.get('h_plus_1_5',0):.0f}%)")
    strat_text = "💡 Supports: " + " & ".join(strat_pts) if strat_pts else "⚖️ Market: Balanced"

    def_logo = "https://media.api-sports.io/football/teams/0.png"
    l_logo   = m.get("league_logo") or "https://media.api-sports.io/football/leagues/39.png"

    st.markdown('<div class="match-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 2.2, 2.5])

    with c1:
        st.markdown(
            f"<span class='tier-badge {t_cls}'>{t_lab}</span> "
            f"{intel_tags} {value_tag} {consensus_tag}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='league-header'><img src='{l_logo}' class='league-logo'> "
            f"{m['league']} • {format_time(m['date'])} CET</div>",
            unsafe_allow_html=True,
        )
        h_form = get_form_html(data["team_forms"].get(str(m.get("home_id"))))
        a_form = get_form_html(data["team_forms"].get(str(m.get("away_id"))))
        st.markdown(
            f"<div class='team-row'><img src='{m.get('home_logo') or def_logo}' class='team-logo'>"
            f"<span class='team-name'>{m['home']}</span>{h_form}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='team-row'><img src='{m.get('away_logo') or def_logo}' class='team-logo'>"
            f"<span class='team-name'>{m['away']}</span>{a_form}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='strategy-headline'>{strat_text}</div>", unsafe_allow_html=True)

        with st.expander("📊 Smart Analytics & Team News", expanded=False):
            sa_c1, sa_c2 = st.columns(2)
            with sa_c1:
                st.markdown("<div class='metric-title' style='text-align:left'>🎯 Key Players</div>", unsafe_allow_html=True)
                if top_scorer_h:
                    st.markdown(
                        f"<div style='background:#1e272e; padding:10px; border-radius:6px; "
                        f"margin-bottom:5px; border-left:3px solid #00b894;'>"
                        f"<b style='color:#fff; font-size:11px;'>{m['home']}</b><br>"
                        f"<span style='color:#ccc; font-size:12px;'>{top_scorer_h}</span></div>",
                        unsafe_allow_html=True,
                    )
                if top_scorer_a:
                    st.markdown(
                        f"<div style='background:#1e272e; padding:10px; border-radius:6px; "
                        f"margin-bottom:5px; border-left:3px solid #d63031;'>"
                        f"<b style='color:#fff; font-size:11px;'>{m['away']}</b><br>"
                        f"<span style='color:#ccc; font-size:12px;'>{top_scorer_a}</span></div>",
                        unsafe_allow_html=True,
                    )
                if not (top_scorer_h or top_scorer_a):
                    st.caption("No key player data available.")
            with sa_c2:
                if details.get("h_injury_alerts") or details.get("a_injury_alerts") or insight.get("suspensions"):
                    st.markdown("<div class='metric-title' style='text-align:left'>🚨 MISSING PLAYERS</div>", unsafe_allow_html=True)
                    for alert in details.get("h_injury_alerts", []):
                        cls = "regular-missing-badge" if "📉" in alert else "key-missing-badge"
                        st.markdown(f"<div class='{cls}'>{alert}</div>", unsafe_allow_html=True)
                    for alert in details.get("a_injury_alerts", []):
                        cls = "regular-missing-badge" if "📉" in alert else "key-missing-badge"
                        st.markdown(f"<div class='{cls}'>{alert}</div>", unsafe_allow_html=True)
                    for s in insight.get("suspensions", []):
                        st.markdown(f"<div class='key-missing-badge'>🛑 Suspended: {s['player_name']}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-title' style='text-align:left; margin-top:8px;'>🚑 Full Injury Report</div>", unsafe_allow_html=True)
                if insight.get("injuries"):
                    for inj in insight["injuries"]:
                        if isinstance(inj, str):
                            st.markdown(f"<div style='background:#2d3436; padding:8px; border-radius:4px; margin-bottom:4px; font-size:11px; border:1px solid #444; color:#ddd;'>🩹 {inj}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"<div style='background:#2d3436; padding:8px; border-radius:4px; margin-bottom:4px; font-size:11px; border:1px solid #444; color:#ddd;'>"
                                f"🩹 {inj.get('team_name','')}: {inj.get('player_name','')} ({inj.get('reason','')})</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("No major injuries reported.")

    with c2:
        total_xg = h_xg + a_xg
        h_p = (h_xg / total_xg * 100) if total_xg > 0 else 50
        st.markdown(
            f"<div class='xg-container-new'>"
            f"<div class='xg-title-new'>Relative xG Model</div>"
            f"<div class='xg-bar-wrapper'>"
            f"<div class='xg-bar-home' style='width:{h_p:.1f}%'><span class='xg-val-text'>{h_xg:.2f}</span></div>"
            f"<div class='xg-bar-away' style='width:{100-h_p:.1f}%'><span class='xg-val-text'>{a_xg:.2f}</span></div>"
            f"</div>"
            f"<div class='xg-total-text' style='color:#888; font-size:11px; margin-top:6px;'>Total: {total_xg:.2f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with c3:
        prob_title = "Win Probability (V3 Meta-Ensemble)" if ML_ENABLED else "Win Probability (Math Engine)"
        bh = clamp_prob(p_h); bd = clamp_prob(p_d); ba = clamp_prob(p_a)
        st.markdown(
            f"<div class='prob-header'>{prob_title}</div>"
            f"<div class='prog-label'><span>Home</span><span>{bh:.0f}%</span></div>"
            f"<div class='custom-bar-bg'><div class='custom-bar-fill' style='width:{bh:.1f}%; background-color:#00b894;'></div></div>"
            f"<div class='prog-label'><span>Draw</span><span>{bd:.0f}%</span></div>"
            f"<div class='custom-bar-bg'><div class='custom-bar-fill' style='width:{bd:.1f}%; background-color:#636e72;'></div></div>"
            f"<div class='prog-label'><span>Away</span><span>{ba:.0f}%</span></div>"
            f"<div class='custom-bar-bg'><div class='custom-bar-fill' style='width:{ba:.1f}%; background-color:#d63031;'></div></div>"
            f"<div class='stat-grid'>"
            f"<div class='stat-box'><div class='stat-label'>O2.5</div><div class='stat-value'>{p_o25:.0f}%</div></div>"
            f"<div class='stat-box'><div class='stat-label'>BTTS</div><div class='stat-value'>{p_btts:.0f}%</div></div>"
            f"<div class='stat-box'><div class='stat-label'>U3.5</div><div class='stat-value'>{p_u35:.0f}%</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander("📈 Advanced Markets (Handicap, Corners, Cards)"):
        am1, am2, am3 = st.columns(3)
        with am1:
            st.markdown(
                f"<div class='market-box'><div class='market-title'>Asian Handicap (1.5)</div>"
                f"<div class='market-row'><span>{m['home']} -1.5</span>"
                f"<span class='market-val'>{details.get('h_minus_1_5',0):.1f}% "
                f"<span class='odds-fair'>@{100/max(0.1, details.get('h_minus_1_5', 0.1)):.2f}</span></span></div>"
                f"<div class='market-row'><span>{m['away']} +1.5</span>"
                f"<span class='market-val'>{details.get('a_plus_1_5',0):.1f}% "
                f"<span class='odds-fair'>@{100/max(0.1, details.get('a_plus_1_5', 0.1)):.2f}</span></span></div>"
                f"<div style='border-top:1px dashed #444; margin:5px 0;'></div>"
                f"<div class='market-row'><span>{m['away']} -1.5</span>"
                f"<span class='market-val'>{details.get('a_minus_1_5',0):.1f}% "
                f"<span class='odds-fair'>@{100/max(0.1, details.get('a_minus_1_5', 0.1)):.2f}</span></span></div>"
                f"<div class='market-row'><span>{m['home']} +1.5</span>"
                f"<span class='market-val'>{details.get('h_plus_1_5',0):.1f}% "
                f"<span class='odds-fair'>@{100/max(0.1, details.get('h_plus_1_5', 0.1)):.2f}</span></span></div>"
                f"</div>", unsafe_allow_html=True,
            )
        with am2:
            total_cor = details.get("h_corners", 0) + details.get("a_corners", 0)
            st.markdown(
                f"<div class='market-box'><div class='market-title'>Alternative Goals</div>"
                f"<div class='market-row'><span>Over 1.5</span>"
                f"<span class='market-val'>{details.get('p_o15',0):.1f}% "
                f"<span class='odds-fair'>@{100/max(0.1, details.get('p_o15', 0.1)):.2f}</span></span></div>"
                f"<div class='market-row'><span>Under 4.5</span>"
                f"<span class='market-val'>{details.get('p_u45',0):.1f}% "
                f"<span class='odds-fair'>@{100/max(0.1, details.get('p_u45', 0.1)):.2f}</span></span></div>"
                f"<div style='border-top:1px dashed #444; margin:5px 0;'></div>"
                f"<div class='market-title' style='margin-top:8px;'>Corners (Proj: {total_cor:.1f})</div>"
                f"<div class='market-row'><span>{m['home']}</span><span class='market-val'>{details.get('h_corners',0):.1f} Exp.</span></div>"
                f"<div class='market-row'><span>{m['away']}</span><span class='market-val'>{details.get('a_corners',0):.1f} Exp.</span></div>"
                f"</div>", unsafe_allow_html=True,
            )
        with am3:
            st.markdown(
                f"<div class='market-box'><div class='market-title'>Discipline (Season Avg)</div>"
                f"<div class='market-row'><span>{m['home']} Cards</span>"
                f"<span class='market-val' style='color:#feca57'>{details.get('h_yellow',1.8):.1f}</span> / "
                f"<span class='market-val' style='color:#ff7675'>{details.get('h_red',0.1):.2f}</span></div>"
                f"<div class='market-row'><span>{m['away']} Cards</span>"
                f"<span class='market-val' style='color:#feca57'>{details.get('a_yellow',1.8):.1f}</span> / "
                f"<span class='market-val' style='color:#ff7675'>{details.get('a_red',0.1):.2f}</span></div>"
                f"</div>", unsafe_allow_html=True,
            )

    if "DIAMOND" in t_lab or "EMERALD" in t_lab:
        with st.expander("💰 Professional Strategy (Kelly)"):
            d1, d2 = st.columns([1.3, 1])
            with d1:
                hist = data["h2h"].get(str(f"{m.get('home_id')}-{m.get('away_id')}"), [])
                h2h_html = (
                    "<div class='h2h-container'>"
                    "<div class='h2h-header-row'><span>Date</span><span>Match</span><span>Result</span></div>"
                )
                if hist:
                    for h in hist:
                        h_logo_url = h.get("home_logo", def_logo)
                        a_logo_url = h.get("away_logo", def_logo)
                        win_color = (
                            "<span class='badge-win'>HOME</span>" if h["winner"]=="Home"
                            else "<span class='badge-loss'>AWAY</span>" if h["winner"]=="Away"
                            else "<span class='badge-draw'>DRAW</span>"
                        )
                        h2h_html += (
                            f"<div class='h2h-row'>"
                            f"<span style='color:#ccc; width:75px;'>{h['date']}</span>"
                            f"<div style='flex:1; display:flex; align-items:center; justify-content:center; color:white; font-weight:700;'>"
                            f"<img src='{h_logo_url}' class='h2h-logo'> {h['score']} <img src='{a_logo_url}' class='h2h-logo'>"
                            f"</div><span>{win_color}</span></div>"
                        )
                else:
                    h2h_html += "<div style='padding:15px; color:#666; text-align:center;'>No recent history found.</div>"
                st.markdown(h2h_html + "</div>", unsafe_allow_html=True)
            with d2:
                if "+1.5" in best_pick_name or "Under 3.5" in best_pick_name or "Under 4.5" in best_pick_name or "Over 1.5" in best_pick_name:
                    est_odds = 1.35
                elif "-1.5" in best_pick_name:    est_odds = 2.20
                elif best_pick_name in ("Home Win","Away Win"): est_odds = 2.10
                elif best_pick_name == "Draw":    est_odds = 3.30
                else:                             est_odds = 2.80

                b = est_odds - 1; p = top_p / 100.0; q = 1 - p
                kelly_f   = (b*p - q) / b if b > 0 else 0
                kelly_pct = max(0, kelly_f * 0.25 * 100)

                st.markdown(
                    f"<div class='value-card'>"
                    f"<div class='value-title'>🤖 Top AI Pick</div>"
                    f"<div style='font-weight:900; color:white; font-size:20px; margin-bottom:10px;'>{best_pick_name}</div>"
                    f"<div class='value-title'>Predicted Score</div>"
                    f"<div class='value-score' style='font-size:24px; margin-bottom:5px;'>{score}</div>"
                    f"<hr class='kelly-divider'>"
                    f"<div class='value-title' style='color:#55efc4;'>💰 Rec. Stake (%)</div>"
                    f"<div class='value-stake'>{kelly_pct:.1f}%</div>"
                    f"<div class='bankroll-sub'>of bankroll (Based on {est_odds:.2f} odds)</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SHARED: best-pick helper (DRY — used in multiple pages)
# ---------------------------------------------------------------------------

def get_best_pick(stats, m: dict, core_only: bool = False):
    p_h, p_d, p_a, h_xg, a_xg, p_o25, p_btts, p_u35, score, det = stats
    pp = data["pro_preds"].get(f"{m['home']} vs {m['away']}", {})
    emerald_thresh = 85.0 if (pp.get("h_true_xg") or pp.get("a_true_xg")) else 90.0

    core = {
        "Home Win": p_h, "Draw": p_d, "Away Win": p_a,
        "Over 2.5": p_o25, "Under 3.5": p_u35, "BTTS": p_btts,
        "Over 1.5": det.get("p_o15", 0), "Under 4.5": det.get("p_u45", 0),
    }
    best_pick = max(core, key=core.get); top_p = core[best_pick]

    if not core_only:
        safe = {
            f"{m['home']} +1.5": det.get("h_plus_1_5", 0),
            f"{m['away']} +1.5": det.get("a_plus_1_5", 0),
            f"{m['home']} -1.5": det.get("h_minus_1_5", 0),
            f"{m['away']} -1.5": det.get("a_minus_1_5", 0),
        }
        best_safe = max(safe, key=safe.get); safe_p = safe[best_safe]

        if safe_p > top_p and top_p < emerald_thresh:
            if safe_p >= emerald_thresh: top_p, best_pick = emerald_thresh - 0.1, best_safe
            elif safe_p >= 65.0:         top_p, best_pick = safe_p, best_safe

    return best_pick, top_p, emerald_thresh


# ===========================================================================
# PAGE ROUTING
# ===========================================================================

if page == "🔮 League Predictions":
    data["upcoming"].sort(key=lambda x: x["date"])
    st.title("🔮 League Predictions")
    league_matches = [m for m in data["upcoming"] if m["league"] not in CUP_NAMES]
    all_leagues    = sorted({m["league"] for m in league_matches})
    default_sel    = [l for l in DEFAULT_LEAGUES if l in all_leagues] or all_leagues
    sel_leagues    = st.multiselect("Filter League", all_leagues, default=default_sel)
    search         = st.text_input("🔍 Search Team")
    count = 0
    for m in league_matches:
        if not sel_leagues or m["league"] in sel_leagues:
            if not search or search.lower() in m["home"].lower() or search.lower() in m["away"].lower():
                render_match_card(m); count += 1
    if count == 0:
        st.info("No league matches found matching criteria.")

elif page == "🏆 National Cup Predictions":
    st.title("🏆 National Cup Predictions")
    cup_matches = sorted(
        [m for m in data["upcoming"] if m["league"] in CUP_NAMES], key=lambda x: x["date"]
    )
    if not cup_matches:
        st.info("No National Cup matches scheduled.")
    else:
        for m in cup_matches: render_match_card(m)

elif page == "🟢 Emerald/Diamond Picks":
    st.title("🟢 Emerald & Diamond Tier Picks")
    st.caption("Sorted: Matchday > Tier Rank > Time")
    display_items = []
    for m in data["upcoming"]:
        stats = get_match_math(m)
        best_pick, top_p, emerald_thresh = get_best_pick(stats, m)
        if top_p >= 65.0:
            tier_rank = 3 if top_p >= emerald_thresh else 2 if top_p >= 75.0 else 1
            display_items.append({
                "match": m, "stats": stats,
                "date_day": m["date"][:10], "rank": tier_rank, "full_date": m["date"],
            })
    display_items.sort(key=lambda x: (x["date_day"], -x["rank"], x["full_date"]))
    if display_items:
        for item in display_items:
            render_match_card(item["match"], pre_calc=item["stats"])
    else:
        st.info("No Diamond/Emerald picks today.")

elif page == "🟢 Emerald/Diamond Results":
    st.title("🟢 Emerald & Diamond Results")
    st.caption("All recent matches — Diamond/Emerald criteria auto-graded.")
    if data["recent"]:
        recent_sorted = sorted(data["recent"], key=lambda x: x["date"], reverse=True)
        dates = sorted({m["date"][:10] for m in recent_sorted}, reverse=True)
        all_hits = []
        for d in dates:
            day_matches  = [m for m in recent_sorted if m["date"].startswith(d)]
            diamond_hits = []
            for m in day_matches:
                try:
                    stats = _cached_match_math(m)
                    ai_pick, top_p, emerald_thresh = get_best_pick(stats, m, core_only=True)
                    if top_p < 65.0: continue
                    score_str = m.get("score", "")
                    if "-" not in str(score_str): continue
                    h_g, a_g  = map(int, score_str.split("-")); total_g = h_g + a_g
                    won = False
                    if ai_pick == "Home Win"  and h_g > a_g:            won = True
                    elif ai_pick == "Draw"    and h_g == a_g:           won = True
                    elif ai_pick == "Away Win" and a_g > h_g:           won = True
                    elif ai_pick == "Over 2.5" and total_g > 2:         won = True
                    elif ai_pick == "BTTS"    and h_g>0 and a_g>0:      won = True
                    elif ai_pick == "Under 3.5" and total_g < 4:        won = True
                    elif ai_pick == "Over 1.5" and total_g > 1:         won = True
                    elif ai_pick == "Under 4.5" and total_g < 5:        won = True
                    tier = (
                        "🟢 EMERALD"  if top_p >= emerald_thresh
                        else "💎 DIAMOND+" if top_p >= 75
                        else "💎 DIAMOND"  if top_p >= 65
                        else "🥇 GOLD"
                    )
                    diamond_hits.append({
                        "Match": f"{m['home']} vs {m['away']}", "Score": score_str,
                        "Prediction": f"{ai_pick} ({top_p:.0f}%)",
                        "Result": "✅ WON" if won else "❌ LOST", "Tier": tier,
                    })
                except Exception as e:
                    log.debug("Results grading error: %s", e)
            if diamond_hits:
                st.markdown(f"### 📅 {d}")
                st.dataframe(pd.DataFrame(diamond_hits), use_container_width=True, hide_index=True)
                st.divider()
                all_hits.extend(diamond_hits)
        if all_hits:
            csv_bytes = pd.DataFrame(all_hits).to_csv(index=False).encode()
            st.download_button("⬇️ Export All Results as CSV", data=csv_bytes,
                               file_name="results_export.csv", mime="text/csv")
    else:
        st.info("No recent results data found.")

elif page == "🏗️ Parlay Builder":
    st.title("🏗️ Smart Parlay Builder")
    if "parlay" not in st.session_state:
        st.session_state.parlay = []

    st.markdown("### 🤖 One-Click AI Generators")
    c_gen1, c_gen2 = st.columns(2)
    for col, min_prob, label in [
        (c_gen1, 75.0, "💎 Diamond+ Express (75%+)"),
        (c_gen2, 65.0, "💰 Safe Value Builder (65%+)"),
    ]:
        with col:
            st.info(f"**{label}**")
            if st.button(f"Generate {label}"):
                st.session_state.parlay = []
                count = 0
                for m in data["upcoming"]:
                    stats = _cached_match_math(m)
                    best_pick, best_prob, _ = get_best_pick(stats, m)
                    if best_prob >= min_prob:
                        st.session_state.parlay.append({
                            "match": f"{m['home']} vs {m['away']}",
                            "market": best_pick, "prob": best_prob,
                        })
                        count += 1
                if count > 0: st.success(f"Added {count} picks!")
                else:         st.warning("No qualifying picks found today.")

    st.divider()
    with st.container():
        st.markdown("<div class='parlay-card'><div class='parlay-header'>1. Add Selection Manually</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.5, 3, 2, 1])
        with c1:
            p_date     = st.date_input("Filter Date", value=datetime.today(), key="parlay_date_picker")
            p_date_str = p_date.strftime("%Y-%m-%d")
        day_m = [m for m in data["upcoming"] if m["date"].startswith(p_date_str)]
        with c2:
            sel_m_name = (
                st.selectbox("Select Match", [f"{m['home']} vs {m['away']}" for m in day_m])
                if day_m else None
            )
        sel_match_data = None
        if day_m and sel_m_name:
            sel_match_data = next(
                (x for x in day_m if f"{x['home']} vs {x['away']}" == sel_m_name), None
            )
        with c3:
            if sel_match_data:
                dynamic_markets = [
                    "Home Win", "Away Win", "Over 2.5", "Under 3.5", "BTTS", "Over 1.5", "Under 4.5",
                    f"{sel_match_data['home']} -1.5", f"{sel_match_data['home']} +1.5",
                    f"{sel_match_data['away']} -1.5", f"{sel_match_data['away']} +1.5",
                ]
                sel_market = st.selectbox("Market", dynamic_markets, key="parlay_market_select")
            else:
                sel_market = st.selectbox(
                    "Market",
                    ["Home Win", "Away Win", "Over 2.5", "Under 3.5", "BTTS", "Over 1.5", "Under 4.5"],
                    key="parlay_market_select",
                )
        with c4:
            st.write("")
            if sel_match_data and st.button("➕ Add"):
                st.session_state.parlay.append({"match": sel_m_name, "market": sel_market, "prob": 0.0})
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.parlay:
        st.markdown("### 📋 Current Parlay")
        total_prob = 1.0
        for i, leg in enumerate(st.session_state.parlay):
            c1, c2 = st.columns([5, 1])
            with c1: st.markdown(f"**{leg['match']}** — {leg['market']} ({leg['prob']:.1f}%)")
            with c2:
                if st.button("❌", key=f"rem_{i}"):
                    st.session_state.parlay.pop(i); st.rerun()
            if leg["prob"] > 0: total_prob *= leg["prob"] / 100.0
        combo_odds = 1 / total_prob if total_prob > 0 else 0
        st.metric("Combined Probability", f"{total_prob*100:.2f}%")
        st.metric("Est. Combo Odds",       f"{combo_odds:.2f}")
        if st.button("🗑️ Clear All"):
            st.session_state.parlay = []; st.rerun()

elif page == "📊 Tables":
    st.title("📊 League Tables")
    if not data["standings"]:
        st.info("No standings data. Run train_master.py.")
    else:
        league_list = list(data["standings"].keys())
        sel_league  = st.selectbox("Select League", league_list)
        try:
            league_node   = data["standings"][sel_league]
            standings_list = league_node.get("standings", [])
            if not standings_list:
                st.info("No standings for this league.")
            else:
                for idx, table_group in enumerate(standings_list):
                    if not table_group: continue
                    if len(standings_list) > 1:
                        st.markdown(f"#### {table_group[0].get('group') or f'Group {idx+1}'}")
                    st.markdown(
                        "<div class='div-table-header'>"
                        "<div class='col-rank'>#</div>"
                        "<div class='col-team'>TEAM</div>"
                        "<div class='col-stats'>PL</div>"
                        "<div class='col-stats'>W-D-L</div>"
                        "<div class='col-gd'>GD</div>"
                        "<div class='col-pts'>PTS</div>"
                        "<div class='col-form'>FORM</div>"
                        "</div>", unsafe_allow_html=True,
                    )
                    rows_html = ""
                    for t in table_group:
                        rank      = int(t.get("rank", 0) or 0)
                        n         = len(table_group)
                        rank_cls  = ("rk-ucl" if rank<=4 else "rk-uel" if rank<=6
                                     else "rk-rel" if rank>=n-2 else "rk-mid")
                        form_html = get_form_html(str(t.get("form", "?????")))
                        all_s     = t.get("all", {})
                        gd        = t.get("goalsDiff", 0)
                        gd_col    = "#00b894" if gd>0 else "#d63031" if gd<0 else "#888"
                        logo      = t.get("team", {}).get("logo", "")
                        name      = t.get("team", {}).get("name", "")
                        rows_html += (
                            f"<div class='div-table-row'>"
                            f"<div class='col-rank'><div class='rank-badge {rank_cls}'>{rank}</div></div>"
                            f"<div class='col-team'><img src='{logo}' class='table-logo'>"
                            f"<span class='team-txt'>{name}</span></div>"
                            f"<div class='col-stats'>{all_s.get('played',0)}</div>"
                            f"<div class='col-stats'>{all_s.get('win',0)}-{all_s.get('draw',0)}-{all_s.get('lose',0)}</div>"
                            f"<div class='col-gd' style='color:{gd_col}'>{'+' if gd>0 else ''}{gd}</div>"
                            f"<div class='col-pts'>{t.get('points',0)}</div>"
                            f"<div class='col-form'>{form_html}</div>"
                            f"</div>"
                        )
                    st.markdown(rows_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Table render error: {e}")

elif page == "📈 Profit Tracker":
    st.title("📈 Smart Profit Tracker")
    df = pd.DataFrame(data["bet_log"])
    if not df.empty:
        total_profit   = df["profit"].sum()
        total_invested = df["stake"].sum()
        roi            = (total_profit / total_invested * 100) if total_invested > 0 else 0.0
        wins           = len(df[df["result"] == "Won"])
        total_resolved = len(df[df["result"].isin(["Won","Lost"])])
        win_rate       = (wins / total_resolved * 100) if total_resolved > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Net Profit",  f"${total_profit:,.2f}", delta=round(total_profit, 2))
        c2.metric("ROI",         f"{roi:.1f}%",           delta=round(roi, 2))
        c3.metric("Win Rate",    f"{win_rate:.1f}%")

        results_map = {
            f"{r['home']} vs {r['away']}": r.get("score","")
            for r in data["recent"] if isinstance(r, dict) and "home" in r and "away" in r
        }
        if st.button("🔄 Auto-Grade Bets"):
            cl = load_bet_log_db()
            updated_log = [calculate_bet_result(bet, results_map) for bet in cl]
            save_bet_log_full(updated_log)
            load_data.clear(); st.success("Graded!"); st.rerun()

    st.markdown("### 💾 Backup")
    st.info("Bets are stored in SQLite. Download a CSV backup periodically.")
    if not df.empty:
        csv_data = df.to_csv(index=False)
        st.download_button("⬇️ Download CSV", data=csv_data, file_name="bet_log_backup.csv", mime="text/csv")

    st.divider()
    st.markdown("### ➕ Log New Bet")
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1: bet_date   = st.date_input("Date", value=datetime.today(), key="bl_date")
    with b2:
        day_matches_bl  = [m for m in data["upcoming"] if m["date"].startswith(str(bet_date))]
        match_options   = [f"{m['home']} vs {m['away']}" for m in day_matches_bl] or ["Custom Match"]
        bet_match       = st.selectbox("Match", match_options, key="bl_match")
    with b3: bet_pick  = st.text_input("Pick",    value="Home Win", key="bl_pick")
    with b4: bet_odds  = st.number_input("Odds",  value=2.00, min_value=1.01, step=0.01, key="bl_odds")
    with b5: bet_stake = st.number_input("Stake", value=10.0, min_value=0.1, step=0.5,  key="bl_stake")
    if st.button("💾 Log Bet"):
        conn = get_db_connection(); c = conn.cursor()
        c.execute(
            "INSERT INTO bet_log (date, match, pick, odds, stake, result, profit) VALUES (?,?,?,?,?,?,?)",
            (str(bet_date), bet_match, bet_pick, bet_odds, bet_stake, "Pending", 0.0),
        )
        conn.commit(); conn.close()
        load_data.clear(); st.success("Bet logged!"); st.rerun()
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "📊 Sharp EV Dashboard":
    st.title("📊 Sharp EV Dashboard")
    ev_rows = []
    for m in data["upcoming"]:
        p_h, p_d, p_a, h_xg, a_xg, p_o25, p_btts, p_u35, score, det = _cached_match_math(m)
        insight = data["insights"].get(str(m.get("id")), {})
        if not insight.get("odds"): continue
        odds_map = insight["odds"]
        for pick, prob, side in [
            (f"{m['home']} Win", p_h,   "Home"),
            ("Draw",             p_d,   "Draw"),
            (f"{m['away']} Win", p_a,   "Away"),
            ("Over 2.5",         p_o25, "Over 2.5"),
        ]:
            raw_odds = odds_map.get(side, 0) or odds_map.get(side.lower(), 0)
            if not raw_odds: continue
            ev = calculate_ev(prob, raw_odds)
            pp_ev = data["pro_preds"].get(f"{m['home']} vs {m['away']}", {})
            if side == "Home":
                edge_pp = float(pp_ev.get("market_edge_h", 0.0))
            elif side == "Away":
                edge_pp = float(pp_ev.get("market_edge_a", 0.0))
            else:
                edge_pp = 0.0
            edge_str = (f"+{edge_pp:.1f}pp" if edge_pp > 0 else
                        f"{edge_pp:.1f}pp"   if edge_pp < 0 else "—")
            if ev > 3.0:
                ev_rows.append({
                    "Match":       f"{m['home']} vs {m['away']}",
                    "League":      m["league"],
                    "Pick":        pick,
                    "Model %":     f"{prob:.1f}%",
                    "Odds":        raw_odds,
                    "EV":          f"+{ev:.2f}%",
                    "Mkt Edge":    edge_str,
                })
    if ev_rows:
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No positive-EV picks found today (need odds data from insights).")

    match_names_ev = [f"{m['home']} vs {m['away']}" for m in data["upcoming"]]
    st.divider()
    st.subheader("🧮 Manual EV & Kelly Calculator")
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_m_ev = st.selectbox("Select Match", match_names_ev, key="ev_match")
        sel_m_ev_data = next((m for m in data["upcoming"] if f"{m['home']} vs {m['away']}" == sel_m_ev), None)
    with c2:
        user_odds_h = st.number_input("Home Win Odds", value=2.10, step=0.01, key="ev_hod")
        user_odds_a = st.number_input("Away Win Odds", value=3.50, step=0.01, key="ev_aod")
    with c3:
        bankroll = st.number_input("Bankroll ($)", value=1000.0, step=50.0, key="ev_bank")
    if sel_m_ev_data:
        p_h, p_d, p_a, h_xg, a_xg, _, _, _, _, _ = _cached_match_math(sel_m_ev_data)
        c1, c2, c3 = st.columns(3)
        with c1:
            ev_h = calculate_ev(p_h, user_odds_h); rec_stake_h = calculate_kelly(p_h, user_odds_h, bankroll)
            colour = "ev-positive" if ev_h > 0 else "ev-negative"; sign = "+" if ev_h > 0 else ""
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Home EV</div><div class='metric-value {colour}'>{sign}{ev_h:.2f}%</div></div>", unsafe_allow_html=True)
            if ev_h > 0: st.caption(f"✅ Rec. Stake: ${rec_stake_h:.2f}")
        with c2:
            ev_a = calculate_ev(p_a, user_odds_a); rec_stake_a = calculate_kelly(p_a, user_odds_a, bankroll)
            colour = "ev-positive" if ev_a > 0 else "ev-negative"; sign = "+" if ev_a > 0 else ""
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Away EV</div><div class='metric-value {colour}'>{sign}{ev_a:.2f}%</div></div>", unsafe_allow_html=True)
            if ev_a > 0: st.caption(f"✅ Rec. Stake: ${rec_stake_a:.2f}")
        with c3:
            st.markdown("#### 🤖 AI Model")
            st.progress(min(100, max(0, int(p_h))))
            st.caption(f"Home: {p_h:.1f}% | Away: {p_a:.1f}%")
            st.write(f"**xG:** {h_xg:.2f} vs {a_xg:.2f}")

elif page == "🔎 Tactical Filters":
    st.title("🔎 Strategic Market Filters")
    tab1, tab2 = st.tabs(["🔥 High-Frequency BTTS", "⏳ Late Game Volatility"])
    with tab1:
        st.markdown("<div class='sharp-box'><b>Strategy: High-Frequency BTTS</b><br><i>Criteria:</i> BTTS Prob > 65%</div>", unsafe_allow_html=True)
        matches_found = []
        for m in data["upcoming"]:
            _, _, _, _, _, _, p_btts, _, _, _ = _cached_match_math(m)
            if p_btts > 65.0:
                matches_found.append({
                    "Match":        f"{m['home']} vs {m['away']}",
                    "League":       m["league"],
                    "BTTS Prob":    f"{p_btts:.1f}%",
                    "Implied Odds": f"{(100/max(0.1, p_btts)):.2f}",
                })
        if matches_found: st.dataframe(pd.DataFrame(matches_found), use_container_width=True)
        else:             st.info("No matches meet the BTTS criteria.")
    with tab2:
        st.markdown("<div class='sharp-box'><b>Strategy: Late Game Volatility</b><br><i>Criteria:</i> Draw Prob > 28% & Exp. Goals > 2.5</div>", unsafe_allow_html=True)
        vol_matches = []
        for m in data["upcoming"]:
            p_h, p_d, p_a, h_xg, a_xg, _, _, _, _, _ = _cached_match_math(m)
            if p_d > 28.0 and (h_xg + a_xg) > 2.5:
                vol_matches.append({
                    "Match":         f"{m['home']} vs {m['away']}",
                    "Draw Prob":     f"{p_d:.1f}%",
                    "Exp. Goals":    f"{(h_xg+a_xg):.2f}",
                    "Target Market": "2nd Half Goals / Late Over",
                })
        if vol_matches: st.dataframe(pd.DataFrame(vol_matches), use_container_width=True)
        else:           st.info("No volatility candidates found.")

elif page == "📉 CLV & Value Tracker":
    st.title("📉 Closing Line Value (CLV) Tracker")
    st.markdown("Track whether your bets beat the market closing price.")
    c1, c2 = st.columns(2)
    with c1: bet_odds   = st.number_input("Your Bet Odds (Taken)", min_value=1.01, value=2.00, step=0.01)
    with c2: close_odds = st.number_input("Closing Odds",          min_value=1.01, value=1.90, step=0.01)
    clv_pct = ((bet_odds / close_odds) - 1) * 100
    st.markdown("### Result:")
    if clv_pct > 0:   st.success(f"✅ You beat the market by **{clv_pct:.2f}%** CLV!")
    elif clv_pct < 0: st.error(  f"❌ Value lost: **{clv_pct:.2f}%** CLV")
    else:             st.info(   "⚖️ Neutral (Market matched your price).")

elif page == "🏦 Bankroll Management":
    st.title("🏦 Advanced Bankroll Manager")
    k_bank = st.number_input("Current Bankroll", value=1000)
    k_odds = st.number_input("Odds Available",   value=2.10)
    k_prob = st.slider("Your Assessed Win Probability (%)", 0, 100, 50)
    rec_amt = calculate_kelly(k_prob, k_odds, k_bank)
    st.metric("Recommended Stake (Quarter Kelly)", f"${rec_amt:.2f}")
    ev = calculate_ev(k_prob, k_odds)
    colour = "ev-positive" if ev > 0 else "ev-negative"; sign = "+" if ev > 0 else ""
    st.markdown(
        f"<div class='metric-card' style='max-width:250px;'>"
        f"<div class='metric-title'>Expected Value</div>"
        f"<div class='metric-value {colour}'>{sign}{ev:.2f}%</div></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# 🧠 MODEL HEALTH  (V3: Brier score, per-league breakdown, SHAP chart)
# ---------------------------------------------------------------------------
elif page == "🧠 Model Health":
    st.title("🧠 Model Health & Performance")
    metrics = load_kv("model_metrics", {})

    if not metrics:
        st.warning("No model metrics found. Run train_ml.py first.")
    else:
        # Header row
        col_v, col_t = st.columns(2)
        col_v.markdown(f"**Model Version:** `{metrics.get('model_version','unknown')}`")
        col_t.markdown(f"**Trained:** `{str(metrics.get('trained_at','?'))[:19]}`")
        if metrics.get("optuna_used"):
            st.success("🔍 Optuna hyperparameter search was used for this model.")
        if metrics.get("n_features"):
            st.caption(f"Features ({metrics['n_features']}): `{', '.join(metrics.get('feature_cols', []))}`")
        st.divider()

        # Training data
        c1, c2 = st.columns(2)
        c1.metric("Real Training Records", metrics.get("train_records_real", "—"))
        c2.metric("Total (incl. synthetic)", metrics.get("train_records_total", "—"))
        st.divider()

        # Base model
        st.subheader("⚙️ Base XGBoost Model")
        c1, c2 = st.columns(2)
        c1.metric("Train Accuracy",  f"{float(metrics.get('base_train_accuracy', 0)):.1%}")
        c2.metric("Train Log-Loss",  f"{float(metrics.get('base_train_logloss', 0)):.4f}")
        st.divider()

        # Meta-model OOF
        st.subheader("🧠 Meta-Ensemble (OOF Cross-Validation)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OOF Accuracy",  f"{float(metrics.get('meta_oof_accuracy', 0)):.1%}")
        c2.metric("OOF Log-Loss",  f"{float(metrics.get('meta_oof_logloss', 0)):.4f}")
        c3.metric("OOF Brier",     f"{float(metrics.get('meta_oof_brier', 0)):.4f}")
        c4.metric("Random Baseline LL", f"{float(metrics.get('random_baseline_logloss', 1.099)):.4f}")

        st.info(
            "**OOF** (Out-of-Fold) metrics are computed on held-out folds — genuine generalisation estimates.  "
            "Log-loss < 1.0 is good; random baseline = ln(3) ≈ 1.099.  "
            "Brier score < 0.33 is good; random baseline ≈ 0.444."
        )

        # Per-league breakdown
        league_metrics = metrics.get("league_metrics", {})
        if league_metrics:
            st.divider()
            st.subheader("🌍 Per-League Accuracy")
            lg_df = pd.DataFrame([
                {"League": lg, "Matches": v["n"], "Accuracy": f"{v['accuracy']:.1%}"}
                for lg, v in sorted(league_metrics.items(), key=lambda x: -x[1]["n"])
            ])
            st.dataframe(lg_df, use_container_width=True, hide_index=True)

        # SHAP importances
        shap_data = load_kv("shap_importances", {})
        if shap_data:
            st.divider()
            st.subheader("🔍 SHAP Feature Importances")
            tab_a, tab_d, tab_h = st.tabs(["Away Win", "Draw", "Home Win"])
            for tab, cls in [(tab_a,"Away Win"), (tab_d,"Draw"), (tab_h,"Home Win")]:
                with tab:
                    importances = shap_data.get(cls, {})
                    if importances:
                        sorted_feats = sorted(importances.items(), key=lambda x: -x[1])
                        max_val = max(v for _, v in sorted_feats) or 1.0
                        html = ""
                        for feat, val in sorted_feats:
                            bar_w = int(val / max_val * 100)
                            html += (
                                f"<div style='margin-bottom:6px;'>"
                                f"<div style='display:flex; justify-content:space-between; font-size:11px; color:#ccc;'>"
                                f"<span>{feat}</span><span>{val:.5f}</span></div>"
                                f"<div class='shap-bar-bg'>"
                                f"<div class='shap-bar-fill' style='width:{bar_w}%; background:#0984e3;'></div>"
                                f"</div></div>"
                            )
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.caption("No SHAP data for this class.")

    # ---------------------------------------------------------------------------
    # TIER PERFORMANCE TRACKING — rolling accuracy by tier & time window
    # ---------------------------------------------------------------------------
    if data["recent"]:
        st.divider()
        st.subheader("📊 Live Tier Performance Tracker")
        st.caption("Accuracy of Diamond/Emerald AI picks graded against actual results (last 90 days).")

        emerald_thresh_global = 90.0  # default; True xG matches use 85

        graded_all = []
        for m in data["recent"]:
            try:
                stats   = _cached_match_math(m)
                ai_pick, top_p, eth = get_best_pick(stats, m)
                score_str = m.get("score", "")
                if "-" not in str(score_str) or top_p < 65.0: continue
                h_g, a_g  = map(int, score_str.split("-")); total_g = h_g + a_g
                won = False
                if   ai_pick == "Home Win"   and h_g > a_g:          won = True
                elif ai_pick == "Draw"        and h_g == a_g:         won = True
                elif ai_pick == "Away Win"    and a_g > h_g:          won = True
                elif ai_pick == "Over 2.5"   and total_g > 2:         won = True
                elif ai_pick == "BTTS"        and h_g>0 and a_g>0:    won = True
                elif ai_pick == "Under 3.5"  and total_g < 4:         won = True
                elif ai_pick == "Over 1.5"   and total_g > 1:         won = True
                elif ai_pick == "Under 4.5"  and total_g < 5:         won = True
                elif "-1.5" in ai_pick:
                    if m["home"] in ai_pick and (h_g - a_g) >= 2:     won = True
                    elif m["away"] in ai_pick and (a_g - h_g) >= 2:   won = True
                elif "+1.5" in ai_pick:
                    if m["home"] in ai_pick and (h_g - a_g) >= -1:    won = True
                    elif m["away"] in ai_pick and (a_g - h_g) >= -1:  won = True

                tier = ("Emerald" if top_p >= eth else
                        "Diamond+" if top_p >= 75.0 else "Diamond")
                try:
                    match_date = datetime.fromisoformat(
                        str(m.get("date", "")).replace("Z", "+00:00")
                    ).date()
                except Exception:
                    match_date = None

                graded_all.append({
                    "date":       match_date,
                    "pick":       ai_pick,
                    "confidence": top_p,
                    "tier":       tier,
                    "won":        won,
                    "match":      f"{m['home']} vs {m['away']}",
                })
            except Exception:
                pass

        if graded_all:
            df_g = pd.DataFrame(graded_all)
            today_d = datetime.now().date()

            # ── Top-line tier summary metrics ──
            tier_order = ["Emerald", "Diamond+", "Diamond"]
            tier_colors = {"Emerald": "#00b894", "Diamond+": "#0abde3", "Diamond": "#74b9ff"}
            summary_cols = st.columns(len(tier_order))
            for col, tier in zip(summary_cols, tier_order):
                tier_df = df_g[df_g["tier"] == tier]
                if len(tier_df) == 0:
                    col.metric(f"{tier} Picks", "—", "No data")
                    continue
                acc     = tier_df["won"].mean()
                n       = len(tier_df)
                avg_conf = tier_df["confidence"].mean()
                col.markdown(
                    f"<div style='background:#1e272e; border:1px solid {tier_colors[tier]}; "
                    f"border-radius:10px; padding:14px; text-align:center;'>"
                    f"<div style='font-size:11px; color:{tier_colors[tier]}; font-weight:700; "
                    f"text-transform:uppercase; letter-spacing:1px;'>{tier}</div>"
                    f"<div style='font-size:28px; font-weight:900; color:#fff;'>{acc:.0%}</div>"
                    f"<div style='font-size:11px; color:#aaa;'>{n} picks · avg {avg_conf:.0f}% conf</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Rolling window selector ──
            window_opt = st.radio(
                "Rolling window", ["7 days", "30 days", "90 days"],
                index=1, horizontal=True, key="mh_window"
            )
            window_days = int(window_opt.split()[0])
            cutoff = today_d - timedelta(days=window_days)
            df_win = df_g[df_g["date"].notna() & (df_g["date"] >= cutoff)]

            if len(df_win) == 0:
                st.info(f"No graded picks in the last {window_days} days.")
            else:
                # ── Rolling daily accuracy chart ──
                df_dated = df_win[df_win["date"].notna()].copy()
                if len(df_dated) >= 3:
                    try:
                        daily = (
                            df_dated.groupby("date")["won"]
                            .agg(["sum", "count"])
                            .rename(columns={"sum": "wins", "count": "total"})
                            .reset_index()
                            .sort_values("date")
                        )
                        daily["accuracy"] = daily["wins"] / daily["total"]
                        # 5-day rolling mean for smoothing
                        daily["rolling_acc"] = daily["accuracy"].rolling(5, min_periods=1).mean()

                        import json as _json
                        chart_dates  = [str(d) for d in daily["date"].tolist()]
                        chart_acc    = [round(float(v)*100, 1) for v in daily["accuracy"].tolist()]
                        chart_roll   = [round(float(v)*100, 1) for v in daily["rolling_acc"].tolist()]
                        chart_counts = [int(v) for v in daily["total"].tolist()]

                        chart_html = f"""
<div style="background:#1e272e; border:1px solid #333; border-radius:10px; padding:16px; margin-bottom:16px;">
  <div style="font-size:11px; color:#b2bec3; font-weight:700; text-transform:uppercase;
              letter-spacing:1px; margin-bottom:12px;">Daily Pick Accuracy — {window_days}-Day Window</div>
  <canvas id="tierChart" style="width:100%; max-height:220px;"></canvas>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<script>
(function(){{
  var ctx = document.getElementById('tierChart');
  if (!ctx) return;
  var labels = {_json.dumps(chart_dates)};
  var daily  = {_json.dumps(chart_acc)};
  var roll   = {_json.dumps(chart_roll)};
  var counts = {_json.dumps(chart_counts)};
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [
        {{
          type: 'bar', label: 'Daily Accuracy (%)',
          data: daily, backgroundColor: 'rgba(116,185,255,0.35)',
          borderColor: '#74b9ff', borderWidth: 1, yAxisID: 'y',
        }},
        {{
          type: 'line', label: '5-Day Rolling Avg (%)',
          data: roll, borderColor: '#00b894', backgroundColor: 'transparent',
          borderWidth: 2.5, pointRadius: 3, tension: 0.4, yAxisID: 'y',
        }},
        {{
          type: 'line', label: 'Picks per day',
          data: counts, borderColor: '#feca57', backgroundColor: 'transparent',
          borderWidth: 1.5, borderDash: [4,4], pointRadius: 2, tension: 0.3, yAxisID: 'y2',
        }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: '#ccc', font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#888', maxRotation: 45, font: {{ size: 9 }} }},
               grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
        y: {{ position: 'left', min: 0, max: 100,
              ticks: {{ color: '#ccc', font: {{ size: 10 }},
                        callback: function(v){{ return v+'%'; }} }},
              grid: {{ color: 'rgba(255,255,255,0.07)' }} }},
        y2: {{ position: 'right', min: 0,
               ticks: {{ color: '#feca57', font: {{ size: 10 }} }},
               grid: {{ drawOnChartArea: false }} }},
      }}
    }}
  }});
}})();
</script>
"""
                        # Use height that fits the chart
                        _stc.html(chart_html, height=280)
                    except Exception as _ce:
                        log.warning("Tier chart error: %s", _ce)

                # ── Tier breakdown table for window ──
                st.markdown(f"**Breakdown by tier — last {window_days} days**")
                rows_win = []
                for tier in tier_order:
                    td = df_win[df_win["tier"] == tier]
                    if len(td) == 0: continue
                    rows_win.append({
                        "Tier":       tier,
                        "Picks":      len(td),
                        "Wins":       int(td["won"].sum()),
                        "Accuracy":   f"{td['won'].mean():.1%}",
                        "Avg Conf":   f"{td['confidence'].mean():.1f}%",
                        "Top Pick":   td["pick"].value_counts().idxmax() if len(td) > 0 else "—",
                    })
                if rows_win:
                    st.dataframe(pd.DataFrame(rows_win), use_container_width=True, hide_index=True)

                # ── Confidence calibration: does higher confidence = higher win rate? ──
                st.markdown("**Confidence calibration**")
                st.caption("Each bin shows win rate for picks in that confidence range.")
                bins = [(65,70),(70,75),(75,80),(80,85),(85,90),(90,100)]
                calib_rows = []
                for lo, hi in bins:
                    bd = df_win[(df_win["confidence"] >= lo) & (df_win["confidence"] < hi)]
                    if len(bd) == 0: continue
                    calib_rows.append({
                        "Confidence Band": f"{lo}–{hi}%",
                        "Picks": len(bd),
                        "Win Rate": f"{bd['won'].mean():.1%}",
                    })
                if calib_rows:
                    st.dataframe(pd.DataFrame(calib_rows), use_container_width=True, hide_index=True)

elif page == "📡 Data Control Center":
    st.title("📡 Data Control Center")
    st.info(
        "⚠️ All data fetching runs via GitHub Actions (`train_master.py`). "
        "Trigger the action manually from your repository's Actions tab."
    )
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🤖 GitHub Actions Sync")
        st.caption("Data is fetched automatically by your GitHub Actions pipeline.")
        repo_url    = os.getenv("GITHUB_REPOSITORY", "")
        actions_url = f"https://github.com/{repo_url}/actions" if repo_url else "https://github.com"
        st.markdown(f"[👉 Open GitHub Actions]({actions_url})")
    with c2:
        st.subheader("💾 System Maintenance")
        if st.button("🧹 Clear Application Cache"):
            load_data.clear()
            st.success("Cache cleared! Data refreshed from SQLite.")