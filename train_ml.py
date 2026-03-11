"""
train_ml.py — V5.0 Elite Quant ML Pipeline
===========================================
Key upgrades vs V4.0:
  • Extended meta passthrough: 10 raw features (was 6) — adds cs_h, cs_a, injury_diff, league_avg_goals
  • StandardScaler on raw meta features → better LogisticRegression gradient flow
  • Half-life increased 30 → 60 days: football team quality is more stable over ~2 months
  • O2.5 meta-learner: DC_o25 + XGB_o25 + LGB_o25 → LR meta (was simple XGB-only)
  • LightGBM O2.5 model trained with separate Optuna-tuned params
  • All meta feature scalers saved to disk for dashboard inference
"""

import json
import logging
import math
import sqlite3
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
# XGBoost 2.0 dropped sklearn ClassifierMixin; restore for sklearn 1.8+ (get_tags) compatibility
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
    xgb.XGBClassifier._estimator_type = "classifier"  # fallback for older sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb

    class _LGBMFloat64(lgb.LGBMClassifier):
        """Wraps predict_proba to return float64 — sklearn 1.8 sigmoid calibration requires it."""
        def predict_proba(self, X, **kwargs):
            return super().predict_proba(X, **kwargs).astype(float)

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logging.warning("lightgbm not installed — falling back to XGB-only stack. pip install lightgbm")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
log.info("🧠 V5.0 ELITE QUANT ML PIPELINE (26+10 meta features, XGB+LGB+scaled) — starting...")

# ---------------------------------------------------------------------------
# LEAGUE-SPECIFIC rho (Dixon-Coles low-score correlation)
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
# FEATURE SET — 26 features; must exactly match what dashboard.py builds
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Core xG — venue-specific from V3.3 (2)
    "h_xg", "a_xg",
    # Pi rating & table position (6)
    "pi_diff", "expected_points_delta",
    "pts_from_safety", "pts_from_title",
    "a_pts_from_safety", "a_pts_from_title",
    # Injury (1)
    "injury_diff",
    # Defensive profile (4 raw + 3 derived)
    "cs_h", "cs_a", "fts_h", "fts_a",
    "cs_diff",              # cs_h - cs_a: net home defensive edge
    "defensive_dominance",  # (cs_h + fts_a)/2: P(home keeps clean sheet)
    "attacking_vulnerability",  # (cs_a + fts_h)/2: P(away keeps clean sheet)
    # Form & tempo — exponential decay from V3.3 (3)
    "form_diff", "h_fast", "a_leak",
    # H2H history (3)
    "h2h_h_win_rate", "h2h_draw_rate", "h2h_a_win_rate",
    # League context (3)
    "league_home_win_rate", "league_draw_rate", "league_avg_goals",
    # Sharp market signal (2)
    "pin_implied_h", "pin_implied_a",
    # Model vs Pinnacle edge — new V3.3 (2)
    "market_edge_h", "market_edge_a",
]
N_FEAT = len(FEATURE_COLS)  # 26

# V5.0: Extended raw features passed through to meta-learner (was 6, now 10)
# Adds defensive profile (cs_h, cs_a), injury context, and league goal rate
META_RAW_COLS = [
    "h_xg", "a_xg", "pi_diff", "form_diff", "pin_implied_h", "pin_implied_a",
    "cs_h", "cs_a", "injury_diff", "league_avg_goals",
]
N_META_RAW = len(META_RAW_COLS)  # 10

# Goals-market feature set — 19 features tuned for O2.5 prediction.
# Drops table-position and 1X2-distribution features (they predict *who* wins,
# not *how many* goals are scored).  Keeps defensive interaction features
# and H2H draw rate as a "boring fixture" proxy.
FEATURE_COLS_O25 = [
    # Core expected goals — primary drivers of total goals (2)
    "h_xg", "a_xg",
    # Defensive profile raw (4) + derived interactions (2)
    "cs_h", "cs_a", "fts_h", "fts_a",
    "defensive_dominance",      # (cs_h + fts_a)/2 — P(home keeps clean sheet)
    "attacking_vulnerability",  # (cs_a + fts_h)/2 — P(away keeps clean sheet)
    # Tempo / style (2)
    "h_fast", "a_leak",
    # League goals baseline (1)
    "league_avg_goals",
    # Pinnacle market signals — encode sharp goal-line opinion (2)
    "pin_implied_h", "pin_implied_a",
    # Model-market divergence (2)
    "market_edge_h", "market_edge_a",
    # Strength gap — large mismatches reduce competitive pressure → fewer goals (1)
    "pi_diff",
    # Missing attackers reduce scoring (1)
    "injury_diff",
    # H2H draw rate — boring fixture proxy; high = historically defensive games (1)
    "h2h_draw_rate",
    # Form diff — chasing team opens up → more goals (1)
    "form_diff",
]


# ---------------------------------------------------------------------------
# DATABASE HELPERS
# ---------------------------------------------------------------------------

def _db():
    return sqlite3.connect("database.sqlite")

def load_kv(key, default=None):
    try:
        conn = _db(); c = conn.cursor()
        c.execute("SELECT value FROM kv_store WHERE key=?", (key,))
        row = c.fetchone(); conn.close()
        if row:
            return json.loads(row[0])
    except Exception as e:
        log.warning("load_kv(%s): %s", key, e)
    return default if default is not None else {}

def save_kv(key, data):
    conn = _db(); c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value TEXT)")
    c.execute("REPLACE INTO kv_store (key,value) VALUES (?,?)", (key, json.dumps(data)))
    conn.commit(); conn.close()


# ---------------------------------------------------------------------------
# DIXON-COLES HELPERS
# ---------------------------------------------------------------------------

def dc_tau(i, j, mu, nu, rho):
    if   i==0 and j==0: return max(0.0, 1.0 - mu*nu*rho)
    elif i==1 and j==0: return max(0.0, 1.0 + nu*rho)
    elif i==0 and j==1: return max(0.0, 1.0 + mu*rho)
    elif i==1 and j==1: return max(0.0, 1.0 - rho)
    return 1.0

def build_prob_matrix(h_xg, a_xg, rho=DEFAULT_RHO, n=10):
    m = np.zeros((n, n))
    for i in range(n):
        ph = math.exp(-h_xg) * (h_xg**i) / math.factorial(i)
        for j in range(n):
            pa = math.exp(-a_xg) * (a_xg**j) / math.factorial(j)
            m[i, j] = ph * pa * dc_tau(i, j, h_xg, a_xg, rho)
    t = m.sum()
    return m / t if t > 0 else m

def get_dc_probs(h_xg, a_xg, rho=DEFAULT_RHO):
    """Returns (p_away, p_draw, p_home) — matches class order 0=Away,1=Draw,2=Home."""
    m = build_prob_matrix(h_xg, a_xg, rho)
    return float(np.sum(np.triu(m,1))), float(np.sum(np.diag(m))), float(np.sum(np.tril(m,-1)))

def form_pts(form_str):
    if not form_str or str(form_str) == "?????": return 5.0
    return float(sum(3 if c=="W" else 1 if c=="D" else 0 for c in str(form_str).upper()))


# ---------------------------------------------------------------------------
# VERSION-AGNOSTIC OOF HELPER
# sklearn 1.4+ removed fit_params from cross_val_predict; 1.5+ requires
# metadata routing for params=.  Manual fold loop avoids all of this.
# ---------------------------------------------------------------------------

def _oof_predict(estimator, X, y, cv, method="predict_proba", sample_weight=None):
    """
    Out-of-fold predictions with optional sample_weight.
    Fully compatible with any sklearn version — uses a manual CV loop.
    """
    from sklearn.base import clone
    n = len(y)
    n_classes = len(np.unique(y))
    if method == "predict_proba":
        out = np.zeros((n, n_classes), dtype=np.float64)
    else:
        out = np.zeros(n, dtype=np.float64)

    for train_idx, val_idx in cv.split(X, y):
        clf = clone(estimator)
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y[train_idx]
        fit_kw = {"sample_weight": sample_weight[train_idx]} if sample_weight is not None else {}
        clf.fit(X_tr, y_tr, **fit_kw)
        if method == "predict_proba":
            # Cast to float64 BEFORE normalising so the division is done in float64.
            # XGBoost/LGB return float32; float32→float64 upcast can shift row sums
            # by ~1e-7 which triggers sklearn's log_loss UserWarning.
            p = clf.predict_proba(X_val).astype(np.float64)
            out[val_idx] = p / p.sum(axis=1, keepdims=True)
        else:
            out[val_idx] = clf.predict(X_val)

    if method == "predict_proba":
        # Final renorm pass guarantees rows sum to 1.0 in float64 after accumulation.
        out /= out.sum(axis=1, keepdims=True)
    return out


def _safe_proba(clf, X):
    """predict_proba cast to float64 and row-normalised — avoids log_loss UserWarning."""
    p = clf.predict_proba(X).astype(np.float64)
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# EXPECTED CALIBRATION ERROR (ECE) — proper calibration metric
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, y_proba, n_bins=15):
    """
    Multiclass ECE: measures how well predicted probabilities match actual frequencies.
    Lower is better. A perfectly calibrated model has ECE = 0.
    """
    ece = 0.0
    n = len(y_true)
    for cls in range(y_proba.shape[1]):
        probs = y_proba[:, cls]
        targets = (y_true == cls).astype(int)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        for b in range(n_bins):
            mask = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1])
            if mask.sum() == 0:
                continue
            avg_pred = probs[mask].mean()
            avg_true = targets[mask].mean()
            ece += mask.sum() / n * abs(avg_pred - avg_true)
    return ece / y_proba.shape[1]


# ---------------------------------------------------------------------------
# TEMPERATURE SCALING — post-hoc calibration sharpening
# ---------------------------------------------------------------------------

def find_best_temperature(logits, y_true, temps=None):
    """
    Find the temperature T that minimises log-loss on logits.
    Returns best_T. Applied as: probs = softmax(logits / T).
    """
    if temps is None:
        temps = np.arange(0.5, 3.01, 0.05)
    best_t, best_ll = 1.0, float("inf")
    for t in temps:
        scaled = logits / t
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        ll = log_loss(y_true, probs)
        if ll < best_ll:
            best_ll = ll
            best_t = float(t)
    return best_t


# ---------------------------------------------------------------------------
# LOAD DATA FROM DATABASE
# ---------------------------------------------------------------------------

log.info("📊 Loading data from database...")
recent          = load_kv("recent", [])
pi_ratings      = load_kv("pi_ratings", {})
pro_preds       = load_kv("pro_preds", {})
team_forms      = load_kv("team_forms", {})
hist_odds_cache = load_kv("hist_odds_cache", {})

dataset = []
match_dates = []  # V4.0: track dates for time-decay weighting

for r in recent:
    try:
        h_score, a_score = map(int, r["score"].split("-"))
        target     = 2 if h_score>a_score else 1 if h_score==a_score else 0
        target_o25 = 1 if (h_score+a_score) > 2 else 0

        h_name, a_name = r["home"], r["away"]
        league = r.get("league", "")
        h_pi = float(pi_ratings.get(h_name, 0.0))
        a_pi = float(pi_ratings.get(a_name, 0.0))
        pi_diff = h_pi - a_pi

        pp = pro_preds.get(f"{h_name} vs {a_name}", {})
        h_xg = float(pp.get("h_xg", max(0.2, 1.45 + pi_diff/2.0)))
        a_xg = float(pp.get("a_xg", max(0.2, 1.20 - pi_diff/2.0)))

        # Historical odds — real Pinnacle probs for completed fixtures.
        # hist_odds_cache is keyed by fixture_id (str); falls back to pro_preds
        # (which has odds for upcoming matches only) then to 0.0.
        fid = str(r.get("fixture_id", ""))
        h_odds = hist_odds_cache.get(fid, {})
        pin_h = float(h_odds.get("pin_implied_h") or pp.get("pin_implied_h") or 0.0)
        pin_a = float(h_odds.get("pin_implied_a") or pp.get("pin_implied_a") or 0.0)
        # Market edge: DC model prob vs Pinnacle (same logic as train_master.py)
        _dc = get_dc_probs(h_xg, a_xg, rho=LEAGUE_RHO.get(r.get("league", ""), DEFAULT_RHO))
        _dc_h, _dc_a = float(_dc[2]), float(_dc[0])  # [2]=home, [0]=away per class order
        m_edge_h = round((_dc_h - pin_h) * 100, 2) if pin_h > 0 else 0.0
        m_edge_a = round((_dc_a - pin_a) * 100, 2) if pin_a > 0 else 0.0

        h_form = form_pts(team_forms.get(str(r.get("home_id", "")), "WWWDD"))
        a_form = form_pts(team_forms.get(str(r.get("away_id", "")), "WWDLL"))

        # Parse match date for time-decay weighting
        match_date_str = r.get("date", "")
        try:
            match_date = datetime.fromisoformat(str(match_date_str).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            match_date = datetime.now() - timedelta(days=45)  # default mid-range
        match_dates.append(match_date)

        dataset.append({
            "h_xg":                  h_xg,
            "a_xg":                  a_xg,
            "pi_diff":               pi_diff,
            "expected_points_delta": float(pp.get("expected_points_delta", 0.0)),
            "pts_from_safety":       float(pp.get("pts_from_safety", 10.0)),
            "pts_from_title":        float(pp.get("pts_from_title", -15.0)),
            "a_pts_from_safety":     float(pp.get("a_pts_from_safety", 10.0)),
            "a_pts_from_title":      float(pp.get("a_pts_from_title", -15.0)),
            "injury_diff":           float(pp.get("injury_diff", 0.0)),
            "cs_h":                  float(pp.get("cs_h", 0.30)),
            "cs_a":                  float(pp.get("cs_a", 0.25)),
            "fts_h":                 float(pp.get("fts_h", 0.25)),
            "fts_a":                 float(pp.get("fts_a", 0.28)),
            "cs_diff":               float(pp.get("cs_h", 0.30)) - float(pp.get("cs_a", 0.25)),
            "defensive_dominance":   (float(pp.get("cs_h", 0.30)) + float(pp.get("fts_a", 0.28))) / 2.0,
            "attacking_vulnerability": (float(pp.get("cs_a", 0.25)) + float(pp.get("fts_h", 0.25))) / 2.0,
            "form_diff":             float(pp.get("h_form_pts", h_form)) - float(pp.get("a_form_pts", a_form)),
            "h_fast":                float(pp.get("h_fast", 22.0)),
            "a_leak":                float(pp.get("a_leak", 16.0)),
            "h2h_h_win_rate":        float(pp.get("h2h_h_win_rate", 0.33)),
            "h2h_draw_rate":         float(pp.get("h2h_draw_rate",  0.25)),
            "h2h_a_win_rate":        float(pp.get("h2h_a_win_rate", 0.33)),
            "league_home_win_rate":  float(pp.get("league_home_win_rate", 0.45)),
            "league_draw_rate":      float(pp.get("league_draw_rate",     0.25)),
            "league_avg_goals":      float(pp.get("league_avg_goals",     2.65)),
            "pin_implied_h":         pin_h,
            "pin_implied_a":         pin_a,
            "market_edge_h":         m_edge_h,
            "market_edge_a":         m_edge_a,
            "target":                target,
            "target_o25":            target_o25,
            "league":                league,
        })
    except Exception as e:
        log.debug("Skipping malformed record: %s", e)

n_real = len(dataset)
log.info("Loaded %d real records.", n_real)


# ---------------------------------------------------------------------------
# V4.0: TIME-DECAY SAMPLE WEIGHTS (exponential, half-life = 30 days)
# ---------------------------------------------------------------------------
now = datetime.now()
# V5.0: 60-day half-life (was 30). Football team quality is stable over ~2 months;
# 30 days over-weights 1-2 recent results which adds noise rather than signal.
HALF_LIFE_DAYS = 60.0
decay_lambda = math.log(2) / HALF_LIFE_DAYS

real_weights = []
for dt in match_dates:
    days_ago = max(0, (now - dt).total_seconds() / 86400)
    real_weights.append(math.exp(-decay_lambda * days_ago))

if real_weights:
    # Normalise so mean weight = 1.0 (preserves effective sample size interpretation)
    w_mean = sum(real_weights) / len(real_weights)
    real_weights = [w / w_mean for w in real_weights]
    log.info("Time-decay weights: min=%.3f  max=%.3f  half-life=%dd",
             min(real_weights), max(real_weights), int(HALF_LIFE_DAYS))


# ---------------------------------------------------------------------------
# SYNTHETIC BOOTSTRAP — V4.0: lower cap (1500), correlated, down-weighted
# ---------------------------------------------------------------------------
SYNTHETIC_CAP = 1500
SYNTHETIC_WEIGHT = 0.3  # synthetic records contribute 30% each vs a typical real record

n_needed = max(0, SYNTHETIC_CAP - n_real)
if n_needed > 0:
    log.info("⚠️  Bootstrapping %d synthetic records (weight=%.1f).", n_needed, SYNTHETIC_WEIGHT)
    rng = np.random.default_rng(42)

    if n_real >= 50:
        df_r = pd.DataFrame(dataset)
        # V4.0: preserve feature correlations via multivariate sampling
        key_cols = ["h_xg", "a_xg", "pi_diff", "form_diff"]
        avail = [c for c in key_cols if c in df_r.columns]
        if len(avail) >= 2:
            corr_means = df_r[avail].mean().values
            corr_cov   = df_r[avail].cov().values
            # Add small ridge for numerical stability
            corr_cov += np.eye(len(avail)) * 0.01
        else:
            corr_means = None
        h_mu, h_sg = df_r["h_xg"].mean(), max(0.2, df_r["h_xg"].std())
        a_mu, a_sg = df_r["a_xg"].mean(), max(0.2, df_r["a_xg"].std())
    else:
        h_mu, h_sg = 1.55, 0.55
        a_mu, a_sg = 1.20, 0.50
        corr_means = None

    for _ in range(n_needed):
        # V4.0: correlated feature generation
        if corr_means is not None and len(avail) >= 2:
            sample = rng.multivariate_normal(corr_means, corr_cov)
            hx = max(0.3, float(sample[0]))
            ax = max(0.3, float(sample[1]))
            s_pi_diff = float(sample[2]) if len(avail) > 2 else float(rng.normal(0, 0.8))
            s_form_diff = float(sample[3]) if len(avail) > 3 else float(rng.uniform(-9, 9))
        else:
            hx = max(0.3, float(rng.normal(h_mu, h_sg)))
            ax = max(0.3, float(rng.normal(a_mu, a_sg)))
            s_pi_diff = float(rng.normal(0, 0.8))
            s_form_diff = float(rng.uniform(-9, 9))

        hg = int(rng.poisson(hx)); ag = int(rng.poisson(ax))
        t  = 2 if hg>ag else 1 if hg==ag else 0
        dataset.append({
            "h_xg":                  hx,
            "a_xg":                  ax,
            "pi_diff":               s_pi_diff,
            "expected_points_delta": float(rng.uniform(-5, 5)),
            "pts_from_safety":       float(rng.uniform(-5, 25)),
            "pts_from_title":        float(rng.uniform(-30, 0)),
            "a_pts_from_safety":     float(rng.uniform(-5, 25)),
            "a_pts_from_title":      float(rng.uniform(-30, 0)),
            "injury_diff":           float(rng.uniform(-0.3, 0.3)),
            "cs_h":                  (_s_cs_h := float(rng.beta(3, 7))),
            "cs_a":                  (_s_cs_a := float(rng.beta(2, 7))),
            "fts_h":                 (_s_fts_h := float(rng.beta(2, 7))),
            "fts_a":                 (_s_fts_a := float(rng.beta(3, 7))),
            "cs_diff":               _s_cs_h - _s_cs_a,
            "defensive_dominance":   (_s_cs_h + _s_fts_a) / 2.0,
            "attacking_vulnerability": (_s_cs_a + _s_fts_h) / 2.0,
            "form_diff":             s_form_diff,
            "h_fast":                float(rng.uniform(10, 40)),
            "a_leak":                float(rng.uniform(8, 35)),
            "h2h_h_win_rate":        float(rng.dirichlet([3, 2, 2])[0]),
            "h2h_draw_rate":         float(rng.beta(2, 6)),
            "h2h_a_win_rate":        float(rng.dirichlet([3, 2, 2])[2]),
            "league_home_win_rate":  float(rng.uniform(0.38, 0.52)),
            "league_draw_rate":      float(rng.uniform(0.20, 0.30)),
            "league_avg_goals":      float(rng.uniform(2.2, 3.2)),
            "pin_implied_h":         0.0,
            "pin_implied_a":         0.0,
            "market_edge_h":         0.0,
            "market_edge_a":         0.0,
            "target":                t,
            "target_o25":            1 if (hg+ag)>2 else 0,
            "league":                "",
        })
else:
    log.info("Enough real data (%d) — no synthetic bootstrap needed.", n_real)

# Build combined sample weights: real (time-decay) + synthetic (down-weighted)
sample_weights = np.array(
    real_weights + [SYNTHETIC_WEIGHT] * n_needed,
    dtype=np.float64,
)

df = pd.DataFrame(dataset)
X       = df[FEATURE_COLS].values.astype(np.float32)
X_o25   = df[FEATURE_COLS_O25].values.astype(np.float32)   # 19-feature goals matrix
y_match = df["target"].values
y_o25   = df["target_o25"].values
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# V5.0: Extract raw features for meta-learner passthrough (10 features)
X_meta_raw = df[META_RAW_COLS].values.astype(np.float32)

# V5.0: StandardScaler on raw meta features.
# LogisticRegression is sensitive to feature scale — unscaled pi_diff, form_diff,
# and league_avg_goals have very different magnitudes vs the probability features.
# Fitting on all data (real + synthetic) is fine here since scaler is deterministic.
meta_scaler = StandardScaler()
X_meta_raw_scaled = meta_scaler.fit_transform(X_meta_raw).astype(np.float32)
log.info("V5.0: MetaScaler fitted on %d records, %d raw features (μ≈0, σ≈1 enforced).",
         len(X_meta_raw), N_META_RAW)

log.info("Total training records: %d (%d real + %d synthetic), %d features",
         len(df), n_real, n_needed, N_FEAT)


# ---------------------------------------------------------------------------
# V4.0: AUTO CALIBRATION SELECTION — isotonic vs sigmoid per model
# ---------------------------------------------------------------------------

def pick_calibration_method(base_clf, X, y, cv, sample_weight=None):
    """
    Compare isotonic vs sigmoid (Platt) calibration via OOF log-loss.
    Returns the method name that produces lower log-loss.
    Note: calibration selection uses uniform weights (method selection is robust to this).
    """
    results = {}
    for method in ["isotonic", "sigmoid"]:
        try:
            cal = CalibratedClassifierCV(base_clf, cv=cv, method=method)
            # cross_val_predict used here without sample_weight to avoid sklearn
            # version incompatibilities (fit_params removed in sklearn 1.5+).
            # Calibration method selection is robust to uniform weights.
            oof = cross_val_predict(cal, X, y, cv=cv, method="predict_proba")
            # Normalise to avoid log_loss UserWarning from isotonic calibration
            oof = oof / np.clip(oof.sum(axis=1, keepdims=True), 1e-12, None)
            results[method] = log_loss(y, oof)
        except Exception:
            results[method] = float("inf")
    best = min(results, key=results.get)
    log.info("Calibration selection: isotonic=%.4f  sigmoid=%.4f → using %s",
             results.get("isotonic", float("inf")), results.get("sigmoid", float("inf")), best)
    return best


# ---------------------------------------------------------------------------
# OPTUNA — XGBoost hyperparameter search (V4.0: 75 trials + sample weights)
# ---------------------------------------------------------------------------
if OPTUNA_AVAILABLE and n_real >= 150:
    log.info("🔍 Optuna XGB match search (75 trials)...")

    def xgb_objective(trial):
        p = dict(
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
            n_estimators     = trial.suggest_int("n_estimators", 100, 800),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 15),
            gamma            = trial.suggest_float("gamma", 0.0, 2.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 3.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 8.0),
        )
        clf = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=42, **p,
        )
        oof = _oof_predict(clf, X, y_match, cv=cv, sample_weight=sample_weights)
        # evaluate on real data only to avoid synthetic noise in metric
        return log_loss(y_match[:n_real], oof[:n_real])

    xgb_study = optuna.create_study(direction="minimize",
                                     pruner=optuna.pruners.MedianPruner())
    xgb_study.optimize(xgb_objective, n_trials=75, n_jobs=1)
    best_xgb_p = xgb_study.best_params
    log.info("✅ XGB Optuna best real-only log-loss=%.4f", xgb_study.best_value)
else:
    if not OPTUNA_AVAILABLE:
        log.warning("optuna not installed — using default XGB params.")
    else:
        log.info("< 150 real records — using default XGB params.")
    best_xgb_p = dict(
        max_depth=5, learning_rate=0.02, n_estimators=300,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
    )


# ---------------------------------------------------------------------------
# OPTUNA — LightGBM hyperparameter search (V4.0: 75 trials)
# ---------------------------------------------------------------------------
best_lgb_p = None
if LGB_AVAILABLE:
    if OPTUNA_AVAILABLE and n_real >= 150:
        log.info("🔍 Optuna LGB search (75 trials)...")

        def lgb_objective(trial):
            p = dict(
                num_leaves       = trial.suggest_int("num_leaves", 20, 200),
                learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
                n_estimators     = trial.suggest_int("n_estimators", 100, 800),
                max_depth        = trial.suggest_int("max_depth", 3, 12),
                subsample        = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
                min_child_samples= trial.suggest_int("min_child_samples", 5, 60),
                reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 3.0),
                reg_lambda       = trial.suggest_float("reg_lambda", 0.0, 8.0),
            )
            clf = _LGBMFloat64(
                objective="multiclass", num_class=3,
                random_state=42, verbose=-1, **p,
            )
            oof = _oof_predict(clf, X, y_match, cv=cv, sample_weight=sample_weights)
            return log_loss(y_match[:n_real], oof[:n_real])

        lgb_study = optuna.create_study(direction="minimize",
                                         pruner=optuna.pruners.MedianPruner())
        lgb_study.optimize(lgb_objective, n_trials=75, n_jobs=1)
        best_lgb_p = lgb_study.best_params
        log.info("✅ LGB Optuna best real-only log-loss=%.4f", lgb_study.best_value)
    else:
        log.info("Using default LGB params.")
        best_lgb_p = dict(
            num_leaves=63, learning_rate=0.02, n_estimators=300,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        )


# ---------------------------------------------------------------------------
# V4.0: OPTUNA — Separate O2.5 hyperparameter search
# ---------------------------------------------------------------------------
if OPTUNA_AVAILABLE and n_real >= 150:
    log.info("🔍 Optuna O2.5 search (50 trials)...")

    def o25_objective(trial):
        p = dict(
            max_depth        = trial.suggest_int("max_depth", 3, 7),
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
            n_estimators     = trial.suggest_int("n_estimators", 100, 600),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 15),
            gamma            = trial.suggest_float("gamma", 0.0, 2.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 3.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 8.0),
        )
        clf = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, **p,
        )
        oof = _oof_predict(clf, X_o25, y_o25, cv=cv, sample_weight=sample_weights)
        return log_loss(y_o25[:n_real], oof[:n_real, 1])

    o25_study = optuna.create_study(direction="minimize")
    o25_study.optimize(o25_objective, n_trials=50, n_jobs=1)
    best_o25_p = o25_study.best_params
    log.info("✅ O2.5 Optuna best real-only log-loss=%.4f", o25_study.best_value)
else:
    best_o25_p = best_xgb_p.copy()
    log.info("O2.5 using match XGB params (Optuna unavailable or <150 real records).")


# ---------------------------------------------------------------------------
# V4.0: AUTO-PICK CALIBRATION METHOD FOR BASE LEARNERS
# ---------------------------------------------------------------------------
log.info("🔬 Selecting calibration method for XGB match...")
xgb_base = xgb.XGBClassifier(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    random_state=42, **best_xgb_p,
)
xgb_cal_method = pick_calibration_method(xgb_base, X, y_match, cv, sample_weights)


# ---------------------------------------------------------------------------
# BASE XGBOOST MODELS (V4.0: with sample weights + auto calibration)
# ---------------------------------------------------------------------------
log.info("⚙️  Training base XGBoost models...")

clf_match = CalibratedClassifierCV(
    xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=42, **best_xgb_p,
    ),
    cv=5, method=xgb_cal_method,
)
clf_match.fit(X, y_match, sample_weight=sample_weights)

clf_o25 = CalibratedClassifierCV(
    xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, **best_o25_p,
    ),
    cv=5, method=xgb_cal_method,
)
clf_o25.fit(X_o25, y_o25, sample_weight=sample_weights)

xgb_train_acc = accuracy_score(y_match[:n_real], clf_match.predict(X[:n_real]))
xgb_train_ll  = log_loss(y_match[:n_real], _safe_proba(clf_match, X[:n_real]))
log.info("XGB base (real-only)  accuracy=%.3f  log-loss=%.4f  cal=%s",
         xgb_train_acc, xgb_train_ll, xgb_cal_method)


# ---------------------------------------------------------------------------
# BASE LIGHTGBM MODELS (V5.0: match + O2.5, with sample weights + auto calibration)
# ---------------------------------------------------------------------------
clf_lgb_match = None
clf_lgb_o25   = None
lgb_cal_method = "isotonic"
if LGB_AVAILABLE and best_lgb_p is not None:
    log.info("🔬 Selecting calibration method for LGB match...")
    lgb_base = _LGBMFloat64(
        objective="multiclass", num_class=3,
        random_state=42, verbose=-1, **best_lgb_p,
    )
    lgb_cal_method = pick_calibration_method(lgb_base, X, y_match, cv, sample_weights)

    log.info("⚙️  Training base LightGBM match model...")
    clf_lgb_match = CalibratedClassifierCV(
        _LGBMFloat64(
            objective="multiclass", num_class=3,
            random_state=42, verbose=-1, **best_lgb_p,
        ),
        cv=5, method=lgb_cal_method,
    )
    clf_lgb_match.fit(X, y_match, sample_weight=sample_weights)

    lgb_train_acc = accuracy_score(y_match[:n_real], clf_lgb_match.predict(X[:n_real]))
    lgb_train_ll  = log_loss(y_match[:n_real], _safe_proba(clf_lgb_match, X[:n_real]))
    log.info("LGB match (real-only) accuracy=%.3f  log-loss=%.4f  cal=%s",
             lgb_train_acc, lgb_train_ll, lgb_cal_method)

    # V5.0: LightGBM O2.5 model (binary) — Optuna params reused from best_o25_p if available
    log.info("⚙️  Training LightGBM O2.5 model...")
    # Build binary LGB params from match params (drop multiclass-specific keys)
    _lgb_o25_p = {k: v for k, v in best_lgb_p.items()
                  if k not in ("num_class", "objective")}
    clf_lgb_o25 = CalibratedClassifierCV(
        _LGBMFloat64(
            objective="binary", random_state=42, verbose=-1, **_lgb_o25_p,
        ),
        cv=5, method=lgb_cal_method,
    )
    clf_lgb_o25.fit(X_o25, y_o25, sample_weight=sample_weights)
    lgb_o25_ll = log_loss(y_o25[:n_real], _safe_proba(clf_lgb_o25, X_o25[:n_real]))
    log.info("LGB O2.5 (real-only)  log-loss=%.4f", lgb_o25_ll)


# ---------------------------------------------------------------------------
# OOF PROBABILITIES FOR META-STACK (V4.0: with sample weights)
# ---------------------------------------------------------------------------
log.info("🧠 Generating out-of-fold probabilities...")

xgb_oof = _oof_predict(
    xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=42, **best_xgb_p,
    ),
    X, y_match, cv=cv, sample_weight=sample_weights,
)

log.info("Computing league-adjusted Dixon-Coles probabilities...")
dc_oof = np.array([
    get_dc_probs(
        float(row["h_xg"]), float(row["a_xg"]),
        rho=LEAGUE_RHO.get(row.get("league", ""), DEFAULT_RHO),
    )
    for _, row in df.iterrows()
])

if LGB_AVAILABLE and best_lgb_p is not None:
    lgb_oof = _oof_predict(
        _LGBMFloat64(
            objective="multiclass", num_class=3,
            random_state=42, verbose=-1, **best_lgb_p,
        ),
        X, y_match, cv=cv, sample_weight=sample_weights,
    )
    # V5.0: DC(3) + XGB(3) + LGB(3) + scaled_raw(10) = 19 features
    X_meta = np.hstack((dc_oof, xgb_oof, lgb_oof, X_meta_raw_scaled))
    log.info("Meta stack: DC(3) + XGB(3) + LGB(3) + scaled_raw(10) = %d features", X_meta.shape[1])

    # Ensemble diversity: how often XGB and LGB disagree on predicted class
    xgb_pred = xgb_oof.argmax(axis=1)
    lgb_pred  = lgb_oof.argmax(axis=1)
    diversity = float((xgb_pred != lgb_pred).mean())
    log.info("📐 XGB vs LGB disagreement rate: %.1f%% (higher = more diverse ensemble)",
             diversity * 100)
else:
    # Fallback: DC + XGB + scaled raw features
    X_meta = np.hstack((dc_oof, xgb_oof, X_meta_raw_scaled))
    diversity = 0.0
    log.info("Meta stack: DC(3) + XGB(3) + scaled_raw(10) = %d features (LGB unavailable)", X_meta.shape[1])


# ---------------------------------------------------------------------------
# V4.0: META LOGISTIC REGRESSION with Optuna-tuned C
# ---------------------------------------------------------------------------
log.info("Training Logistic Regression meta-learner...")

# Tune meta-learner C regularisation
best_meta_C = 1.0
if OPTUNA_AVAILABLE and n_real >= 100:
    log.info("🔍 Optuna meta-learner C search (30 trials)...")

    def meta_objective(trial):
        c_val = trial.suggest_float("C", 0.01, 50.0, log=True)
        meta = LogisticRegression(
            solver="lbfgs",
            max_iter=2000, C=c_val, random_state=42,
            class_weight="balanced",  # upweight draw class (~25% of matches)
        )
        oof = _oof_predict(meta, X_meta, y_match, cv=cv, sample_weight=sample_weights)
        return log_loss(y_match[:n_real], oof[:n_real])

    meta_study = optuna.create_study(direction="minimize")
    meta_study.optimize(meta_objective, n_trials=30, n_jobs=1)
    best_meta_C = meta_study.best_params["C"]
    log.info("✅ Meta C=%.4f  best real-only log-loss=%.4f", best_meta_C, meta_study.best_value)

meta_model = LogisticRegression(
    solver="lbfgs",
    max_iter=2000, C=best_meta_C, random_state=42,
    class_weight="balanced",  # upweight draw class (~25% of matches)
)
meta_model.fit(X_meta, y_match, sample_weight=sample_weights)

assert list(meta_model.classes_) == [0, 1, 2], \
    f"Bad meta class order: {meta_model.classes_}"


# ---------------------------------------------------------------------------
# V4.0: TEMPERATURE SCALING on meta output
# ---------------------------------------------------------------------------
log.info("🌡️  Finding optimal temperature scaling...")
meta_oof_raw = _oof_predict(meta_model, X_meta, y_match, cv=cv, sample_weight=sample_weights)

# Convert OOF probabilities to logits for temperature scaling
meta_oof_logits = np.log(np.clip(meta_oof_raw, 1e-15, 1.0))
best_temperature = find_best_temperature(meta_oof_logits[:n_real], y_match[:n_real])
log.info("Optimal temperature: T=%.2f (1.0=no change, <1=sharpen, >1=smooth)", best_temperature)

# Apply temperature scaling to OOF for evaluation
if abs(best_temperature - 1.0) > 0.01:
    scaled_logits = meta_oof_logits / best_temperature
    exp_s = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
    meta_oof = exp_s / exp_s.sum(axis=1, keepdims=True)
    log.info("Temperature scaling applied to OOF probabilities.")
else:
    meta_oof = meta_oof_raw
    log.info("Temperature ~1.0 — no scaling needed.")


# ---------------------------------------------------------------------------
# V4.0: EVALUATION — real-data only (not polluted by synthetic)
# ---------------------------------------------------------------------------
meta_acc   = accuracy_score(y_match[:n_real], meta_oof[:n_real].argmax(axis=1))
meta_ll    = log_loss(y_match[:n_real], meta_oof[:n_real])
meta_brier = float(np.mean([
    brier_score_loss((y_match[:n_real]==cls).astype(int), meta_oof[:n_real, i])
    for i, cls in enumerate([0, 1, 2])
]))
meta_ece = expected_calibration_error(y_match[:n_real], meta_oof[:n_real])

log.info("Meta OOF (REAL ONLY, n=%d):", n_real)
log.info("  accuracy=%.3f  log-loss=%.4f  brier=%.4f  ECE=%.4f  (rnd baseline ll=%.4f)",
         meta_acc, meta_ll, meta_brier, meta_ece, math.log(3))

# Also show full-dataset metrics for comparison
meta_acc_all = accuracy_score(y_match, meta_oof.argmax(axis=1))
meta_ll_all  = log_loss(y_match, meta_oof)
log.info("Meta OOF (all %d):  accuracy=%.3f  log-loss=%.4f", len(df), meta_acc_all, meta_ll_all)


# ---------------------------------------------------------------------------
# V5.0: O2.5 META-LEARNER — DC_o25 + XGB_o25 + LGB_o25 → LogisticRegression
# ---------------------------------------------------------------------------
log.info("🧠 Building O2.5 meta-stack...")
meta_o25_model = None
meta_o25_ll    = None

# Compute DC O2.5 OOF
dc_o25_oof = np.array([
    sum(
        build_prob_matrix(float(row["h_xg"]), float(row["a_xg"]),
                          rho=LEAGUE_RHO.get(row.get("league", ""), DEFAULT_RHO))[i, j]
        for i in range(10) for j in range(10) if i + j > 2
    )
    for _, row in df.iterrows()
], dtype=np.float32).reshape(-1, 1)

# XGB O2.5 OOF (uses 19-feature goals matrix)
xgb_o25_oof = _oof_predict(
    xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, **best_o25_p,
    ),
    X_o25, y_o25, cv=cv, sample_weight=sample_weights,
)[:, 1].reshape(-1, 1)

if LGB_AVAILABLE and clf_lgb_o25 is not None:
    # LGB O2.5 OOF (uses 19-feature goals matrix)
    lgb_o25_oof = _oof_predict(
        _LGBMFloat64(
            objective="binary", random_state=42, verbose=-1,
            **{k: v for k, v in best_lgb_p.items() if k not in ("num_class", "objective")},
        ),
        X_o25, y_o25, cv=cv, sample_weight=sample_weights,
    )[:, 1].reshape(-1, 1)
    X_meta_o25 = np.hstack((dc_o25_oof, xgb_o25_oof, lgb_o25_oof))
    log.info("O2.5 meta stack: DC(1) + XGB(1) + LGB(1) = 3 features")
else:
    X_meta_o25 = np.hstack((dc_o25_oof, xgb_o25_oof))
    log.info("O2.5 meta stack: DC(1) + XGB(1) = 2 features (LGB unavailable)")

# Tune meta-O2.5 C via Optuna
best_meta_o25_C = 1.0
if OPTUNA_AVAILABLE and n_real >= 100:
    def meta_o25_objective(trial):
        c_val = trial.suggest_float("C", 0.01, 20.0, log=True)
        meta_o25 = LogisticRegression(solver="lbfgs", max_iter=1000, C=c_val, random_state=42)
        oof = _oof_predict(meta_o25, X_meta_o25, y_o25, cv=cv, sample_weight=sample_weights)
        return log_loss(y_o25[:n_real], oof[:n_real, 1])
    meta_o25_study = optuna.create_study(direction="minimize")
    meta_o25_study.optimize(meta_o25_objective, n_trials=20, n_jobs=1)
    best_meta_o25_C = meta_o25_study.best_params["C"]
    log.info("✅ O2.5 meta C=%.4f  best log-loss=%.4f", best_meta_o25_C, meta_o25_study.best_value)

meta_o25_model = LogisticRegression(
    solver="lbfgs", max_iter=1000, C=best_meta_o25_C, random_state=42,
)
meta_o25_model.fit(X_meta_o25, y_o25, sample_weight=sample_weights)

# Evaluate O2.5 meta
o25_meta_oof = _oof_predict(
    meta_o25_model, X_meta_o25, y_o25, cv=cv, sample_weight=sample_weights,
)
meta_o25_ll = log_loss(y_o25[:n_real], o25_meta_oof[:n_real, 1])
meta_o25_acc = accuracy_score(y_o25[:n_real], o25_meta_oof[:n_real].argmax(axis=1))
log.info("O2.5 meta OOF (real-only): accuracy=%.3f  log-loss=%.4f", meta_o25_acc, meta_o25_ll)


# ---------------------------------------------------------------------------
# PER-LEAGUE PERFORMANCE BREAKDOWN (V5.0: real-only)
# ---------------------------------------------------------------------------
league_metrics = {}
if n_real >= 100:
    df_real = df.iloc[:n_real].copy()
    df_real["pred"] = meta_oof[:n_real].argmax(axis=1)
    df_real["ok"]   = (df_real["pred"] == df_real["target"])
    for lg, grp in df_real.groupby("league"):
        if len(grp) >= 10 and lg:
            league_metrics[lg] = {"n": int(len(grp)), "accuracy": round(grp["ok"].mean(), 3)}
    log.info("Per-league breakdown: %d leagues", len(league_metrics))


# ---------------------------------------------------------------------------
# SHAP FEATURE IMPORTANCES (on XGB inner model)
# ---------------------------------------------------------------------------
shap_summary = {}
if SHAP_AVAILABLE and n_real >= 50:
    log.info("🔍 Computing SHAP importances...")
    try:
        inner = clf_match.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(inner)
        shap_vals = explainer.shap_values(X[:n_real])
        for ci, label in enumerate(["Away Win", "Draw", "Home Win"]):
            imps = np.abs(shap_vals[ci]).mean(axis=0)
            shap_summary[label] = {
                feat: round(float(v), 5)
                for feat, v in zip(FEATURE_COLS, imps)
            }
        log.info("SHAP done — top Home Win driver: %s",
                 max(shap_summary["Home Win"], key=shap_summary["Home Win"].get))
    except Exception as e:
        log.warning("SHAP failed: %s", e)
elif not SHAP_AVAILABLE:
    log.warning("shap not installed — skipping.")


# ---------------------------------------------------------------------------
# SAVE MODELS (V5.0: adds meta_scaler, lgb_o25, meta_o25)
# ---------------------------------------------------------------------------
joblib.dump(clf_match,         "xgb_match_model.joblib")
joblib.dump(clf_o25,           "xgb_o25_model.joblib")
joblib.dump(meta_model,        "meta_match_model.joblib")
joblib.dump(meta_scaler,       "meta_scaler.joblib")       # V5.0: StandardScaler for raw meta features
joblib.dump(FEATURE_COLS_O25,  "feature_cols_o25.joblib")  # 19-col goals feature list

saved_models = ["xgb_match", "xgb_o25", "meta_match", "meta_scaler", "feature_cols_o25"]

if clf_lgb_match is not None:
    joblib.dump(clf_lgb_match, "lgb_match_model.joblib")
    saved_models.append("lgb_match")

if clf_lgb_o25 is not None:
    joblib.dump(clf_lgb_o25, "lgb_o25_model.joblib")   # V5.0
    saved_models.append("lgb_o25")

if meta_o25_model is not None:
    joblib.dump(meta_o25_model, "meta_o25_model.joblib")  # V5.0
    saved_models.append("meta_o25")

log.info("💾 Saved: %s", ", ".join(saved_models))


# ---------------------------------------------------------------------------
# SAVE METRICS TO DB (V5.0: includes scaler info, O2.5 meta metrics)
# ---------------------------------------------------------------------------
metrics = {
    "model_version":             "v5.0",
    "trained_at":                pd.Timestamp.now().isoformat(),
    "n_features":                N_FEAT,
    "feature_cols":              FEATURE_COLS,
    "meta_raw_cols":             META_RAW_COLS,
    "n_meta_raw":                N_META_RAW,
    "train_records_real":        n_real,
    "train_records_total":       len(df),
    "synthetic_weight":          SYNTHETIC_WEIGHT,
    "time_decay_half_life_days": HALF_LIFE_DAYS,
    "temperature":               best_temperature,
    "optuna_used":               OPTUNA_AVAILABLE and n_real >= 150,
    "lgb_available":             LGB_AVAILABLE,
    "meta_stack_width":          X_meta.shape[1],
    "meta_C":                    best_meta_C,
    "meta_o25_C":                best_meta_o25_C,
    "xgb_calibration_method":    xgb_cal_method,
    "lgb_calibration_method":    lgb_cal_method if LGB_AVAILABLE else None,
    "xgb_lgb_diversity":         round(diversity, 4),
    "best_xgb_params":           best_xgb_p,
    "best_lgb_params":           best_lgb_p,
    "best_o25_params":           best_o25_p,
    "base_train_accuracy":       round(xgb_train_acc, 4),
    "base_train_logloss":        round(xgb_train_ll,  4),
    "meta_oof_accuracy":         round(meta_acc,  4),
    "meta_oof_logloss":          round(meta_ll,   4),
    "meta_oof_brier":            round(meta_brier, 4),
    "meta_oof_ece":              round(meta_ece,  4),
    "meta_o25_oof_logloss":      round(meta_o25_ll, 4) if meta_o25_ll else None,
    "meta_o25_oof_accuracy":     round(meta_o25_acc, 4),
    "random_baseline_logloss":   round(math.log(3), 4),
    "random_baseline_brier":     0.444,
    "league_metrics":            league_metrics,
}
save_kv("model_metrics", metrics)
if shap_summary:
    save_kv("shap_importances", shap_summary)

log.info("✅ V5.0 pipeline complete.")
log.info("   Real-only OOF accuracy : %.1f%%", meta_acc*100)
log.info("   Real-only OOF log-loss : %.4f  (random %.4f)", meta_ll, math.log(3))
log.info("   Real-only OOF Brier    : %.4f  (random 0.444)", meta_brier)
log.info("   Real-only OOF ECE      : %.4f  (perfect 0.000)", meta_ece)
log.info("   Temperature            : %.2f", best_temperature)
log.info("   Meta C                 : %.4f", best_meta_C)
log.info("   Stack width            : %d features", X_meta.shape[1])
log.info("   XGB/LGB diversity      : %.1f%%", diversity*100)
log.info("   Calibration            : XGB=%s  LGB=%s", xgb_cal_method,
         lgb_cal_method if LGB_AVAILABLE else "N/A")

