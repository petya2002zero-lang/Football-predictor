# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate the venv first (always required)
source venv/Scripts/activate          # Windows/bash
# or: venv\Scripts\activate.bat       # Windows/cmd

# 1. Validate the database before any data fetch
python check_db.py

# 2. Fetch fresh data from API-Football (requires API_KEY env var)
API_KEY=your_key python train_master.py

# 3. Retrain all ML models
python train_ml.py

# 4. Run the dashboard
streamlit run dashboard.py
```

No test suite exists. Validate changes by running `python -m py_compile <file>.py` before executing.

## Gitignored Files

`.gitignore` excludes `venv/`, `__pycache__/`, `*.sqlite`, `*.joblib`, `*.7z`, `hf_data.json`, and `.env`. These files are generated locally by the pipeline and should not be committed manually. The CI workflow uses `git add -f` to force-commit `database.sqlite` and `.joblib` model files on the `main` branch.

## Utility Scripts

- **`check_db.py`** — Runs SQLite `PRAGMA integrity_check`; deletes `database.sqlite` if corrupted so the next `train_master.py` run rebuilds it cleanly.
- **`export_hf_data.py`** — Reads the entire `kv_store` table, excludes `bet_log`, and writes `hf_data.json` for Hugging Face Spaces sync (SQLite exceeds HF's 10 MB limit).
- **`backtest.py`** — Offline ROI analysis against `recent` results. Loads all models and kv_store keys, reconstructs feature vectors, runs the full prediction stack, and prints ROI tables by tier and pick type. CLI: `python backtest.py [--min-tier diamond] [--days 30]`. Uses current ratings as a proxy for historical values — look-ahead bias applies to older records.
- **`data_quality_check.py`** — Inspects data completeness without importing `train_master.py`. Reports: (1) pro-preds feature coverage for upcoming matches (% fully-populated vs partial), (2) upcoming matches missing Pinnacle odds (`pin_implied_h=0.0`), (3) `hist_odds_cache` coverage for recent results plus an estimate of runs needed for full coverage. CLI: `python data_quality_check.py`.

## Architecture

The pipeline has three sequential stages that must run in order:

### Stage 1 — `train_master.py` (Data Engine)
Fetches from API-Football v3 and writes everything to `database.sqlite` via `save_kv(key, data)`. The DB is a flat key-value store (`kv_store` table); all values are JSON blobs. Key output keys:

| kv_store key | Contents |
|---|---|
| `recent` | Last 90 days of completed fixtures (2000+ records) — training labels |
| `upcoming` | Next ~10 days of fixtures — inference targets |
| `pro_preds` | Pre-computed features for upcoming 150 matches |
| `hist_odds_cache` | Historical Pinnacle odds keyed by `fixture_id` (fills 120/run) |
| `pi_ratings` | Elo/Pi ratings per team + `{team}_safety`, `{team}_title`, `{team}_xpts_delta` |
| `league_averages` | Per-league goal averages and H/D/A rates |
| `insights` | Per-fixture injury, odds, sharp-action data |
| `standings`, `team_forms`, `h2h` | Supporting lookup data |

API responses are cached in the `api_cache` table with `If-Modified-Since` headers. The soft quota limit is 7000 requests/day (hard limit 7500). `_quota_used` is a global counter that gates all live requests.

The pipeline runs in numbered sections:
1. Recent results (adds `fixture_id` to each record — required for `hist_odds_cache`)
2. Pi ratings (Elo fallback if `penaltyblog` unavailable)
3. Standings → `league_averages`, `team_forms`, `pi_ratings` (safety/title/xpts extensions)
4. Team stats (weekly-cached per team+league cycle key; `get_team_stats(team_id, league_id, fixture_date=None)` accepts an optional `fixture_date` — if the cached `last_date` is within 3 days of the fixture, the entry is invalidated and re-fetched automatically)
5. Match insights — H2H, odds, injuries for upcoming matches
5.5. Historical odds — fetches Pinnacle odds for 120 most-recent unfetched completed fixtures
6. Pro predictions — compiles full feature dict per upcoming match into `pro_preds`

### Stage 2 — `train_ml.py` (ML Pipeline — V5.0)
Reads from `database.sqlite`, builds a 3-layer ensemble, saves 7 `.joblib` files.

**Feature contract:** `FEATURE_COLS` (29 features), `META_RAW_COLS` (10 features), and `FEATURE_COLS_O25` (19 features) are defined at the top of both `train_ml.py` and `dashboard.py` and **must match exactly**. Changing any list requires updating both files simultaneously.

`FEATURE_COLS` includes 3 derived defensive features computed from existing `pro_preds` values — **no `train_master.py` change needed** for derived features:
- `cs_diff` = `cs_h - cs_a` (net home defensive edge)
- `defensive_dominance` = `(cs_h + fts_a) / 2` (P home keeps clean sheet)
- `attacking_vulnerability` = `(cs_a + fts_h) / 2` (P away keeps clean sheet)

`FEATURE_COLS_O25` is a separate 19-feature set used only by the O2.5 models — it drops table-position and 1X2-distribution features irrelevant to total goals. The O2.5 models train on `X_o25 = df[FEATURE_COLS_O25]`, not `X`.

**Training data assembly:** Each record in `recent` is matched to `pro_preds` by `"{home} vs {away}"` key for team-specific features. Pinnacle odds come from `hist_odds_cache[str(fixture_id)]`. Market edge is computed in `train_ml.py` using `get_dc_probs()`.

**Model stack (19-feature meta):**
```
DC probs(3) + XGB probs(3) + LGB probs(3) + StandardScaler(META_RAW_COLS)(10) = 19
```

**Saved files and what breaks without them:**

| File | Role | Missing effect |
|---|---|---|
| `xgb_match_model.joblib` | XGBoost 3-class (Away=0, Draw=1, Home=2) | ML disabled |
| `lgb_match_model.joblib` | LightGBM 3-class | Falls back to XGB-only |
| `meta_match_model.joblib` | LogReg meta-learner (19 features) | Falls back to DC blend |
| `meta_scaler.joblib` | StandardScaler for META_RAW_COLS | Raw features unscaled → degraded meta |
| `xgb_o25_model.joblib` | XGBoost binary O2.5 (19-feature FEATURE_COLS_O25) | O2.5 ML disabled |
| `lgb_o25_model.joblib` | LightGBM binary O2.5 | Falls back to XGB-only O2.5 |
| `meta_o25_model.joblib` | LogReg O2.5 meta | Falls back to DC/XGB blend |
| `feature_cols_o25.joblib` | Saved FEATURE_COLS_O25 list | backtest.py uses hardcoded fallback |

All models use `CalibratedClassifierCV` wrapping the base learner. Compatibility patches at the top of `train_ml.py` exist:
- `xgboost==2.0.3` dropped `ClassifierMixin` → `__sklearn_tags__` patch for sklearn 1.8+
- sklearn 1.4+ removed `fit_params` from `cross_val_predict` → manual 5-fold OOF loop (`_oof_predict`)
- sklearn Cython calibration (`CyHalfBinomialLoss`) requires all inputs to match `X` dtype (`float32`) — LightGBM predict_proba wrapped in `_LGBMFloat32`; `sample_weights` array must also be `dtype=np.float32`

### Stage 3 — `dashboard.py` (Streamlit UI)
Loads all 7 `.joblib` files and all `kv_store` keys at startup. Inference for each match follows:
1. Build 29-feature vector (`FEATURE_COLS`) from `pro_preds` + `pi_ratings` + `team_forms` + `insights`; includes 3 derived defensive features computed inline from `cs_h/cs_a/fts_h/fts_a`
2. Get XGB probs, LGB probs, Dixon-Coles probs
3. Scale `META_RAW_COLS` via `meta_scaler`
4. Concatenate → 19-feature vector → `meta_match_model` → temperature scaling → final probs
5. O2.5: separate 19-feature `features_o25` (`FEATURE_COLS_O25`) → `xgb_o25_model` → blended with DC O2.5

**`get_match_math(m)` caching:** `_cached_match_math(m)` wraps `get_match_math` with a `st.session_state` dict invalidated by `id(data["upcoming"])`. All loop call sites use `_cached_match_math` — never call `get_match_math` in a loop directly.

**Display constants:** `DEFAULT_LEAGUES` (18 leagues) auto-selected in sidebar. Tier thresholds in `render_match_card()`: Emerald ≥85% (≥90% if no true xG), Diamond+ ≥75%, Diamond ≥65%, Gold <65%. Timezone hardcoded to `Europe/Budapest` in `format_time()`.

## Critical Constraints

- **FEATURE_COLS, FEATURE_COLS_O25, and META_RAW_COLS must be identical** in `train_ml.py` and `dashboard.py`. Verify with:
  ```bash
  python -c "
  from dashboard import FEATURE_COLS, FEATURE_COLS_O25
  from train_ml import FEATURE_COLS as F2, FEATURE_COLS_O25 as O2
  assert F2==FEATURE_COLS, f'FEATURE_COLS mismatch: {set(F2)^set(FEATURE_COLS)}'
  assert O2==FEATURE_COLS_O25, f'FEATURE_COLS_O25 mismatch: {set(O2)^set(FEATURE_COLS_O25)}'
  "
  ```
- **Model class order is Away=0, Draw=1, Home=2** — `probs[2]` is home win, not `probs[0]`.
- **Never modify `database.sqlite` schema** — the kv_store is append/replace only. Never drop tables.
- **Never overwrite a model file** — always retrain from scratch with `train_ml.py`; it overwrites atomically via `joblib.dump`.
- **`api_cache` is the quota guard** — historical API responses are cached indefinitely by `endpoint||params` key. Clearing it will burn quota on the next `train_master.py` run.
- **`bet_log` is private** — `export_hf_data.py` explicitly excludes it when syncing to Hugging Face Spaces.
- **`hist_odds_cache` fills gradually** — 120 fixtures per `train_master.py` run, newest first. After ~18 runs all 2100+ training records will have real Pinnacle features.
- **All data reads must try SQLite first, fall back to `hf_data.json`** — `load_kv()` handles this. Never add a data source that only works locally.
- **Dixon-Coles rho is league-specific** — use `LEAGUE_RHO.get(league, DEFAULT_RHO)`, never hardcode `-0.130`.

## API Quota Rules

`QUOTA_SOFT_LIMIT = 7000`. The inner guard in `fetch_api()` is the last resort — add loop-level guards too:

```python
# In loops over many items:
for item in items:
    if _quota_used >= QUOTA_SOFT_LIMIT:
        break
    fetch_api(...)

# In helper functions called per-match:
def get_last_match_date(team_id):
    if _quota_used >= QUOTA_SOFT_LIMIT:
        return <default>
    ...fetch_api(...)

# For optional sub-calls within a helper, leave headroom:
if _quota_used < QUOTA_SOFT_LIMIT - 200:
    fetch_api(...)   # expensive optional call
```

`get_last_match_date()`, `is_new_manager()`, `has_exodus()`, and `get_team_stats()` all have early-return quota guards. Any new helper that calls `fetch_api()` must follow the same pattern.

## xG Formula Constraints (Section 6 of train_master.py)

The geometric blend formula `(attack_avg × defence_avg) / league_avg` can produce extreme values without these guards — do not remove them:

```python
# Ceiling of 5.0 on the geometric blend output
h_xg = max(0.2, min(5.0, (h_scored_home * a_conceded_away) / max(0.5, l_h_avg)))

# Pi-rating adjustment clamped to ±1.5 goals in the fallback path
pi_adj = max(-1.5, min(1.5, (h_pi - a_pi) / 2.0))

# true_xg=0.0 is a valid value — use `is not None`, not truthiness
if h_true_xg is not None and h_xg_count > 0:

# Floor xG before Dixon-Coles call (Poisson requires λ > 0)
h_xg = max(0.10, h_xg); a_xg = max(0.10, a_xg)
_model_h, _model_d, _model_a = _dc_home_prob(h_xg, a_xg, _rho)

# Final floor consistent with initial floor
h_xg = max(0.20, h_xg); a_xg = max(0.20, a_xg)
```

## Adding a New ML Feature

**Raw feature** (fetched from API — requires train_master.py change):
```python
# 1. Compute in train_master.py (section 6, inside the match loop):
pro_predictions[match_key]["new_feature"] = round(new_feature, 4)

# 2. Add to FEATURE_COLS in BOTH train_ml.py and dashboard.py

# 3. Extract in train_ml.py dataset.append():
"new_feature": float(pp.get("new_feature", 0.0)),

# 4. Provide synthetic fallback in the bootstrap section of train_ml.py:
"new_feature": float(rng.uniform(min_val, max_val)),
```

**Derived feature** (computed from existing pro_preds values — no train_master.py change needed):
```python
# 1. Add to FEATURE_COLS in BOTH train_ml.py and dashboard.py

# 2. In train_ml.py dataset.append() — compute inline:
"new_derived": float(pp.get("base_a", 0.0)) - float(pp.get("base_b", 0.0)),

# 3. In train_ml.py synthetic bootstrap — pre-compute base values then derive:
"base_a": (_s_a := float(rng.beta(3,7))),
"base_b": (_s_b := float(rng.beta(2,7))),
"new_derived": _s_a - _s_b,

# 4. In dashboard.py get_match_math() — compute from already-loaded variables
#    before the pd.DataFrame([[...]]) build:
new_derived = base_a - base_b
```

**O2.5-only feature** — add to `FEATURE_COLS_O25` (not `FEATURE_COLS`) in both files; uses the `X_o25` matrix in train_ml.py and `features_o25` DataFrame in dashboard.py.

## Common Gotchas

- **`recent` records must have `fixture_id`** — required for `hist_odds_cache` lookup in `train_ml.py`. Records missing it silently get default odds (0.0).
- **`team_forms` keys are bare team IDs** (integers) for some lookups and bare team name strings for others — check which the calling code expects.
- **League name matching uses API strings** (e.g., `"Premier League"`, not abbreviations) for xG blending and league averages lookup.
- **Form string padding** — missing results use `"?"`; `form_pts()` returns `5.0` (neutral) for unknown characters.
- **Pi ratings fallback** — `_find_pi_ratings_class()` scans penaltyblog across package versions; if unavailable or fit fails, falls back to `compute_elo_ratings()` (K=32, home_adv=60, initial=1000).
- **Pinnacle odds** — `_extract_odds()` prefers bookie id=4 (Pinnacle), falls back to id=8 (Bet365); returns `(0.0, 0.0, 0.0)` if neither found.
- **sklearn calibration dtype** — `sample_weights` (and any array passed to `CalibratedClassifierCV.fit`) must be `np.float32` to match `X` dtype. Cython `CyHalfBinomialLoss.loss_gradient` raises `ValueError: Buffer dtype mismatch` on float64/float32 mismatches.

## CI/CD

`.github/workflows/daily_bot.yml` runs twice daily (07:00 and 13:00 UTC):
- Always runs `train_master.py` and force-commits `database.sqlite` (gitignored locally, `git add -f` in CI)
- Runs `train_ml.py` and force-commits all 7 `.joblib` files only when triggered with `retrain_models=true`
- Syncs to Hugging Face Spaces via force-push to `hf-deploy` branch (uses `export_hf_data.py` to export `hf_data.json` instead of the SQLite file, which exceeds HF's 10 MB limit)

Secrets required: `API_KEY` (API-Football), `HF_TOKEN` (Hugging Face).

## Testing Checklist Before Pushing

- [ ] `python check_db.py` passes
- [ ] Feature contract check passes (see Critical Constraints)
- [ ] `python -m py_compile train_master.py train_ml.py dashboard.py backtest.py`
- [ ] `streamlit run dashboard.py` loads without errors and at least one match card renders correctly
- [ ] Model health page shows metrics (not "No model metrics found")
- [ ] `python backtest.py --min-tier diamond` runs without errors (needs populated DB)

## Performance Targets

V5.0 baseline (2026-03-09 retrain): accuracy 55.7%, log-loss 0.9318, Brier 0.1843. Regressions below these numbers indicate a problem.

- Meta OOF accuracy: > 55% (random baseline ~33%)
- Meta OOF log-loss: < 0.95 (random baseline ln(3) ≈ 1.099)
- Meta OOF Brier: < 0.20 (random baseline ~0.444)
- API requests per run: < 6500 (keep 500 buffer below 7000 soft limit)
