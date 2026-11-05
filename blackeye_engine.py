"""
BlackEye - blackeye_hybrid.py   (dual-engine hybrid: trains BOTH models, fuses them, evaluates, writes audit artifacts)
Run AFTER split.py and train_models.py (reads results_v3/results.json). blackeye_engine.py stays untouched.

 Engine 1  supervised   : LightGBM (best params) + Platt calibration            -> p_sup  (known fraud typologies)
 Engine 2  unsupervised : IsolationForest on behavioural features, NO labels     -> anomaly percentile a in [0,1]
 Fusion    transparent  : risk = 1 - (1 - p_sup) * (1 - LAMBDA * h(a)),  h = clip((a - A0) / (1 - A0), 0, 1)
                          -> the anomaly engine can lift a txn to STEP-UP, but never to BLOCK by itself (LAMBDA < T_BLOCK)
 Evaluation             : holdout (database_ready.csv) + ZERO-DAY SIMULATION (a fraud typology is removed from training
                          and the models are tested on exactly that typology: this is where Engine 2 must earn its place)
Install: pip install lightgbm scikit-learn pandas numpy joblib
"""
import hashlib
import json
import os
import time
import uuid
import warnings
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore", message=".*eval_set.*")

# ------------------------------------------------------------------ config
DATA_DIR, RESULTS, OUT, TARGET, SEED = "data", "results_v3/results.json", "blackeye_v2", "is_fraud", 42
MODELS_DIR = "models"   # production inference artifacts read by app.py (OUT is kept as an audit copy)
EXCLUDE_PREFIXES = ("age", "gender_", "prior_txn_count")      # same as blackeye_engine.py
R_STEP_UP, R_BLOCK = 20, 1
T_STEP, T_BLOCK = 1 / (1 + R_STEP_UP), 1 / (1 + R_BLOCK)
A0 = 0.999            # anomaly gate: only the top 0.1% most anomalous (vs train) can boost the risk
LAMBDA = 0.30         # max boost; must stay < T_BLOCK so anomaly alone => STEP-UP, never BLOCK
A0_GRID = [0.99, 0.995, 0.998, 0.999, 0.9995]
ANOM_CANDIDATES = ["transaction_amount", "amount_deviation_score", "amt_to_prior_mean", "distance_from_home_km",
                   "time_since_last_txn_hrs", "geo_velocity_proxy", "dist_from_prev_txn_km", "merchant_velocity_kmh",
                   "txn_count_1h", "amt_sum_1h", "txn_count_24h", "amt_sum_24h", "hour", "is_night"]
IF_TREES, IF_FIT_ROWS, REF_ROWS = 100, 500_000, 300_000
N_ZERO_DAY = 3        # number of fraud typologies (merchant categories) to remove in the simulation
TOP_K, N_LOG_EACH, N_LATENCY = 5, 50, 500
assert LAMBDA < T_BLOCK and T_STEP < T_BLOCK
LOG = []
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def flush_log():
    os.makedirs(OUT, exist_ok=True)       # OneDrive can remove/lock folders during long runs
    open(f"{OUT}/report.txt", "w", encoding="utf-8").write("\n".join(LOG))


sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
ap, auc = average_precision_score, roc_auc_score


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ load
res = json.load(open(RESULTS))
F = [f for f in res["kept_features"] if not f.startswith(EXCLUDE_PREFIXES)]
ANOM = [f for f in ANOM_CANDIDATES if f in F]
bp = {k: (int(v) if k in ("num_leaves", "min_child_samples") else v) for k, v in res["best_params"].items()}
rd = lambda n: pd.read_csv(os.path.join(DATA_DIR, n), usecols=F + [TARGET])
tr, va, db = rd("train_ready.csv"), rd("validation_ready.csv"), rd("database_ready.csv")
X_tr, y_tr, X_va, y_va, X_db, y_db = tr[F], tr[TARGET].values, va[F], va[TARGET].values, db[F], db[TARGET].values
say(f"train {X_tr.shape} | val {X_va.shape} | holdout {X_db.shape} | supervised features {len(F)} | anomaly features {len(ANOM)}")
say(f"routing: STEP-UP >= {T_STEP:.4f} (R={R_STEP_UP}) | BLOCK >= {T_BLOCK:.4f} (R={R_BLOCK}) | fusion: A0={A0}, LAMBDA={LAMBDA}")

MU, SD = np.zeros(len(F)), np.ones(len(F))
sp = os.path.join(DATA_DIR, "scaler.json")
RAW_OK = os.path.exists(sp)
if RAW_OK:
    sc = json.load(open(sp))
    pos = {c: i for i, c in enumerate(sc["cols"])}
    for j, f in enumerate(F):
        if f in pos:
            MU[j], SD[j] = sc["mean"][pos[f]], sc["scale"][pos[f]]
Fidx = {f: i for i, f in enumerate(F)}

PARAMS = dict(n_estimators=4000, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=10.0, n_jobs=-1, verbosity=-1, metric="average_precision", random_state=SEED)
PARAMS.update(bp)


def fit_lgb(Xa, ya, Xv, yv, **over):
    m = lgb.LGBMClassifier(**{**PARAMS, **over})
    m.fit(Xa, ya, eval_set=[(Xv, yv)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    return m


# ------------------------------------------------------------------ ENGINE 1: supervised
say("\n===== ENGINE 1: SUPERVISED (LightGBM + Platt) =====")
model = fit_lgb(X_tr, y_tr, X_va, y_va)
booster = model.booster_
pr_va, pr_db = model.predict_proba(X_va)[:, 1], model.predict_proba(X_db)[:, 1]
platt = LogisticRegression(C=1e6, max_iter=1000).fit(lg(pr_va).reshape(-1, 1), y_va)
cal = lambda p: platt.predict_proba(lg(p).reshape(-1, 1))[:, 1]
ps_va, ps_db = cal(pr_va), cal(pr_db)
MODEL_HASH = hashlib.sha256(booster.model_to_string().encode()).hexdigest()
MODEL_VERSION = f"blackeye-hybrid-{datetime.now(timezone.utc):%Y%m%d}-{MODEL_HASH[:8]}"
say(f"{MODEL_VERSION} | trees {model.best_iteration_}")
say(f"val     PR-AUC {ap(y_va, ps_va):.4f} | ROC-AUC {auc(y_va, ps_va):.4f}")
say(f"holdout PR-AUC {ap(y_db, ps_db):.4f} | ROC-AUC {auc(y_db, ps_db):.4f}")

# ------------------------------------------------------------------ ENGINE 2: unsupervised
say("\n===== ENGINE 2: UNSUPERVISED (IsolationForest, labels never used) =====")
rs = np.random.RandomState(SEED)
Xa_tr, Xa_va, Xa_db = X_tr[ANOM].values, X_va[ANOM].values, X_db[ANOM].values
iforest = IsolationForest(n_estimators=IF_TREES, max_samples=256, random_state=SEED, n_jobs=-1)
iforest.fit(Xa_tr[rs.choice(len(Xa_tr), min(IF_FIT_ROWS, len(Xa_tr)), replace=False)])
raw_anom = lambda X: -iforest.score_samples(X)                  # higher = more anomalous
ref_idx = rs.choice(len(Xa_tr), min(REF_ROWS, len(Xa_tr)), replace=False)
REF = np.sort(raw_anom(Xa_tr[ref_idx]))                          # train score distribution -> percentile
SORTED_FEAT = np.sort(Xa_tr[ref_idx], axis=0)                    # per-feature train distribution -> rarity explanation
pct = lambda s: np.searchsorted(REF, s, side="right") / len(REF)
iforest.set_params(n_jobs=1)                                     # single-row scoring in production path
a_va, a_db = pct(raw_anom(Xa_va)), pct(raw_anom(Xa_db))
say(f"unsupervised alone, holdout: PR-AUC {ap(y_db, a_db):.4f} (base rate {y_db.mean():.4f}) | ROC-AUC {auc(y_db, a_db):.4f}")
for q in (0.99, 0.999):
    m_ = a_db >= q
    say(f"  top {100 * (1 - q):.1f}% most anomalous: {m_.sum():,} txns | fraud rate {y_db[m_].mean():.2%} | catches {y_db[m_].sum() / y_db.sum():.1%} of all fraud")
ov = np.corrcoef(a_db, ps_db)[0, 1]
say(f"rank-correlation-like overlap with supervised score (Pearson on values): {ov:.3f}  (low = complementary)")
flush_log()


# ------------------------------------------------------------------ FUSION
def fuse(p, a, a0=None, lam=LAMBDA):
    h = np.clip((a - (A0 if a0 is None else a0)) / (1 - (A0 if a0 is None else a0)), 0, 1)
    return 1 - (1 - p) * (1 - lam * h)


route_of = lambda p: np.where(p >= T_BLOCK, "BLOCK", np.where(p >= T_STEP, "STEP-UP", "APPROVE"))


def alert_stats(p, y):
    al = p >= T_STEP
    bl = p >= T_BLOCK
    return dict(alerts_1000=al.mean() * 1000, precision=y[al].mean() if al.any() else np.nan, recall=y[al].sum() / max(y.sum(), 1),
                block_1000=bl.mean() * 1000, block_prec=y[bl].mean() if bl.any() else np.nan, block_recall=y[bl].sum() / max(y.sum(), 1),
                frauds_caught=int(y[al].sum()), alerts=int(al.sum()))


say("\n===== FUSION TRADE-OFF: what the anomaly engine adds on top of supervised (extra alerts vs extra frauds) =====")
say("(in-distribution this is expected to be small; its real value is the zero-day simulation below)")
rows = []
for nm, p_s, a, y in (("validation", ps_va, a_va, y_va), ("holdout", ps_db, a_db, y_db)):
    base = alert_stats(p_s, y)
    for a0 in A0_GRID:
        f = alert_stats(fuse(p_s, a, a0), y)
        ea, ef = f["alerts"] - base["alerts"], f["frauds_caught"] - base["frauds_caught"]
        rows.append((nm, a0, ea, ef, ef / ea if ea else np.nan, f["alerts_1000"], f["precision"], f["recall"]))
say(pd.DataFrame(rows, columns=["set", "A0", "extra_alerts", "extra_frauds", "incr_precision", "alerts/1000", "precision", "recall"])
    .round(4).to_string(index=False))
p_fu_va, p_fu_db = fuse(ps_va, a_va), fuse(ps_db, a_db)


# ------------------------------------------------------------------ SAVE ARTIFACTS (right after both engines + fusion exist,
# BEFORE the slow zero-day / audit / latency sections, so an interrupted run still leaves a usable model)
def save_artifacts(*dirs):
    card = {"model_version": MODEL_VERSION, "supervised_sha256": MODEL_HASH, "features": F, "anomaly_features": ANOM,
            "excluded_features": list(EXCLUDE_PREFIXES), "trees": int(model.best_iteration_),
            "fusion": {"formula": "risk = 1-(1-p_sup)*(1-LAMBDA*clip((a-A0)/(1-A0),0,1))", "A0": A0, "LAMBDA": LAMBDA,
                       "note": "LAMBDA < T_BLOCK: anomaly engine alone can only trigger STEP-UP"},
            "routing": {"STEP-UP": f">= {T_STEP:.6f} (R={R_STEP_UP})", "BLOCK": f">= {T_BLOCK:.6f} (R={R_BLOCK})"},
            "holdout": {"supervised_pr_auc": float(ap(y_db, ps_db)), "fused_pr_auc": float(ap(y_db, p_fu_db)),
                        "anomaly_alone_pr_auc": float(ap(y_db, a_db)), "base_rate": float(y_db.mean()), "n": int(len(y_db))},
            "limitations": ["Synthetic (Sparkov) data - do not quote accuracy as production performance",
                            "Zero-day simulation uses merchant categories as proxy typologies",
                            "Re-fit Platt, A0, LAMBDA and thresholds on production-like data before go-live"]}
    bundle = {"platt": platt, "iforest": iforest, "anomaly_features": ANOM, "kept_features": F,
              "T_STEP": T_STEP, "T_BLOCK": T_BLOCK, "A0": A0, "LAMBDA": LAMBDA}
    for d in dirs:
        os.makedirs(d, exist_ok=True)                       # OneDrive can remove/lock folders during long runs
        booster.save_model(f"{d}/supervised_model.txt")
        joblib.dump(bundle, f"{d}/engines.joblib")
        np.savez_compressed(f"{d}/anomaly_reference.npz", score_ref=REF, feature_ref=SORTED_FEAT)
        json.dump(card, open(f"{d}/model_card.json", "w"), indent=2)
    say(f"artifacts saved -> {', '.join(dirs)}  ({MODEL_VERSION})")


save_artifacts(MODELS_DIR, OUT)
flush_log()

say("\n===== HOLDOUT: supervised only vs FUSED (current A0) =====")
for nm, p in (("supervised only", ps_db), ("fused (hybrid)", p_fu_db)):
    s = alert_stats(p, y_db)
    say(f"  {nm:<16} PR-AUC {ap(y_db, p):.4f} | alerts/1000 {s['alerts_1000']:.2f} | precision {s['precision']:.4f} | recall {s['recall']:.4f} | "
        f"BLOCK/1000 {s['block_1000']:.2f} (precision {s['block_prec']:.4f}, recall {s['block_recall']:.4f})")
rt_s, rt_f = route_of(ps_db), route_of(p_fu_db)
say(f"  decisions changed by the anomaly engine: {int((rt_s != rt_f).sum())} of {len(y_db):,} "
    f"(frauds among them: {int(y_db[rt_s != rt_f].sum())})")
flush_log()

# ------------------------------------------------------------------ ZERO-DAY SIMULATION
say(f"\n===== ZERO-DAY SIMULATION (a fraud typology is REMOVED from training; test only on that typology in the holdout) =====")
say("a typology = fraud inside one merchant category; its legit rows stay in training (worst case: model learns 'this category is clean')")
cat_cols = [c for c in F if c.startswith("category_")]
top = sorted(cat_cols, key=lambda c: -int(y_db[X_db[c].values == 1].sum()))[:N_ZERO_DAY]
zd_rows = []
for col in top:
    dtr, dva = (X_tr[col].values == 1) & (y_tr == 1), (X_va[col].values == 1) & (y_va == 1)
    mz = fit_lgb(X_tr[~dtr], y_tr[~dtr], X_va[~dva], y_va[~dva], learning_rate=0.1)
    pl = LogisticRegression(C=1e6, max_iter=1000).fit(lg(mz.predict_proba(X_va[~dva])[:, 1]).reshape(-1, 1), y_va[~dva])
    pz = pl.predict_proba(lg(mz.predict_proba(X_db)[:, 1]).reshape(-1, 1))[:, 1]
    mk = X_db[col].values == 1
    yk, nf = y_db[mk], int(y_db[mk].sum())
    say(f"-- typology '{col[9:]}': removed {int(dtr.sum())} train frauds | holdout rows {int(mk.sum()):,} with {nf} frauds")
    for nm, p in (("reference (model saw this typology)", ps_db), ("supervised, typology unseen", pz),
                  ("anomaly engine alone (a >= A0)", np.where(a_db >= A0, 1.0, 0.0)), ("FUSED, typology unseen", fuse(pz, a_db))):
        al = p[mk] >= T_STEP
        say(f"   {nm:<36} recall {yk[al].sum() / max(nf, 1):.3f} | alerts/1000 in category {al.mean() * 1000:7.2f} | "
            f"precision {yk[al].mean() if al.any() else float('nan'):.3f} | PR-AUC {ap(yk, p[mk]):.4f}")
    zd_rows.append((col[9:], nf))
    flush_log()

# ------------------------------------------------------------------ per-decision record (both engines) + audit chain
def label(f):
    if f.startswith("category_"):
        return "Merchant category = " + f[9:]
    return {"transaction_amount": "Transaction amount", "amount_deviation_score": "Amount deviation vs card baseline",
            "amt_to_prior_mean": "Amount / card's average amount", "distance_from_home_km": "Distance home-merchant (km)",
            "is_first_txn": "First txn seen for card", "time_since_last_txn_hrs": "Hours since previous txn",
            "geo_velocity_proxy": "Geo-velocity proxy", "dist_from_prev_txn_km": "Distance from previous txn (km)",
            "merchant_velocity_kmh": "Travel speed between txns (km/h)", "txn_count_1h": "Txns in last 1h",
            "amt_sum_1h": "Spend in last 1h", "txn_count_24h": "Txns in last 24h", "amt_sum_24h": "Spend in last 24h",
            "hour": "Hour of day", "day_of_week": "Day of week", "is_night": "Night-time txn (22-03h)",
            "log_city_pop": "Log city population"}.get(f, f)


def score_one(x):
    """x: standardized feature vector (len F). One audit-ready decision from BOTH engines."""
    t0 = time.perf_counter()
    c = booster.predict(x.reshape(1, -1), pred_contrib=True, num_threads=1)[0]
    p_raw = float(sigmoid(c.sum()))
    p_sup = float(cal(np.array([p_raw]))[0])
    xa = x[[Fidx[f] for f in ANOM]]
    a = float(pct(-iforest.score_samples(xa.reshape(1, -1)))[0])
    p_fu = float(fuse(np.array([p_sup]), np.array([a]))[0])
    route, route_sup = str(route_of(np.array([p_fu]))[0]), str(route_of(np.array([p_sup]))[0])
    phi = c[:-1]
    tot = np.abs(phi).sum() + 1e-12
    raw = x * SD + MU
    sup_reasons = [{"feature": F[i], "label": label(F[i]), "value": round(float(raw[i]), 4), "impact_logodds": round(float(phi[i]), 4),
                    "direction": "increases_risk" if phi[i] > 0 else "decreases_risk",
                    "share_of_total_impact": round(float(abs(phi[i]) / tot), 4)} for i in np.argsort(-np.abs(phi))[:TOP_K]]
    tails = []
    for j, f in enumerate(ANOM):
        col = SORTED_FEAT[:, j]
        fr = (np.searchsorted(col, xa[j], "left") + np.searchsorted(col, xa[j], "right")) / 2 / len(col)
        tails.append(2 * min(fr, 1 - fr))
    anom_reasons = [{"feature": ANOM[j], "label": label(ANOM[j]), "value": round(float(raw[Fidx[ANOM[j]]]), 4),
                     "two_sided_tail_probability_vs_train": round(float(tails[j]), 6)} for j in np.argsort(tails)[:3]]
    driver = "anomaly_override" if route != route_sup else ("anomaly_assist" if a >= A0 else "supervised")
    thr = T_BLOCK if route == "BLOCK" else T_STEP
    return {"decision_id": str(uuid.uuid4()), "ts_utc": datetime.now(timezone.utc).isoformat(), "model_version": MODEL_VERSION,
            "risk_score": round(100 * p_fu, 2), "route": route, "decision_driver": driver,
            "engine_supervised": {"p_calibrated": round(p_sup, 6), "route_if_alone": route_sup, "top_reasons": sup_reasons},
            "engine_anomaly": {"percentile_vs_train": round(a, 6), "gate_A0": A0, "boost_active": bool(a > A0),
                               "rarest_features": anom_reasons},
            "decision_confidence": round(1 - p_fu if route == "APPROVE" else p_fu, 4),
            "route_margin_logodds": round(float(abs(np.log((p_fu + 1e-9) / (1 - p_fu + 1e-9)) - np.log(thr / (1 - thr)))), 4),
            "thresholds": {"step_up": round(T_STEP, 6), "block": round(T_BLOCK, 6)},
            "features_sha256": hashlib.sha256(x.astype(np.float64).tobytes()).hexdigest(),
            "scoring_ms": round((time.perf_counter() - t0) * 1000, 3)}


say("\n===== AUDIT TRAIL (hash-chained JSONL, both engines' reasons) =====")
LOGP = f"{OUT}/decisions.jsonl"
open(LOGP, "w").close()
prev = "GENESIS"


def log_decision(rec):
    global prev
    rec = dict(rec, prev_hash=prev)
    rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    open(LOGP, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
    prev = rec["hash"]


def verify_chain(path):
    p = "GENESIS"
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        rec = json.loads(line)
        h = rec.pop("hash")
        if rec["prev_hash"] != p or hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest() != h:
            return False, n
        p = h
    return True, n


Xv = X_db.values
picks = [np.where(rt_f == r)[0] for r in ("BLOCK", "STEP-UP", "APPROVE")] + [np.where(rt_s != rt_f)[0]]
for idx in picks:
    for i in (rs.choice(idx, min(N_LOG_EACH, len(idx)), replace=False) if len(idx) else []):
        rec = score_one(Xv[i])
        rec["ground_truth_is_fraud"] = int(y_db[i])        # offline evaluation only; absent in production
        log_decision(rec)
ok, n = verify_chain(LOGP)
say(f"{n} decisions logged -> {LOGP} | chain verification: {'PASS' if ok else 'FAIL at line ' + str(n)}")
ov_i = picks[3]
if len(ov_i):
    ex = score_one(Xv[ov_i[0]])
    say("example anomaly-driven decision:\n" + json.dumps({k: ex[k] for k in ("route", "risk_score", "decision_driver", "engine_supervised", "engine_anomaly")}, indent=1))
    say(f"(that transaction is_fraud = {int(y_db[ov_i[0]])})")

# ------------------------------------------------------------------ latency + artifacts
say("\n===== LATENCY (single txn, both engines + SHAP + record; excludes feature computation) =====")
lat = []
for i in rs.choice(len(Xv), N_LATENCY, replace=False):
    t0 = time.perf_counter()
    score_one(Xv[i])
    lat.append((time.perf_counter() - t0) * 1000)
say(f"p50 {np.percentile(lat, 50):.2f} ms | p95 {np.percentile(lat, 95):.2f} ms | p99 {np.percentile(lat, 99):.2f} ms | target < 100 ms")

flush_log()
print(f"\nDone. Send me {OUT}/report.txt")
