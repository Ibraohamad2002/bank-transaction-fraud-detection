"""
BlackEye - train_models_v3.py   (train + validation only, database NOT used)
Goal: squeeze what is left in the data (honestly: expect SMALL gains) and make the operating
point a business decision instead of "max F1".
  A. Ablation        : does dropping a one-hot group (country/city/payment/device) hurt PR-AUC?
  B. Random search   : N_SEARCH LightGBM configs, early stopping on average_precision only
  C. Seed bagging    : N_SEEDS models of the best config, probabilities averaged
  D. LR blend        : rank-blend bagged LightGBM with Logistic Regression (v2: LR was 0.118 vs LGBM 0.128)
  E. Calibration     : decile table (is probability ~ real fraud rate?)
  F. Cost thresholds : best alert threshold for each cost ratio  C_missed_fraud / C_false_alert
All differences are reported with a paired-bootstrap 95% CI: if the CI contains 0, treat it as noise.
Install: pip install lightgbm scikit-learn pandas numpy matplotlib
"""
import json
import os
import warnings

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

warnings.filterwarnings("ignore", message=".*eval_set.*")

# ------------------------------------------------------------------ config
DATA_DIR = "data"            # "." if the csv files are next to this script
TARGET = "is_fraud"
SEED = 42
RUN_ABLATION = True
N_SEARCH = 8                 # random-search trials (each = 1 LightGBM fit)
N_SEEDS = 3                  # bagging size
N_BOOT = 100                 # bootstrap resamples for confidence intervals
COST_RATIOS = [1, 5, 10, 20, 50, 100]   # C_missed_fraud / C_false_alert
GROUPS = {"state": "state_", "category": "category_", "gender": "gender_"}
OUT = "results_v3"
MODELS_DIR = "models"          # trained artifacts for inference (app.py / blackeye_engine.py)
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

BASE = dict(n_estimators=4000, learning_rate=0.05, num_leaves=15, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
            n_jobs=-1, verbosity=-1, metric="average_precision")

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------------------ load
train = pd.read_csv(os.path.join(DATA_DIR, "train_ready.csv"))
val = pd.read_csv(os.path.join(DATA_DIR, "validation_ready.csv"))
features = [c for c in train.columns if c != TARGET]
X_tr, y_tr = train[features], train[TARGET].values
X_va, y_va = val[features], val[TARGET].values
N, P = len(y_va), int(y_va.sum())
base_rate = P / N
say(f"train {X_tr.shape} | val {X_va.shape} | base rate {base_rate:.4%} (random PR-AUC = {base_rate:.4f})")


# ------------------------------------------------------------------ helpers
def fit(params, cols, seed=SEED):
    p = dict(BASE)
    p.update(params)
    p["random_state"] = seed
    m = lgb.LGBMClassifier(**p)
    m.fit(X_tr[cols], y_tr, eval_set=[(X_va[cols], y_va)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
    pv = m.predict_proba(X_va[cols])[:, 1]
    return m, pv, average_precision_score(y_va, pv)


def paired_boot(a, b, n=N_BOOT, seed=0):
    """mean and 95% CI of AP(a) - AP(b) over bootstrap resamples of the validation set"""
    rs = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        i = rs.randint(0, N, N)
        d.append(average_precision_score(y_va[i], a[i]) - average_precision_score(y_va[i], b[i]))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def fmt_diff(a, b):
    m, lo, hi = paired_boot(a, b)
    verdict = "REAL" if (lo > 0 or hi < 0) else "noise (CI contains 0)"
    return f"{m:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}] -> {verdict}"


def rank01(s):
    return pd.Series(s).rank(pct=True).values


# ------------------------------------------------------------------ A. ablation
say("\n===== A. ABLATION (drop one group of one-hot columns) =====")
m0, p0, ap0 = fit({}, features)
say(f"all {len(features)} features: PR-AUC {ap0:.4f} | ROC-AUC {roc_auc_score(y_va, p0):.4f} | iters {m0.best_iteration_}")
keep = list(features)
if RUN_ABLATION:
    for g, pre in GROUPS.items():
        cols = [c for c in features if not c.startswith(pre)]
        _, pg, apg = fit({}, cols)
        say(f"  without {g:<18} ({len(features) - len(cols):>2} cols): PR-AUC {apg:.4f}  "
            f"delta vs all: {fmt_diff(pg, p0)}")
        if apg >= ap0 - 0.0005 and g != "category":      # category carries signal, never drop
            keep = [c for c in keep if not c.startswith(pre)]
say(f"-> keeping {len(keep)} features (dropped groups whose removal did not lower PR-AUC by > 0.0005)")

# ------------------------------------------------------------------ B. random search
say("\n===== B. RANDOM SEARCH =====")
space = {
    "num_leaves": [7, 15, 31, 63],
    "min_child_samples": [50, 100, 200, 500, 1000],
    "colsample_bytree": [0.4, 0.5, 0.7, 0.9],
    "subsample": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 10.0, 50.0, 100.0],
}
rs = np.random.RandomState(SEED)
best_params, best_ap = {}, fit({}, keep)[2]
say(f"default config on kept features: PR-AUC {best_ap:.4f}")
for t in range(N_SEARCH):
    prm = {k: v[rs.randint(len(v))] for k, v in space.items()}
    _, _, ap = fit(prm, keep)
    say(f"  trial {t + 1}/{N_SEARCH}: PR-AUC {ap:.4f}  {prm}")
    if ap > best_ap:
        best_ap, best_params = ap, prm
say(f"-> best params: {best_params or 'default'} (PR-AUC {best_ap:.4f})")

# ------------------------------------------------------------------ C. seed bagging
say("\n===== C. SEED BAGGING =====")
final_params = dict(best_params, learning_rate=0.03)
bag, bag_models = [], []
for i in range(N_SEEDS):
    m, pv, ap = fit(final_params, keep, seed=SEED + i)
    say(f"  seed {SEED + i}: PR-AUC {ap:.4f} (iters {m.best_iteration_})")
    bag.append(pv)
    bag_models.append(m)
p_bag = np.mean(bag, axis=0)
say(f"bagged LightGBM: PR-AUC {average_precision_score(y_va, p_bag):.4f} | ROC-AUC {roc_auc_score(y_va, p_bag):.4f}")

# ------------------------------------------------------------------ D. blend with logistic regression
say("\n===== D. LOGISTIC REGRESSION BLEND =====")
lr = LogisticRegression(max_iter=500, C=1.0).fit(X_tr[keep], y_tr)
p_lr = lr.predict_proba(X_va[keep])[:, 1]
say(f"logistic regression: PR-AUC {average_precision_score(y_va, p_lr):.4f}")
bw, bap = 0.0, -1
for w in np.round(np.arange(0, 0.51, 0.05), 2):
    ap = average_precision_score(y_va, (1 - w) * rank01(p_bag) + w * rank01(p_lr))
    if ap > bap:
        bw, bap = float(w), ap
p_blend = (1 - bw) * rank01(p_bag) + bw * rank01(p_lr)
say(f"best weight on LR = {bw} -> blend PR-AUC {bap:.4f}")

# ---- save trained models right after C & D (cost thresholds are added to the same bundle after section F)
BUNDLE = {"bag_models": bag_models, "lr": lr, "kept_features": keep, "best_params": best_params, "blend_weight_lr": bw,
          "base_rate": float(base_rate), "cost_table": None}
joblib.dump(BUNDLE, f"{MODELS_DIR}/train_models_bundle.joblib")
for i, m_ in enumerate(bag_models):
    m_.booster_.save_model(f"{MODELS_DIR}/lgbm_seed{SEED + i}.txt")
json.dump({"kept_features": keep, "best_params": {k: float(v) for k, v in best_params.items()}, "blend_weight_lr": bw,
           "n_seeds": N_SEEDS, "train_rows": int(len(y_tr)), "val_rows": int(N), "base_rate": float(base_rate),
           "pr_auc_bagged_val": float(average_precision_score(y_va, p_bag)), "cost_table": None},
          open(f"{MODELS_DIR}/model_metadata.json", "w"), indent=2)
say(f"saved bagged LightGBM + LR -> {MODELS_DIR}/train_models_bundle.joblib, model_metadata.json")

# ------------------------------------------------------------------ comparison
say("\n===== COMPARISON (reference = LightGBM, all features, default params) =====")
models = {"lightgbm_all_features": p0, "lightgbm_tuned_bagged": p_bag, "logistic_regression": p_lr,
          "blend_lgbm_lr": p_blend}
summary = {}
for name, s in models.items():
    ap, auc = average_precision_score(y_va, s), roc_auc_score(y_va, s)
    summary[name] = {"pr_auc": float(ap), "roc_auc": float(auc), "lift": float(ap / base_rate)}
    extra = "" if name == "lightgbm_all_features" else f" | vs reference: {fmt_diff(s, p0)}"
    say(f"  {name:<24} PR-AUC {ap:.4f} | ROC-AUC {auc:.4f} | lift {ap / base_rate:.2f}x{extra}")

# ------------------------------------------------------------------ E. calibration
say("\n===== E. CALIBRATION of bagged LightGBM (rank-bucket table; deciles are useless at a 0.58% base rate) =====")
dec = pd.qcut(pd.Series(p_bag).rank(method="first"), [0, .5, .9, .99, .995, 1.0],
              labels=["0-50%", "50-90%", "90-99%", "99-99.5%", "top0.5%"])
cal = pd.DataFrame({"pred": p_bag, "actual": y_va, "d": dec}).groupby("d", observed=True).agg(
    n=("actual", "size"), mean_pred=("pred", "mean"), actual_rate=("actual", "mean"))
say(cal.round(4).to_string())
say(f"top-0.5% lift = {cal['actual_rate'].iloc[-1] / base_rate:.2f}x | bottom-50% = {cal['actual_rate'].iloc[0] / base_rate:.2f}x")

# ------------------------------------------------------------------ F. cost-based thresholds
say("\n===== F. COST-BASED OPERATING POINT (bagged LightGBM) =====")
say("cost = false_alerts*1 + missed_frauds*R ; R = how many times worse a missed fraud is than a false alert")
say("saving is measured against the best trivial policy (flag nobody, or flag everybody when R > (1-base)/base)")
order = np.argsort(-p_bag)
ys, ps = y_va[order], p_bag[order]
cum_tp = np.cumsum(ys)
k = np.arange(1, N + 1)
cum_fp = k - cum_tp
rows = []
for R in COST_RATIOS:
    cost = cum_fp + R * (P - cum_tp)
    kb = int(np.argmin(cost))
    cost0 = min(R * P, N - P)                        # best trivial policy: flag nobody OR flag everybody
    if cost[kb] >= cost0:
        rows.append((R, 1 / (1 + R), None, 0, 0, 0, 0, 0.0))
        continue
    rows.append((R, 1 / (1 + R), float(ps[kb]), (kb + 1) / N * 1000, cum_tp[kb] / (kb + 1),
                 cum_tp[kb] / P, cum_fp[kb] / (N - P), 1 - cost[kb] / cost0))
tab = pd.DataFrame(rows, columns=["R", "theory_thr(1/(1+R))", "best_thr", "alerts_per_1000",
                                  "precision", "recall", "false_positive_rate", "saving_vs_best_trivial_policy"])
say(tab.round(4).to_string(index=False))
say("(if best_thr is far from theory_thr the probabilities are not well calibrated - see section E)")

# ---- add the cost-based operating points to the saved bundle / metadata
BUNDLE["cost_table"] = tab.to_dict(orient="records")
joblib.dump(BUNDLE, f"{MODELS_DIR}/train_models_bundle.joblib")
_meta = json.load(open(f"{MODELS_DIR}/model_metadata.json"))
_meta["cost_table"] = BUNDLE["cost_table"]
json.dump(_meta, open(f"{MODELS_DIR}/model_metadata.json", "w"), indent=2, default=str)

# ------------------------------------------------------------------ G. final holdout
say("\n===== G. FINAL HOLDOUT (database_ready.csv - never used for tuning / early stopping / thresholds) =====")
db = pd.read_csv(os.path.join(DATA_DIR, "database_ready.csv"))
X_db, y_db = db[keep], db[TARGET].values
p_db = np.mean([m.predict_proba(X_db)[:, 1] for m in bag_models], axis=0)
ap_db, auc_db = average_precision_score(y_db, p_db), roc_auc_score(y_db, p_db)
say(f"holdout: n {len(y_db):,} | base rate {y_db.mean():.4%} | PR-AUC {ap_db:.4f} | ROC-AUC {auc_db:.4f}")
say("thresholds were chosen on VALIDATION and are applied as-is:")
for r in tab.itertuples(index=False):
    if pd.isna(r.best_thr):
        continue
    f = p_db >= r.best_thr
    tp = int((f & (y_db == 1)).sum())
    say(f"  R={r.R:<4} thr {r.best_thr:.4f}: alerts/1000 {f.mean() * 1000:.2f} | "
        f"precision {tp / max(int(f.sum()), 1):.4f} | recall {tp / max(int(y_db.sum()), 1):.4f}")
summary["holdout_bagged_lgbm"] = {"pr_auc": float(ap_db), "roc_auc": float(auc_db),
                                  "lift": float(ap_db / y_db.mean())}

# ------------------------------------------------------------------ artifacts
os.makedirs(OUT, exist_ok=True)   # re-create: OneDrive may remove/lock the folder during the long run
open(f"{OUT}/results.txt", "w", encoding="utf-8").write("\n".join(LOG))   # save the log FIRST
json.dump({"summary": summary, "kept_features": keep, "best_params": {k: float(v) for k, v in best_params.items()},
           "blend_weight_lr": bw, "cost_table": tab.to_dict(orient="records")},
          open(f"{OUT}/results.json", "w"), indent=2, default=str)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for name, s in models.items():
    p, r, _ = precision_recall_curve(y_va, s)
    ax[0].plot(r, p, label=f"{name} (AP={average_precision_score(y_va, s):.3f})")
ax[0].axhline(base_rate, color="gray", ls="--", label="random")
ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision"); ax[0].set_title("PR curves (validation)"); ax[0].legend(fontsize=8)
ax[1].plot(cal["mean_pred"], cal["actual_rate"], "o-", label="bagged LightGBM")
mx = max(cal["mean_pred"].max(), cal["actual_rate"].max())
ax[1].plot([0, mx], [0, mx], "--", color="gray", label="perfect calibration")
ax[1].set_xlabel("mean predicted probability"); ax[1].set_ylabel("actual fraud rate"); ax[1].set_title("Calibration (deciles)"); ax[1].legend()
fig.tight_layout()
fig.savefig(f"{OUT}/pr_and_calibration.png", dpi=200)
plt.close(fig)
print(f"\nDone. Send me {OUT}/results.txt")