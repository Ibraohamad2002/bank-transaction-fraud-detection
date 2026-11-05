"""
BlackEye - leakage_audit.py   (is the model too good because of leakage / overfitting / card memorization?)
Run AFTER split.py and train_models.py. Needs data/engineered_data.csv (to recover cc_num, which the ready files dropped).

 T1 shuffled labels : train on permuted labels -> holdout PR-AUC must collapse to the base rate (else the pipeline leaks)
 T2 single-feature  : ROC-AUC of every raw feature alone on holdout (a lone feature > 0.95 smells like a leak)
 T3 seen vs unseen cards : same model, same (future) holdout period, cards the model trained on vs cards it never saw
                           -> the gap is entity memorization (an identity proxy such as city_pop / age / gender)
 T4 identity-proxy ablation : drop log_city_pop (a near-unique card fingerprint) and compare on UNSEEN cards
Install: pip install lightgbm scikit-learn pandas numpy
"""
import json
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore", message=".*eval_set.*")

DATA_DIR, RESULTS, OUT, TARGET, SEED = "data", "results_v3/results.json", "blackeye_v1", "is_fraud", 42
EXCLUDED = ("age", "gender_", "prior_txn_count")        # same as blackeye_engine.py
IDENTITY_PROXIES = ["log_city_pop"]                      # near-unique per card (879 values for 983 cards)
CARD_FRAC_SEEN = 0.70
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def flush():
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/leakage_report.txt", "w", encoding="utf-8").write("\n".join(LOG))


res = json.load(open(RESULTS))
F_ALL = res["kept_features"]
F_LIVE = [f for f in F_ALL if not f.startswith(EXCLUDED)]
bp = {k: (int(v) if k in ("num_leaves", "min_child_samples") else v) for k, v in res["best_params"].items()}
rd = lambda n: pd.read_csv(os.path.join(DATA_DIR, n), usecols=F_ALL + [TARGET])
tr, va, db = rd("train_ready.csv"), rd("validation_ready.csv"), rd("database_ready.csv")
y_tr, y_va, y_db = tr[TARGET].values, va[TARGET].values, db[TARGET].values

# --- recover card ids with the SAME ordering split.py used (stable sort by time) and prove alignment via labels
eng = pd.read_csv(os.path.join(DATA_DIR, "engineered_data.csv"), usecols=["cc_num", "txn_datetime", TARGET, "src"],
                  parse_dates=["txn_datetime"])
eng = eng.sort_values("txn_datetime", kind="mergesort").reset_index(drop=True)
if (eng["src"] == "test").any():
    mm, dbc = eng[eng["src"] == "train"], eng[eng["src"] == "test"]
    n_tr = int(len(mm) * 0.85)
    trc, vac = mm.iloc[:n_tr], mm.iloc[n_tr:]
else:
    a, b = int(len(eng) * 0.70), int(len(eng) * 0.85)
    trc, vac, dbc = eng.iloc[:a], eng.iloc[a:b], eng.iloc[b:]
assert len(trc) == len(tr) and len(vac) == len(va) and len(dbc) == len(db), "row counts differ from ready files"
assert (trc[TARGET].values == y_tr).all() and (vac[TARGET].values == y_va).all() and (dbc[TARGET].values == y_db).all(), \
    "row order differs from ready files - re-run split.py"
c_tr, c_va, c_db = trc["cc_num"].values, vac["cc_num"].values, dbc["cc_num"].values
cards = np.unique(c_tr)
rs = np.random.RandomState(SEED)
seen = set(rs.choice(cards, int(len(cards) * CARD_FRAC_SEEN), replace=False))
say(f"train {tr.shape} | val {va.shape} | holdout {db.shape} | cards: {len(cards)} ({len(seen)} 'seen', {len(cards) - len(seen)} 'unseen')")
say(f"holdout base rate {y_db.mean():.4%} | cards in holdout that never appear in train: {len(set(c_db) - set(cards))}")

BASE = dict(n_estimators=4000, learning_rate=0.05, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=10.0, n_jobs=-1, verbosity=-1, metric="average_precision", random_state=SEED)
BASE.update(bp)


def fit(cols, Xa, ya, Xv=None, yv=None, **over):
    m = lgb.LGBMClassifier(**{**BASE, **over})
    if Xv is None:
        m.fit(Xa[cols], ya)
    else:
        m.fit(Xa[cols], ya, eval_set=[(Xv[cols], yv)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    return m


ap = average_precision_score

# ------------------------------------------------------------------ T1 shuffled labels
say("\n===== T1. SHUFFLED-LABEL SANITY (labels permuted in train; PR-AUC must be ~ base rate) =====")
m = fit(F_LIVE, tr, rs.permutation(y_tr), n_estimators=200)
a1 = ap(y_db, m.predict_proba(db[F_LIVE])[:, 1])
say(f"holdout PR-AUC with shuffled labels: {a1:.4f} (base rate {y_db.mean():.4f}) -> "
    f"{'PASS (no pipeline leak)' if a1 < 3 * y_db.mean() else 'FAIL - features carry label information'}")
flush()

# ------------------------------------------------------------------ T2 single-feature AUC
say("\n===== T2. SINGLE-FEATURE ROC-AUC on holdout (top 6; > 0.95 alone = suspicious) =====")
sa = {}
for f in F_LIVE:
    if db[f].nunique() > 1:
        a = roc_auc_score(y_db, db[f])
        sa[f] = max(a, 1 - a)
say(pd.Series(sa).sort_values(ascending=False).head(6).round(4).to_string())

# ------------------------------------------------------------------ T3 seen vs unseen cards
say(f"\n===== T3. SEEN vs UNSEEN CARDS (train only on {int(CARD_FRAC_SEEN * 100)}% of cards; evaluate on the FUTURE holdout) =====")
m_tr, m_va = np.isin(c_tr, list(seen)), np.isin(c_va, list(seen))
m_dbs = np.isin(c_db, list(seen))
Xa, Xv = tr[m_tr], va[m_va]
say(f"holdout rows: seen cards {int(m_dbs.sum()):,} ({int(y_db[m_dbs].sum())} frauds) | unseen cards {int((~m_dbs).sum()):,} ({int(y_db[~m_dbs].sum())} frauds)")


def boot_gap(p, n=30):
    r, d = np.random.RandomState(0), []
    ys, ps, yu, pu = y_db[m_dbs], p[m_dbs], y_db[~m_dbs], p[~m_dbs]
    for _ in range(n):
        i, j = r.randint(0, len(ys), len(ys)), r.randint(0, len(yu), len(yu))
        d.append(ap(ys[i], ps[i]) - ap(yu[j], pu[j]))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def seen_unseen(name, cols):
    m = fit(cols, Xa, y_tr[m_tr], Xv, y_va[m_va])
    p = m.predict_proba(db[cols])[:, 1]
    s, u = ap(y_db[m_dbs], p[m_dbs]), ap(y_db[~m_dbs], p[~m_dbs])
    lo, hi = boot_gap(p)
    say(f"  {name:<34} seen-cards PR-AUC {s:.4f} | unseen-cards PR-AUC {u:.4f} | gap {s - u:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]")
    return u


u_all = seen_unseen(f"all {len(F_ALL)} features (incl. age/gender)", F_ALL)
flush()
u_live = seen_unseen(f"live {len(F_LIVE)} features", F_LIVE)
flush()

# ------------------------------------------------------------------ T4 identity-proxy ablation
say("\n===== T4. DROP IDENTITY PROXIES (evaluated on UNSEEN cards only = honest 'new customer' performance) =====")
F_NOID = [f for f in F_LIVE if f not in IDENTITY_PROXIES]
m = fit(F_NOID, Xa, y_tr[m_tr], Xv, y_va[m_va])
p = m.predict_proba(db[F_NOID])[:, 1]
u_noid = ap(y_db[~m_dbs], p[~m_dbs])
say(f"  without {IDENTITY_PROXIES}: unseen-cards PR-AUC {u_noid:.4f} (live set {u_live:.4f}, all features {u_all:.4f})")

# ------------------------------------------------------------------ verdict
say("\n===== VERDICT RULES =====")
say("- T1 PASS + T2 no feature > 0.95  -> no label leakage in the pipeline.")
say("- T3 gap ~ 0 (CI contains 0)       -> no card memorization: performance transfers to NEW customers.")
say("- T3 gap > ~0.02 and larger with age/gender than without -> those features act as card fingerprints: keep them excluded.")
say("- T4 unseen-card PR-AUC unchanged after dropping log_city_pop -> it is not a fingerprint; keep. If it drops, treat as identity proxy.")
say("- If unseen-card PR-AUC is still ~0.95+: remaining strength comes from the synthetic generator (rule-based fraud), not leakage.")
flush()
print(f"\nDone. Send me {OUT}/leakage_report.txt")
