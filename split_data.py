"""File 1 — splits the data 80/20 into files and runs the leak-free preprocessing.
Run:  python split_data.py [path/to/engineered_data.csv]
Needs fraud_preprocessing.py in the same folder.
Outputs (artifacts/): train_raw.parquet, test_unseen_raw.csv, X_train/X_test/y_train/y_test.parquet,
                      preprocessor.joblib, meta.json
"""
import sys, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from fraud_preprocessing import build_preprocessor, TARGET, DROP_COLS, NUM_COLS

IN_CSV = sys.argv[1] if len(sys.argv) > 1 else "engineered_data.csv"
OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)
SEED, TEST_SIZE, MIN_HISTORY = 42, 0.20, 3

# 1. Load
df = pd.read_csv(IN_CSV)
df["row_id"] = np.arange(len(df))          # permanent key: proves zero overlap + used by the future database

# 2. Repair amount_deviation_score (original produced |z| up to 3e7 because prior std ~ 0 with ~5 txns/customer)
d = df.assign(_dt=pd.to_datetime(df["transaction_date"] + " " + df["transaction_time"])) \
      .sort_values(["customer_id", "_dt"])
amt, cust = d["transaction_amount"], d["customer_id"]
n = d.groupby("customer_id").cumcount()
s1 = amt.groupby(cust).cumsum() - amt
s2 = (amt ** 2).groupby(cust).cumsum() - amt ** 2
mean = s1 / n.replace(0, np.nan)
std = np.sqrt((s2 / n.replace(0, np.nan) - mean ** 2).clip(lower=0))
denom = np.maximum(np.maximum(std, 0.1 * mean.abs()), 1.0)          # floor: max(std, 10% of mean, 1.0)
df.loc[d.index, "amount_deviation_score"] = ((amt - mean) / denom).where(n >= MIN_HISTORY, 0.0).fillna(0.0)

# 3. SPLIT FIRST — 80% train / 20% test, stratified so the fraud rate is identical in both
train_raw, test_raw = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET],
                                       random_state=SEED, shuffle=True)
train_raw, test_raw = train_raw.sort_values("row_id"), test_raw.sort_values("row_id")

# 4. Fit preprocessing on TRAIN ONLY, transform both
prep = build_preprocessor()
X_train = prep.fit_transform(train_raw.drop(columns=DROP_COLS)).astype("float32").reset_index(drop=True)
X_test = prep.transform(test_raw.drop(columns=DROP_COLS)).astype("float32").reset_index(drop=True)
y_train = train_raw[TARGET].reset_index(drop=True)
y_test = test_raw[TARGET].reset_index(drop=True)

# 5. Readiness checks
assert len(train_raw) + len(test_raw) == len(df)
assert set(train_raw.row_id).isdisjoint(test_raw.row_id), "row overlap!"
assert list(X_train.columns) == list(X_test.columns)
assert all(np.issubdtype(t, np.number) for t in X_train.dtypes), "non-numeric column"
assert not X_train.isna().any().any() and not X_test.isna().any().any()
assert np.isfinite(X_train.values).all() and np.isfinite(X_test.values).all()
assert not ({TARGET, "customer_id", "row_id"} & set(X_train.columns)), "ID/target leaked into X"
assert abs(y_train.mean() - y_test.mean()) < 5e-4, "class ratio drifted"
assert np.allclose(X_train[NUM_COLS].mean(), 0, atol=1e-4)
assert np.allclose(X_train[NUM_COLS].std(ddof=0), 1, atol=1e-3)

# 6. Save to files
train_raw.to_parquet(OUT / "train_raw.parquet")             # EDA uses TRAIN only (never peek at test)
test_raw.to_csv(OUT / "test_unseen_raw.csv", index=False)   # load this into the database later
X_train.to_parquet(OUT / "X_train.parquet"); X_test.to_parquet(OUT / "X_test.parquet")
y_train.to_frame().to_parquet(OUT / "y_train.parquet"); y_test.to_frame().to_parquet(OUT / "y_test.parquet")
joblib.dump(prep, OUT / "preprocessor.joblib")
json.dump({"features": list(X_train.columns), "seed": SEED, "test_size": TEST_SIZE,
           "rows_total": len(df), "columns_before": df.shape[1] - 1,   # minus helper row_id
           "train_rows": len(X_train), "test_rows": len(X_test)},
          open(OUT / "meta.json", "w"), indent=2)
print(f"train {X_train.shape} | test {X_test.shape} | fraud rate train {y_train.mean():.4%} test {y_test.mean():.4%}")
print("All readiness assertions passed. Files saved to", OUT.resolve())
