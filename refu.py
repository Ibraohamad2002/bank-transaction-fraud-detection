"""
BlackEye - refu.py  (feature engineering, Kaggle "fraud-detection" / Sparkov schema)
Input : data/fraudTrain.csv  (+ data/fraudTest.csv if present -> processed together so per-card history
        continues across the train/test boundary, exactly like production)
Output: data/engineered_data.csv   (column `src` = train/test, `txn_datetime` kept for chronological split)
All per-card statistics use PRIOR transactions only (no self / future leakage).
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
NROWS = None            # e.g. 200_000 for a quick smoke test
TIME_CAP_H = 720.0      # first txn of a card / very long gaps are capped at 30 days

parts = []
for name, src in [("fraudTrain.csv", "train"), ("fraudTest.csv", "test")]:
    p = os.path.join(DATA_DIR, name)
    if os.path.exists(p):
        d = pd.read_csv(p, nrows=NROWS)
        d["src"] = src
        parts.append(d)
        print(f"loaded {name}: {d.shape}")
df = pd.concat(parts, ignore_index=True)
del parts

# --- Drop leakage / ID / PII columns (kept: cc_num + txn_datetime, needed below; dropped in split.py) ---
df = df.drop(columns=["Unnamed: 0", "trans_num", "first", "last", "street", "unix_time"], errors="ignore")

# Preserve original row order (chronological sort below is required for correctness)
df["_row_id"] = np.arange(len(df))
df["txn_datetime"] = pd.to_datetime(df["trans_date_trans_time"])
df = df.drop(columns=["trans_date_trans_time"]).sort_values(["cc_num", "txn_datetime"]).reset_index(drop=True)

cust = df["cc_num"]
amt = df["amt"]
g = df.groupby("cc_num", sort=False)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# === 1. amount_deviation_score - z-score vs card's PRIOR history only (no self-leakage) ===
prior_n = g.cumcount()
prior_sum = amt.groupby(cust).cumsum() - amt
amt_sq = amt ** 2
prior_sum_sq = amt_sq.groupby(cust).cumsum() - amt_sq
n_safe = prior_n.replace(0, np.nan)
prior_mean = prior_sum / n_safe
prior_std = np.sqrt(((prior_sum_sq / n_safe) - prior_mean ** 2).clip(lower=0))
df["amount_deviation_score"] = ((amt - prior_mean) / (prior_std + 1e-3)).fillna(0).clip(-50, 50)
df["amt_to_prior_mean"] = (amt / (prior_mean + 0.01)).fillna(1).clip(0, 500)
df["prior_txn_count"] = prior_n

# === 2. time / geo behaviour ===
df["distance_from_home_km"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
dt_h = g["txn_datetime"].diff().dt.total_seconds() / 3600.0
df["is_first_txn"] = dt_h.isna().astype(int)
df["time_since_last_txn_hrs"] = dt_h.fillna(TIME_CAP_H).clip(upper=TIME_CAP_H)
# keeps the original BlackEye feature name/definition
df["geo_velocity_proxy"] = df["distance_from_home_km"] / (df["time_since_last_txn_hrs"] + 0.001)
# true velocity: previous merchant location -> current merchant location (km/h)
pl, po = g["merch_lat"].shift(), g["merch_long"].shift()
dist_prev = haversine(pl, po, df["merch_lat"], df["merch_long"])
df["dist_from_prev_txn_km"] = dist_prev.fillna(0)
df["merchant_velocity_kmh"] = (dist_prev / (df["time_since_last_txn_hrs"] + 0.01)).fillna(0).clip(upper=5000)

# === 3. velocity windows (count / spend in last 1h & 24h, EXCLUDING the current txn) ===
for w in ("1h", "24h"):
    r = df.groupby("cc_num", sort=False).rolling(w, on="txn_datetime")["amt"]
    df[f"txn_count_{w}"] = r.count().values - 1
    df[f"amt_sum_{w}"] = r.sum().values - amt.values

# === 4. calendar / demographic ===
h = df["txn_datetime"].dt.hour
df["hour"] = h
df["day_of_week"] = df["txn_datetime"].dt.dayofweek
df["is_night"] = ((h >= 22) | (h <= 3)).astype(int)
df["age"] = ((df["txn_datetime"] - pd.to_datetime(df["dob"])).dt.days / 365.25).round(1)
df["log_city_pop"] = np.log1p(df["city_pop"])

# --- Restore original row order; drop helper columns ---
df = df.sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)
df = df.rename(columns={"amt": "transaction_amount"})   # keep the BlackEye feature name

print(f"Post-engineering shape: {df.shape}")
print(df.groupby("src")["is_fraud"].agg(["size", "mean"]))
df.to_csv(os.path.join(DATA_DIR, "engineered_data.csv"), index=False)
