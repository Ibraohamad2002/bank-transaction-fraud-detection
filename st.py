import pandas as pd
import numpy as np
from scipy.stats import zscore

df = pd.read_csv('data/fraudTrain.csv', index_col=0)   # col 0 is a pandas row index
N = len(df)

# --- Auto-detect target column ---
TARGET_CANDIDATES = ['isFraud', 'is_fraud', 'Is_Fraud', 'Fraud', 'fraud_label', 'Class']
target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)

# --- 1. Missing Values ---
missing = pd.DataFrame({
    'missing_count': df.isnull().sum(),
    'missing_pct': (df.isnull().sum() / N * 100).round(2)
}).query('missing_count > 0').sort_values('missing_pct', ascending=False)
print("=== MISSING VALUES ===")
print(missing if not missing.empty else "None", "\n")

# --- 2. Dtypes & Cardinality ---
dtype_summary = pd.DataFrame({
    'dtype': df.dtypes.astype(str),
    'unique_count': df.nunique(),
    'unique_pct': (df.nunique() / N * 100).round(3)
}).sort_values('unique_count', ascending=False)
print("=== DTYPES & CARDINALITY ===")
print(dtype_summary, "\n")

# --- 3. Class Imbalance ---
if target_col:
    counts = df[target_col].value_counts()
    pct = (counts / N * 100).round(4)
    print("=== CLASS IMBALANCE ===")
    print(pd.DataFrame({'count': counts, 'pct': pct}))
    if len(counts) == 2:
        maj, minr = counts.max(), counts.min()
        print(f"Imbalance ratio (Majority:Minority) = {maj/minr:.1f} : 1\n")
else:
    print("=== CLASS IMBALANCE === Target column not found — set target_col manually.\n")

# --- 4. Outlier Detection (IQR) on numeric features ---
ID_LIKE = ['cc_num', 'zip', 'unix_time']                                   # numeric but not real measurements
num_cols = df.select_dtypes(include=[np.number]).columns.difference(([target_col] if target_col else []) + ID_LIKE)
outlier_rows = []
for col in num_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
    outlier_rows.append({'feature': col, 'iqr': round(iqr, 2), 'lower': round(lo, 2),
                          'upper': round(hi, 2), 'outlier_count': n_out,
                          'outlier_pct': round(n_out / N * 100, 2)})
outlier_df = pd.DataFrame(outlier_rows).sort_values('outlier_pct', ascending=False)
print("=== OUTLIER SUMMARY (IQR, k=1.5) ===")
print(outlier_df, "\n")

# --- 5. Leakage / time-coverage checks ---
ts = pd.to_datetime(df['trans_date_trans_time'])
print("=== TIME COVERAGE ===")
print(ts.min(), "->", ts.max(), "| cards:", df['cc_num'].nunique(), "| txns/card median:", int(df.groupby('cc_num').size().median()), "\n")
if target_col:
    print("=== FRAUD RATE BY CATEGORY / HOUR (signal check) ===")
    print((df.groupby('category')[target_col].mean() * 100).round(2).sort_values(ascending=False).head(6))
    print((df.groupby(ts.dt.hour)[target_col].mean() * 100).round(2).sort_values(ascending=False).head(6))
