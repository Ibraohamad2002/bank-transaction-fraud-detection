import pandas as pd
import numpy as np
from scipy.stats import zscore

df = pd.read_csv('bank_fraud.csv')
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

# High-cardinality flag (potential ID leakage / needs hashing, not one-hot)
high_card = dtype_summary[(dtype_summary['dtype'] == 'object') & (dtype_summary['unique_pct'] > 50)]
if not high_card.empty:
    print("⚠ HIGH-CARDINALITY CATEGORICALS (avoid naive one-hot):\n", high_card.index.tolist(), "\n")

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
num_cols = df.select_dtypes(include=[np.number]).columns.difference([target_col] if target_col else [])
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

# --- 5. Z-score cross-check (top 5 IQR-flagged features) ---
print("=== Z-SCORE CROSS-CHECK (|z|>3) ===")
for col in outlier_df.head(5)['feature']:
    z_out = int((np.abs(zscore(df[col].dropna())) > 3).sum())
    print(f"{col}: {z_out} rows ({z_out/N*100:.2f}%)")

# --- 6. Fraud-conditional outlier skew (if target exists) — tells you if outliers ARE the signal ---
if target_col:
    print("\n=== OUTLIER RATE BY CLASS (top numeric features) ===")
    for col in outlier_df.head(5)['feature']:
        q1, q3 = df[col].quantile([0.25, 0.75]); iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        is_out = (df[col] < lo) | (df[col] > hi)
        rate = df.groupby(target_col)[col].apply(lambda s: ((s < lo) | (s > hi)).mean() * 100)
        print(f"{col}: {dict(rate.round(2))}")