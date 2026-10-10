import pandas as pd
import numpy as np

df = pd.read_csv('bank_fraud.csv')

# --- Drop leakage / ID columns ---
df = df.drop(columns=['transaction_id', 'fraud_type'], errors='ignore')

# Preserve original row order (chronological sort below is required for correctness)
df['_row_id'] = np.arange(len(df))

# --- Chronological order per customer_id (expanding stats must respect time to avoid leakage) ---
df['txn_datetime'] = pd.to_datetime(df['transaction_date'] + ' ' + df['transaction_time'])
df = df.sort_values(['customer_id', 'txn_datetime'])

# === 1. amount_deviation_score — z-score vs customer's PRIOR history only (no self-leakage) ===
cust = df['customer_id']
amt = df['transaction_amount']

prior_n = df.groupby(cust).cumcount()                                   # count of prior txns (0-indexed)
prior_sum = amt.groupby(cust).cumsum() - amt                             # sum excl. current row
amt_sq = amt ** 2
prior_sum_sq = amt_sq.groupby(cust).cumsum() - amt_sq                    # sumsq excl. current row

n_safe = prior_n.replace(0, np.nan)
prior_mean = prior_sum / n_safe
prior_var = (prior_sum_sq / n_safe) - prior_mean ** 2
prior_std = np.sqrt(prior_var.clip(lower=0))

df['amount_deviation_score'] = ((amt - prior_mean) / (prior_std + 1e-3)).fillna(0)

# === 2. geo_velocity_proxy ===
df['geo_velocity_proxy'] = df['distance_from_home_km'] / (df['time_since_last_txn_hrs'] + 0.001)

# === 3. failed_intl_interaction ===
df['failed_intl_interaction'] = df['failed_attempts'] * df['is_international']

# === 4. balance_drain_ratio ===
df['balance_drain_ratio'] = df['transaction_amount'] / (df['account_balance'] + 0.001)

# --- Restore original row order; drop helper columns ---
df = df.sort_values('_row_id').drop(columns=['_row_id', 'txn_datetime']).reset_index(drop=True)

print(f"Post-engineering shape: {df.shape}")
# حفظ البيانات في ملف جديد
df.to_csv('engineered_data.csv', index=False)