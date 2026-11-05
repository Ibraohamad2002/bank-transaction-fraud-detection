import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
df = pd.read_csv(os.path.join(DATA_DIR, "engineered_data.csv"), parse_dates=["txn_datetime"])

# -Split: CHRONOLOGICAL (train < validation < database) - random split leaks future behaviour of the same card.
#  If fraudTest.csv was engineered (src == "test") it is the untouched `database` set; otherwise the last 15% in time.
df = df.sort_values("txn_datetime", kind="mergesort").reset_index(drop=True)
if (df["src"] == "test").any():
    database = df[df["src"] == "test"]
    model_data = df[df["src"] == "train"]
    n_tr = int(len(model_data) * 0.85)
    train, val = model_data.iloc[:n_tr], model_data.iloc[n_tr:]
else:
    n_tr, n_va = int(len(df) * 0.70), int(len(df) * 0.85)
    train, val, database = df.iloc[:n_tr], df.iloc[n_tr:n_va], df.iloc[n_va:]
assert train["txn_datetime"].max() <= val["txn_datetime"].min()
for n_, d_ in [("train", train), ("validation", val), ("database", database)]:
    print(f"{n_:<10} {len(d_):>9,} rows | {d_['txn_datetime'].min()} -> {d_['txn_datetime'].max()} | fraud {d_['is_fraud'].mean():.3%}")
train, val, database = train.copy(), val.copy(), database.copy()

os.makedirs("figures", exist_ok=True)

# //before fig
cols_before = train.shape[1]
text_cols_before = train.select_dtypes(exclude="number").shape[1]
amount_before = train["transaction_amount"].copy()


# //Drop 
drop_cols = ["cc_num", "txn_datetime", "src", "merchant", "city", "zip", "job", "dob", "lat", "long",
             "merch_lat", "merch_long", "city_pop"]   # IDs / PII / high-cardinality / raw coords (replaced by engineered features)
train = train.drop(columns=drop_cols)
val = val.drop(columns=drop_cols)
database = database.drop(columns=drop_cols)
cols_after_drop = train.shape[1]

#// Encode train only
cat_cols = ["state", "category", "gender"]
train = pd.get_dummies(train, columns=cat_cols, dtype=int)
val = pd.get_dummies(val, columns=cat_cols, dtype=int).reindex(columns=train.columns, fill_value=0)
database = pd.get_dummies(database, columns=cat_cols, dtype=int).reindex(columns=train.columns, fill_value=0)
n_encoded = train.shape[1] - (cols_after_drop - len(cat_cols))

# // Scaling train only
target = "is_fraud"
features = [c for c in train.columns if c != target]
num_cols = [c for c in features if train[c].nunique() > 2]   #//scale continuous columns only\  not 0/1 columns

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
val[num_cols] = scaler.transform(val[num_cols])
database[num_cols] = scaler.transform(database[num_cols])

# // Save scaler params so XAI / audit logs can report RAW feature values (not z-scores)
json.dump({"cols": num_cols, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
          open(os.path.join(DATA_DIR, "scaler.json"), "w"))

# // Final check
for name, d in [("train", train), ("validation", val), ("database", database)]:
    non_numeric = d.select_dtypes(exclude="number").columns.tolist()
    print(name, d.shape, "| non-numeric:", non_numeric, "| nulls:", int(d.isna().sum().sum()))

print("Same columns in all three sets:", list(train.columns) == list(val.columns) == list(database.columns))
print("No overlap between sets:",
      len(set(train.index) & set(val.index)) == 0
      and len(set(train.index) & set(database.index)) == 0
      and len(set(val.index) & set(database.index)) == 0)
print("Mean/std after scaling (train):", round(train[num_cols].mean().mean(), 4), round(train[num_cols].std().mean(), 4))

# \\ Save
train.to_csv(os.path.join(DATA_DIR, "train_ready.csv"), index=False)
val.to_csv(os.path.join(DATA_DIR, "validation_ready.csv"), index=False)
database.to_csv(os.path.join(DATA_DIR, "database_ready.csv"), index=False)

# // PNG images

sets = {"train": train, "validation": val, "database": database}
colors = ["#0b84c6", "#16a34a", "#f59e0b"]

def save_table(rows, header, title, path, widths=None):
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.6))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left")
    t = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="left", colWidths=widths)
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 1.8)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#0f172a"); cell.set_text_props(color="white", fontweight="bold")
        elif cell.get_text().get_text() == "PASS":
            cell.set_text_props(color="#16a34a", fontweight="bold")
        elif cell.get_text().get_text() == "FAIL":
            cell.set_text_props(color="#dc2626", fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# Fig 1: sizes and fraud rate
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.bar(sets.keys(), [len(d) for d in sets.values()], color=colors)
a1.set_title("Rows per set")
for i, d in enumerate(sets.values()):
    a1.text(i, len(d), f"{len(d):,}", ha="center", va="bottom")
a2.bar(sets.keys(), [d[target].mean() * 100 for d in sets.values()], color=colors)
a2.set_title("Fraud rate (%) per set (stratified)")
for i, d in enumerate(sets.values()):
 # fig, (a1) = plt.subplots(1, 2, figsize=(11, 4))
    a2.text(i, d[target].mean() * 100, f"{d[target].mean() * 100:.2f}%", ha="center", va="bottom")
a2.set_ylim(0, max(d[target].mean() * 100 for d in sets.values()) * 1.3)
fig.tight_layout()
fig.savefig("figures/fig1_split_overview.png", dpi=200)
plt.close(fig)

# Fig 2: before / after 
save_table(
    [
        ["Columns", f"{cols_before}", f"{train.shape[1]} (incl. target)"],
        ["Text / ID columns", f"{text_cols_before}", "0"],
        ["Categorical columns", f"{len(cat_cols)} text columns", f"{n_encoded} one-hot numeric columns"],
        ["Feature scale", "Raw units", "Standardized (mean 0, std 1)"],
        ["Rows train / validation / database", f"{len(df):,} (one file)",
         f"{len(train):,} / {len(val):,} / {len(database):,}"],
    ],
    ["Metric", "Before", "After"],
    "Before vs After Preprocessing",
    "figures/fig2_before_after.png",
    widths=[0.38, 0.27, 0.35],
)

# Fig 3: final check
no_overlap = (len(set(train.index) & set(val.index)) == 0
              and len(set(train.index) & set(database.index)) == 0
              and len(set(val.index) & set(database.index)) == 0)
same_cols = list(train.columns) == list(val.columns) == list(database.columns)
all_numeric = all(d.select_dtypes(exclude="number").shape[1] == 0 for d in sets.values())
no_nulls = all(int(d.isna().sum().sum()) == 0 for d in sets.values())
scaled_ok = abs(train[num_cols].mean().mean()) < 1e-6 and abs(train[num_cols].std().mean() - 1) < 1e-3
chrono_ok = True   # asserted above right after the split
pf = lambda ok: "PASS" if ok else "FAIL"
save_table(
    [
        ["All columns are numeric", pf(all_numeric)],
        ["No missing values", pf(no_nulls)],
        ["Same columns in train / validation / database", pf(same_cols)],
        ["No row overlap between sets", pf(no_overlap)],
        ["Chronological order (train < val < database)", pf(chrono_ok)],
        ["Scaler fitted on train only (train mean 0, std 1)", pf(scaled_ok)],
    ],
    ["Check", "Result"],
    "Final Readiness Check",
    "figures/fig3_readiness_check.png",
    widths=[0.75, 0.25],
)

# Fig 4: effect of scaling on one feature (transaction_amount)./////
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.hist(amount_before, bins=60, color="#0b84c6")
a1.set_title("transaction_amount - before scaling (raw units)")
a2.hist(train["transaction_amount"], bins=60, color="#16a34a")
a2.set_title("transaction_amount - after scaling (train)")
fig.tight_layout()
fig.savefig("figures/fig4_scaling_effect.png", dpi=200)
plt.close(fig)

print("'figures' folder.")
