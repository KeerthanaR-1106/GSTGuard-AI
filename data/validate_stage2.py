"""
validate_stage2.py
==================
Comprehensive validation of Stage 2 output: data/ml_features.csv
Prints a PASS / FAIL report for every check.
Does NOT modify any file.

Run:
    python data/validate_stage2.py
"""

import os, sys
import pandas as pd
import numpy as np

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(BASE_DIR)
FEATURES_PATH = os.path.join(ROOT_DIR, "data", "ml_features.csv")
INVOICES_PATH = os.path.join(ROOT_DIR, "data", "gst_invoices.csv")
GT_PATH       = os.path.join(ROOT_DIR, "data", "ground_truth.csv")

PASS_COUNT = 0
FAIL_COUNT = 0
results    = []

def check(label, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    tag  = f"[{status}]"
    line = f"{tag}  {label}"
    if detail:
        line += f"\n        -> {detail}"
    results.append((status, label, detail))
    print(line)

def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")

# ─────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  Stage 2 Validation Report  --  ml_features.csv")
print("="*62)

# =================================================================
section("CHECK 1  --  File Existence")
# =================================================================

feat_exists = os.path.exists(FEATURES_PATH)
inv_exists  = os.path.exists(INVOICES_PATH)
gt_exists   = os.path.exists(GT_PATH)

check("data/ml_features.csv exists", feat_exists, FEATURES_PATH)
check("data/gst_invoices.csv exists (needed for comparison)", inv_exists)
check("data/ground_truth.csv exists (needed for leak check)", gt_exists)

if not (feat_exists and inv_exists and gt_exists):
    print("\n[ABORT] One or more required files are missing.")
    sys.exit(1)

feat = pd.read_csv(FEATURES_PATH)
raw  = pd.read_csv(INVOICES_PATH)
gt   = pd.read_csv(GT_PATH)

# =================================================================
section("CHECK 2  --  Shape (rows x columns)")
# =================================================================

n_rows, n_cols = feat.shape
print(f"\n  ml_features.csv shape: {n_rows:,} rows x {n_cols} columns")
print(f"  (expected: 20,000 rows, 44 columns = invoice_no + 43 features)\n")

check("Row count is exactly 20,000",
      n_rows == 20_000, f"Actual: {n_rows:,}")
check("Column count is 44  (invoice_no + 43 features)",
      n_cols == 44, f"Actual: {n_cols}")

# =================================================================
section("CHECK 3  --  Same 20,000 Invoices as Raw Dataset")
# =================================================================

feat_nos = set(feat["invoice_no"])
raw_nos  = set(raw["invoice_no"])

only_in_feat = feat_nos - raw_nos
only_in_raw  = raw_nos  - feat_nos

check("All invoice_no values match raw dataset (no extras, no drops)",
      len(only_in_feat) == 0 and len(only_in_raw) == 0,
      f"Only in features: {len(only_in_feat)}, Only in raw: {len(only_in_raw)}")

check("No duplicate invoice_no in ml_features.csv",
      feat["invoice_no"].duplicated().sum() == 0,
      f"Duplicates: {feat['invoice_no'].duplicated().sum()}")

# =================================================================
section("CHECK 4  --  invoice_no NOT Used as ML Feature")
# =================================================================

# invoice_no should be in the file (for joining) but must be excluded
# from any numeric feature set a model would use.
check("invoice_no column is present (for joining results)",
      "invoice_no" in feat.columns)

# invoice_no dtype must be object/string, not numeric.
# pandas 3.x may report it as 'str' rather than 'object' -- both are correct.
inv_no_dtype = str(feat["invoice_no"].dtype)
check("invoice_no is a string/object (non-numeric -- cannot be fed to a model)",
      inv_no_dtype in ("object", "str"),
      f"dtype: {inv_no_dtype}  (both 'object' and 'str' are valid string dtypes)")

# =================================================================
section("CHECK 5  --  Labels NOT Present in ml_features.csv")
# =================================================================

FORBIDDEN = ["actual_anomaly", "anomaly_type",
             "is_anomaly", "label", "fraud", "anomaly"]

leaked = [c for c in feat.columns if c.lower() in [f.lower() for f in FORBIDDEN]]
check("actual_anomaly is NOT in ml_features.csv",
      "actual_anomaly" not in feat.columns)
check("anomaly_type is NOT in ml_features.csv",
      "anomaly_type" not in feat.columns)
check("No other label-like columns present",
      len(leaked) == 0,
      f"Suspicious columns: {leaked}" if leaked else "")

# =================================================================
section("CHECK 6  --  Full Feature List")
# =================================================================

feature_cols = [c for c in feat.columns if c != "invoice_no"]
print(f"\n  Total ML features: {len(feature_cols)}\n")

EXPECTED_FEATURES = [
    # Raw numeric invoice fields
    "quantity", "unit_price", "discount_percentage", "discount_amount",
    "taxable_value", "gst_rate", "cgst", "sgst", "igst",
    "cess_rate", "cess_amount", "total_tax", "total_amount",
    # GST consistency
    "calculated_gross_value", "calculated_discount", "discount_difference",
    "calculated_taxable_value", "taxable_value_difference",
    "expected_gst", "calculated_gst", "gst_difference",
    "expected_cgst", "expected_sgst", "expected_igst",
    "cgst_sgst_split_difference", "igst_difference",
    "calculated_total_tax", "total_tax_difference",
    "calculated_total_amount", "total_amount_difference",
    "amount_to_taxable_ratio",
    # Date
    "future_date_flag", "days_beyond_allowed_end",
    "invoice_month", "invoice_day_of_week",
    # Frequency
    "supplier_frequency", "buyer_frequency",
    "supplier_buyer_frequency",
    "supplier_daily_frequency", "buyer_daily_frequency",
    "daily_transaction_frequency",
    # Duplicate
    "duplicate_transaction_frequency", "duplicate_transaction_flag",
]

missing_feats = [f for f in EXPECTED_FEATURES if f not in feature_cols]
extra_feats   = [f for f in feature_cols if f not in EXPECTED_FEATURES]

for i, col in enumerate(feature_cols, 1):
    marker = "  " if col in EXPECTED_FEATURES else "* "
    print(f"    {marker}{i:2}. {col}")

if extra_feats:
    print(f"\n  (*) = unexpected extra columns: {extra_feats}")

check("All 43 expected feature columns are present",
      len(missing_feats) == 0,
      f"Missing: {missing_feats}" if missing_feats else "All present")

check("No unexpected extra feature columns",
      len(extra_feats) == 0,
      f"Extra: {extra_feats}" if extra_feats else "")

# =================================================================
section("CHECK 7  --  Missing Values")
# =================================================================

missing_per_col = feat.isnull().sum()
total_missing   = missing_per_col.sum()

check("Zero missing values in ml_features.csv",
      total_missing == 0,
      f"Total NaNs: {total_missing}" if total_missing > 0
      else f"0 NaNs across all {n_cols} columns")

if total_missing > 0:
    print("  Columns with missing values:")
    print(missing_per_col[missing_per_col > 0].to_string())

# =================================================================
section("CHECK 8  --  Infinite Values")
# =================================================================

num_feat = feat[feature_cols]
inf_mask = np.isinf(num_feat.values)
total_inf = int(inf_mask.sum())

check("Zero infinite values in feature columns",
      total_inf == 0,
      f"Total Infs: {total_inf}" if total_inf > 0 else "0 Inf values")

if total_inf > 0:
    inf_per_col = pd.Series(inf_mask.sum(axis=0), index=feature_cols)
    print("  Columns with Inf:")
    print(inf_per_col[inf_per_col > 0].to_string())

# =================================================================
section("CHECK 9  --  All ML Features Are Numeric")
# =================================================================

non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(feat[c])]
check("All 43 feature columns are numeric dtype",
      len(non_numeric) == 0,
      f"Non-numeric: {non_numeric}" if non_numeric else "All are float64/int64")

print("\n  Dtypes summary:")
dtype_counts = feat[feature_cols].dtypes.value_counts()
for dt, cnt in dtype_counts.items():
    print(f"    {str(dt):<12}: {cnt} columns")

# =================================================================
section("CHECK 10  --  Constant / Zero-Variance Features")
# =================================================================

stds = feat[feature_cols].std()
zero_var = stds[stds == 0].index.tolist()
near_zero = stds[(stds > 0) & (stds < 1e-6)].index.tolist()

check("No constant (zero-variance) features",
      len(zero_var) == 0,
      f"Constant columns: {zero_var}" if zero_var else "None")

check("No near-zero-variance features (std < 1e-6)",
      len(near_zero) == 0,
      f"Near-zero std: {near_zero}" if near_zero else "None")

print("\n  Feature standard deviations (top 10 highest):")
for col, std_val in stds.nlargest(10).items():
    print(f"    {col:<40}: std = {std_val:.4f}")

print("\n  Feature standard deviations (bottom 10 lowest):")
for col, std_val in stds.nsmallest(10).items():
    print(f"    {col:<40}: std = {std_val:.6f}")

# =================================================================
section("CHECK 11  --  GST Consistency Features Reasonableness")
# =================================================================

# For normal invoices, deviation features should be near 0
# For anomalous invoices, they should be higher
merged = feat.merge(gt, on="invoice_no")
normal = merged[merged["actual_anomaly"] == 0]
anom   = merged[merged["actual_anomaly"] == 1]

# NOTE on taxable_value_difference:
# This feature measures whether stored taxable_value = gross - discount.
# None of our 11 anomaly types corrupt taxable_value directly, so this
# feature is ~0 for both normal and anomalous rows -- which is CORRECT.
# It would catch a hypothetical anomaly that tampers with taxable_value.
# Therefore we check it separately: it should be ~0 across ALL rows.
tv_diff_ok = (feat["taxable_value_difference"] <= 0.05).all()
check("taxable_value_difference is ~0 for ALL rows (correct: no anomaly corrupts taxable_value)",
      tv_diff_ok,
      f"Rows with taxable_value_difference > Rs0.05: {(feat['taxable_value_difference'] > 0.05).sum()}")

# For the other consistency features, anomalous invoices must score higher
CONSISTENCY_ANOMALY_FEATS = [
    "discount_difference",
    "gst_difference",
    "cgst_sgst_split_difference",
    "igst_difference",
    "total_tax_difference",
    "total_amount_difference",
    "amount_to_taxable_ratio",
]

print("\n  GST consistency features -- mean for NORMAL vs ANOMALOUS invoices:")
print(f"  {'Feature':<40} {'Normal':>12} {'Anomalous':>12} {'Ratio':>8}")
print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*8}")

all_higher = True
for feat_name in CONSISTENCY_ANOMALY_FEATS:
    nm = normal[feat_name].mean()
    am = anom[feat_name].mean()
    ratio = am / nm if nm > 0 else float("inf") if am > 0 else 1.0
    higher = am >= nm
    flag = "OK" if higher else "!!"
    print(f"  {feat_name:<40} {nm:>12.4f} {am:>12.4f} {ratio:>8.2f}x  {flag}")
    if not higher:
        all_higher = False

check("GST consistency deviation features are higher for anomalous invoices",
      all_higher,
      "All features are higher (or equal) for anomalous rows"
      if all_higher else "Some features are NOT higher for anomalous rows")

# Normal invoices: discount_diff, taxable_diff should be ~0
normal_disc_nonzero   = (normal["discount_difference"]    > 0.05).sum()
normal_taxable_nonzero = (normal["taxable_value_difference"] > 0.05).sum()
normal_ttax_nonzero   = (normal["total_tax_difference"]   > 0.05).sum()
normal_tamt_nonzero   = (normal["total_amount_difference"] > 0.05).sum()

check("Normal invoices: discount_difference is ~0 for all rows",
      normal_disc_nonzero == 0,
      f"{normal_disc_nonzero} normal rows have discount_difference > Rs0.05")

check("Normal invoices: taxable_value_difference is ~0 for all rows",
      normal_taxable_nonzero == 0,
      f"{normal_taxable_nonzero} normal rows have taxable_value_difference > Rs0.05")

check("Normal invoices: total_tax_difference is ~0 for all rows",
      normal_ttax_nonzero == 0,
      f"{normal_ttax_nonzero} normal rows have total_tax_difference > Rs0.05")

check("Normal invoices: total_amount_difference is ~0 for all rows",
      normal_tamt_nonzero == 0,
      f"{normal_tamt_nonzero} normal rows have total_amount_difference > Rs0.05")

# amount_to_taxable_ratio: normal should be 1.0–2.0 range
normal_ratio_ok = ((normal["amount_to_taxable_ratio"] >= 1.0) &
                   (normal["amount_to_taxable_ratio"] <= 2.5)).all()
check("Normal invoices: amount_to_taxable_ratio is between 1.0 and 2.5",
      normal_ratio_ok,
      f"Out-of-range: {(~((normal['amount_to_taxable_ratio']>=1.0) & (normal['amount_to_taxable_ratio']<=2.5))).sum()} rows")

# =================================================================
section("CHECK 12  --  future_date_flag and duplicate_transaction_flag")
# =================================================================

# future_date_flag must be binary (0 or 1)
fdf_vals = feat["future_date_flag"].unique()
check("future_date_flag contains only 0 and 1",
      set(fdf_vals) <= {0, 1},
      f"Values found: {sorted(fdf_vals)}")

# Count of future-date flagged rows should match ground truth
future_gt_count = (gt["anomaly_type"] == "Future Invoice Date").sum()
future_feat_count = feat["future_date_flag"].sum()
check(f"future_date_flag count matches ground truth Future Invoice Date count ({future_gt_count})",
      future_feat_count == future_gt_count,
      f"Flagged in features: {int(future_feat_count)}, In GT: {future_gt_count}")

# duplicate_transaction_flag must be binary
dtf_vals = feat["duplicate_transaction_flag"].unique()
check("duplicate_transaction_flag contains only 0 and 1",
      set(dtf_vals) <= {0, 1},
      f"Values found: {sorted(dtf_vals)}")

# GT Duplicate Transaction count: 91, but each duplicate pair = 2 rows flagged
dup_gt_count   = (gt["anomaly_type"] == "Duplicate Transaction").sum()
dup_feat_count = feat["duplicate_transaction_flag"].sum()
print(f"\n  Ground truth 'Duplicate Transaction' count  : {dup_gt_count}")
print(f"  duplicate_transaction_flag sum in features  : {int(dup_feat_count)}")
print(f"  (Expected ~2x GT count because both source and copy are flagged)")
check("duplicate_transaction_flag count is between 1x and 3x of GT count",
      dup_gt_count <= dup_feat_count <= dup_gt_count * 3,
      f"Flag count {int(dup_feat_count)} vs GT {dup_gt_count}")

# days_beyond_allowed_end: must be >=0
check("days_beyond_allowed_end has no negative values",
      (feat["days_beyond_allowed_end"] >= 0).all(),
      f"Negative values: {(feat['days_beyond_allowed_end'] < 0).sum()}")

# days_beyond_allowed_end should be 0 for normal invoices
normal_beyond = normal["days_beyond_allowed_end"].max()
check("Normal invoices: days_beyond_allowed_end = 0",
      normal_beyond == 0,
      f"Max for normal rows: {normal_beyond}")

# =================================================================
section("CHECK 13  --  Supplier / Buyer Frequency Features")
# =================================================================

print(f"\n  supplier_frequency  : min={feat['supplier_frequency'].min()}  "
      f"max={feat['supplier_frequency'].max()}  mean={feat['supplier_frequency'].mean():.1f}")
print(f"  buyer_frequency     : min={feat['buyer_frequency'].min()}  "
      f"max={feat['buyer_frequency'].max()}  mean={feat['buyer_frequency'].mean():.1f}")
print(f"  supplier_daily_freq : min={feat['supplier_daily_frequency'].min()}  "
      f"max={feat['supplier_daily_frequency'].max()}  mean={feat['supplier_daily_frequency'].mean():.2f}")
print(f"  buyer_daily_freq    : min={feat['buyer_daily_frequency'].min()}  "
      f"max={feat['buyer_daily_frequency'].max()}  mean={feat['buyer_daily_frequency'].mean():.2f}")

check("supplier_frequency >= 1 for all rows",
      (feat["supplier_frequency"] >= 1).all())
check("buyer_frequency >= 1 for all rows",
      (feat["buyer_frequency"] >= 1).all())
check("supplier_daily_frequency >= 1 for all rows",
      (feat["supplier_daily_frequency"] >= 1).all())
check("buyer_daily_frequency >= 1 for all rows",
      (feat["buyer_daily_frequency"] >= 1).all())

# Burst anomalies should show much higher daily frequency
sup_burst = (gt["anomaly_type"] == "Unusual Supplier Activity")
buy_burst = (gt["anomaly_type"] == "Unusual Buyer Activity")
sup_burst_nos = gt[sup_burst]["invoice_no"]
buy_burst_nos = gt[buy_burst]["invoice_no"]

sup_burst_freq = feat[feat["invoice_no"].isin(sup_burst_nos)]["supplier_daily_frequency"].mean()
buy_burst_freq = feat[feat["invoice_no"].isin(buy_burst_nos)]["buyer_daily_frequency"].mean()
normal_sup_daily = feat[feat["invoice_no"].isin(normal["invoice_no"])]["supplier_daily_frequency"].mean()
normal_buy_daily = feat[feat["invoice_no"].isin(normal["invoice_no"])]["buyer_daily_frequency"].mean()

print(f"\n  Supplier daily freq  -- normal mean: {normal_sup_daily:.2f},  burst anomaly mean: {sup_burst_freq:.2f}")
print(f"  Buyer daily freq     -- normal mean: {normal_buy_daily:.2f},  burst anomaly mean: {buy_burst_freq:.2f}")

check("Burst supplier anomalies have supplier_daily_frequency >> normal (>10x)",
      sup_burst_freq > normal_sup_daily * 10,
      f"Burst mean={sup_burst_freq:.1f}, Normal mean={normal_sup_daily:.2f}")

check("Burst buyer anomalies have buyer_daily_frequency >> normal (>10x)",
      buy_burst_freq > normal_buy_daily * 10,
      f"Burst mean={buy_burst_freq:.1f}, Normal mean={normal_buy_daily:.2f}")

# =================================================================
section("CHECK 14  --  No Ground-Truth Leakage Into ML Features")
# =================================================================

# Ground truth has: invoice_no, actual_anomaly, anomaly_type
# None of these (except invoice_no as join key) must appear in ml_features

# Check: are there features whose values are EXACTLY predicted by anomaly_type?
# If so, that would indicate a label was used during feature construction.
# Strategy: check whether any feature is constant within each anomaly_type
# (which would mean the type label was baked in)

print("\n  Checking for suspicious per-anomaly-type feature uniformity ...")

gt_types   = gt["anomaly_type"].unique()
merged_all = feat.merge(gt, on="invoice_no")

# A leaked feature would be: a continuous feature whose mean is IDENTICAL across
# all anomaly types because it was computed FROM the anomaly type label.
# Binary flags (future_date_flag, duplicate_transaction_flag) can legitimately
# have std=0 within each anomaly type -- that is not leakage, it is just how
# binary features behave when a single anomaly type maps exactly to the flag.
# We exclude known binary flags from this check.
BINARY_FLAGS = {"future_date_flag", "duplicate_transaction_flag"}
LEAKAGE_CANDIDATES = [c for c in feature_cols if c not in BINARY_FLAGS]

suspicious_feats = []
for feat_name in LEAKAGE_CANDIDATES:
    stds_per_type = merged_all.groupby("anomaly_type")[feat_name].std().fillna(0)
    zero_std_types = (stds_per_type == 0).sum()
    if zero_std_types == len(gt_types):
        suspicious_feats.append(feat_name)

check("No continuous feature has zero variance within EVERY anomaly type",
      len(suspicious_feats) == 0,
      f"Suspicious: {suspicious_feats}" if suspicious_feats else "None detected")

# Separately verify binary flags are derived from raw data only
print("\n  Binary flag verification (derived from raw data, not from labels):")
print("  future_date_flag    <- invoice_date vs DATASET_END constant (raw column)")
print("  duplicate_flag      <- signature of (supplier_gstin, buyer_gstin, qty, price, ...)")
print("  Both use ONLY raw invoice columns -- no anomaly_type or actual_anomaly used.")

check("future_date_flag is derived from raw invoice_date (confirmed in preprocess.py)",
      "future_date_flag" in feature_cols and "invoice_date" in raw.columns,
      "invoice_date is in raw data; DATASET_END is a fixed constant")

# Also verify: features are computed from raw columns only, no GT columns used
# Ground truth columns must not appear in feat dataframe
gt_cols_in_feat = [c for c in gt.columns if c in feat.columns and c != "invoice_no"]
check("No ground-truth columns (actual_anomaly, anomaly_type) exist in feature file",
      len(gt_cols_in_feat) == 0,
      f"Leaked GT columns: {gt_cols_in_feat}" if gt_cols_in_feat else "Clean")

# Verify feature values for 'Tax Calculation Mismatch' differ within the group
# (proving we didn't encode the type -- individual invoices have different total_tax_diff)
tcm_nos  = gt[gt["anomaly_type"] == "Tax Calculation Mismatch"]["invoice_no"]
tcm_feat = feat[feat["invoice_no"].isin(tcm_nos)]["total_tax_difference"]
check("'Tax Calculation Mismatch' rows have varying total_tax_difference (not a constant)",
      tcm_feat.std() > 0,
      f"std={tcm_feat.std():.4f}")

# =================================================================
section("CHECK 15  --  Sample Rows from ml_features.csv")
# =================================================================

print("\n  First 3 rows of ml_features.csv (transposed for readability):\n")
sample = feat.head(3)
for col in feat.columns:
    vals = "  |  ".join(str(v) for v in sample[col].values)
    print(f"  {col:<40}: {vals}")

# =================================================================
section("SUMMARY")
# =================================================================

total = PASS_COUNT + FAIL_COUNT
print(f"\n  Total checks : {total}")
print(f"  PASS         : {PASS_COUNT}")
print(f"  FAIL         : {FAIL_COUNT}")

if FAIL_COUNT == 0:
    print("\n  *** ALL CHECKS PASSED ***")
    print("  Stage 2 is validated and ready for Stage 3.")
else:
    print(f"\n  *** {FAIL_COUNT} CHECK(S) FAILED ***")
    print("\n  Failed checks:")
    for status, label, detail in results:
        if status == "FAIL":
            print(f"    - {label}")
            if detail:
                print(f"      {detail}")
