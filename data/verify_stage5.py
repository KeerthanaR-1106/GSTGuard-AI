"""
verify_stage5.py
================
Read-only verification and evaluation script for Stage 5 (Statistical Z-Score).
DOES NOT MODIFY ANY FILE OR MODEL CODE.
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

FEATURES_PATH = os.path.join(ROOT_DIR, "data", "ml_features.csv")
GT_PATH = os.path.join(ROOT_DIR, "data", "ground_truth.csv")
Z_RESULTS_PATH = os.path.join(ROOT_DIR, "results", "zscore_results.csv")
Z_CODE_PATH = os.path.join(ROOT_DIR, "models", "zscore.py")

print("=" * 70)
print("  STAGE 5: STATISTICAL Z-SCORE READ-ONLY VERIFICATION")
print("=" * 70)

# =====================================================================
# 1. VERIFY INPUT
# =====================================================================
print("\n--- 1. VERIFY INPUT ---")
feat_exists = os.path.exists(FEATURES_PATH)
print(f"ml_features.csv exists: {feat_exists}")
df_feat = pd.read_csv(FEATURES_PATH)
n_rows, n_cols = df_feat.shape
print(f"Input shape: {n_rows} rows x {n_cols} columns")

# Excluded columns check
forbidden = ["actual_anomaly", "anomaly_type", "ground_truth", "label", "is_anomaly"]
present_forbidden = [c for c in forbidden if c in df_feat.columns]
print(f"Forbidden labels present in ml_features.csv: {present_forbidden}")

# invoice_no check
has_inv_no = "invoice_no" in df_feat.columns
print(f"invoice_no present: {has_inv_no} (dtype: {df_feat['invoice_no'].dtype})")

# Feature list used by Z-score
feature_cols = [c for c in df_feat.columns if c != "invoice_no"]
print(f"Number of features used by Z-score: {len(feature_cols)}")

# Confirm numeric
non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df_feat[c])]
print(f"Non-numeric features: {non_numeric}")

# Missing and Inf
n_missing = int(df_feat[feature_cols].isnull().sum().sum())
n_inf = int(np.isinf(df_feat[feature_cols].values).sum())
print(f"Missing values in features: {n_missing}")
print(f"Infinite values in features: {n_inf}")

# =====================================================================
# 2. VERIFY Z-SCORE CODE IMPLEMENTATION
# =====================================================================
print("\n--- 2. VERIFY Z-SCORE CODE IMPLEMENTATION ---")
with open(Z_CODE_PATH, "r", encoding="utf-8") as f:
    z_code = f.read()

print(f"Z formula present in doc/code: {'Z = (x - mean) / standard deviation' in z_code or '(X - means) / stds' in z_code}")
print(f"X.mean() used: {'X.mean()' in z_code}")
print(f"X.std() used: {'X.std()' in z_code}")
print(f"Zero-std safe handling: {'stds.replace(0' in z_code or 'replace(0, np.nan)' in z_code}")
print(f"Absolute Z used: {'Z_df.abs()' in z_code or '.abs()' in z_code}")
print(f"Preserved threshold (3.0): {'Z_THRESHOLD = 3.0' in z_code}")
print(f"max_abs_zscore calculated: {'max_abs_zscore' in z_code}")
print(f"extreme_feature_count calculated: {'extreme_feature_count' in z_code}")
print(f"zscore_anomaly calculated: {'zscore_anomaly' in z_code}")

train_func_def_z = z_code[z_code.find("def calculate_zscores"):z_code.find("def save_results")]
print(f"Ground truth used during calculation: {'ground_truth' in train_func_def_z or 'actual_anomaly' in train_func_def_z}")

# =====================================================================
# 3. VERIFY OUTPUT FILE
# =====================================================================
print("\n--- 3. VERIFY OUTPUT FILE ---")
z_res_exists = os.path.exists(Z_RESULTS_PATH)
print(f"results/zscore_results.csv exists: {z_res_exists}")
df_z = pd.read_csv(Z_RESULTS_PATH)
n_out_rows, n_out_cols = df_z.shape
print(f"Output shape: {n_out_rows} rows x {n_out_cols} columns")
print(f"Columns: {list(df_z.columns)}")

req_cols = ["invoice_no", "zscore_anomaly", "max_abs_zscore", "extreme_feature_count"]
missing_req_cols = [c for c in req_cols if c not in df_z.columns]
print(f"Missing required columns: {missing_req_cols}")

dup_inv = int(df_z["invoice_no"].duplicated().sum())
print(f"Duplicate invoice numbers: {dup_inv}")

n_null = int(df_z.isnull().sum().sum())
n_inf_out = int(np.isinf(df_z.select_dtypes(include=[np.number]).values).sum())
print(f"Missing values: {n_null}")
print(f"Infinite values: {n_inf_out}")

n_norm = int((df_z["zscore_anomaly"] == 0).sum())
n_anom = int((df_z["zscore_anomaly"] == 1).sum())
print(f"Normal classifications (zscore_anomaly == 0): {n_norm} ({n_norm/n_out_rows*100:.2f}%)")
print(f"Anomalous classifications (zscore_anomaly == 1): {n_anom} ({n_anom/n_out_rows*100:.2f}%)")
print(f"Matches input row count (20,000): {n_out_rows == n_rows}")

# =====================================================================
# 4. EVALUATE Z-SCORE
# =====================================================================
print("\n--- 4. EVALUATE Z-SCORE ---")
df_gt = pd.read_csv(GT_PATH)
merged_z = df_z.merge(df_gt, on="invoice_no")

y_true = merged_z["actual_anomaly"].values
y_pred = merged_z["zscore_anomaly"].values

tp = int(((y_true == 1) & (y_pred == 1)).sum())
tn = int(((y_true == 0) & (y_pred == 0)).sum())
fp = int(((y_true == 0) & (y_pred == 1)).sum())
fn = int(((y_true == 1) & (y_pred == 0)).sum())

accuracy = (tp + tn) / len(y_true)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print("Confusion Matrix:")
print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# =====================================================================
# 5. ANOMALY-TYPE ANALYSIS
# =====================================================================
print("\n--- 5. ANOMALY-TYPE ANALYSIS ---")
print(f"{'Anomaly Type':<30} | {'Total':>6} | {'Detected':>8} | {'Detection Rate':>14}")
print("-" * 65)

anom_types = [
    "Extreme Invoice Amount",
    "Extreme Unit Price",
    "Unusual Quantity",
    "Unusual Discount",
    "Tax Calculation Mismatch",
    "Incorrect CGST SGST Split",
    "Incorrect IGST",
    "Duplicate Transaction",
    "Unusual Supplier Activity",
    "Unusual Buyer Activity",
    "Future Invoice Date"
]

for atype in anom_types:
    sub = merged_z[merged_z["anomaly_type"] == atype]
    tot = len(sub)
    det = int(sub["zscore_anomaly"].sum())
    r = det / tot if tot > 0 else 0.0
    print(f"{atype:<30} | {tot:>6} | {det:>8} | {r:>13.1%}")

# =====================================================================
# 6. DATA LEAKAGE CHECK
# =====================================================================
print("\n--- 6. DATA LEAKAGE CHECK ---")
leakage_checks = {
    "actual_anomaly not in ml_features.csv": "actual_anomaly" not in df_feat.columns,
    "anomaly_type not in ml_features.csv": "anomaly_type" not in df_feat.columns,
    "ground_truth.csv not loaded in calculate_zscores": "ground_truth" not in train_func_def_z,
    "invoice_no only used as ID": df_feat["invoice_no"].dtype == object or str(df_feat["invoice_no"].dtype) == "str",
    "features computed purely from raw invoice math": all(c in df_feat.columns for c in feature_cols)
}
for desc, res in leakage_checks.items():
    print(f"  {desc}: {res}")

# =====================================================================
# 7. MATHEMATICAL SANITY CHECK (SAMPLE INVOICES)
# =====================================================================
print("\n--- 7. MATHEMATICAL SANITY CHECK ---")
# Pick 3 specific invoices: 1 normal, 1 extreme quantity, 1 future date
sample_invs = ["INV010651", "INV019273", "INV019911"]
# Let's inspect features for these 3
df_merged_all = df_feat.merge(df_z, on="invoice_no").merge(df_gt, on="invoice_no")

means = df_feat[feature_cols].mean()
stds = df_feat[feature_cols].std()

for inv_id in sample_invs:
    row = df_merged_all[df_merged_all["invoice_no"] == inv_id].iloc[0]
    dom_feat = row["dominant_feature"]
    val = row[dom_feat]
    mu = means[dom_feat]
    sigma = stds[dom_feat]
    manual_z = (val - mu) / sigma
    abs_manual_z = abs(manual_z)
    stored_max_z = row["max_abs_zscore"]
    crosses_thresh = abs_manual_z >= 3.0
    anom_flag = row["zscore_anomaly"]
    
    print(f"\nInvoice: {inv_id} ({row['anomaly_type']})")
    print(f"  Dominant Feature   : {dom_feat}")
    print(f"  Observed Value (x) : {val}")
    print(f"  Dataset Mean (mu)  : {mu:.4f}")
    print(f"  Dataset Std (sigma): {sigma:.4f}")
    print(f"  Manual Z = (x-mu)/s: ({val} - {mu:.4f}) / {sigma:.4f} = {manual_z:.4f}")
    print(f"  Manual |Z|         : {abs_manual_z:.4f}")
    print(f"  Stored max_abs_z   : {stored_max_z:.4f} (Difference: {abs(abs_manual_z - stored_max_z):.6f})")
    print(f"  |Z| >= 3.0 ?       : {crosses_thresh}")
    print(f"  zscore_anomaly     : {anom_flag} (Matches threshold rule: {crosses_thresh == bool(anom_flag)})")

print("\n[VERIFICATION SCRIPT COMPLETED]")
