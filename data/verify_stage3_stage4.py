"""
verify_stage3_stage4.py
=======================
Read-only verification and evaluation script for Stage 3 (Isolation Forest)
and Stage 4 (Local Outlier Factor).
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
IF_RESULTS_PATH = os.path.join(ROOT_DIR, "results", "isolation_forest_results.csv")
LOF_RESULTS_PATH = os.path.join(ROOT_DIR, "results", "lof_results.csv")
IF_CODE_PATH = os.path.join(ROOT_DIR, "models", "isolation_forest.py")
LOF_CODE_PATH = os.path.join(ROOT_DIR, "models", "lof.py")

print("=" * 70)
print("  STAGE 3 & 4 READ-ONLY VERIFICATION & EVALUATION")
print("=" * 70)

# =====================================================================
# 1. VERIFY INPUT DATA
# =====================================================================
print("\n--- 1. INPUT DATA VERIFICATION ---")
feat_exists = os.path.exists(FEATURES_PATH)
print(f"ml_features.csv exists: {feat_exists}")
df_feat = pd.read_csv(FEATURES_PATH)
n_rows, n_cols = df_feat.shape
print(f"Shape: {n_rows} rows x {n_cols} columns")

# Check forbidden columns
forbidden = [
    "actual_anomaly", "anomaly_type", "supplier_id", "buyer_id",
    "supplier_gstin", "buyer_gstin"
]
present_forbidden = [c for c in forbidden if c in df_feat.columns]
print(f"Forbidden features present in ml_features.csv: {present_forbidden}")

# Check invoice_no
has_inv_no = "invoice_no" in df_feat.columns
print(f"invoice_no present in file: {has_inv_no}")
inv_no_dtype = str(df_feat["invoice_no"].dtype)
print(f"invoice_no dtype: {inv_no_dtype}")

# ML feature columns
ml_feature_cols = [c for c in df_feat.columns if c != "invoice_no"]
print(f"Number of ML feature columns: {len(ml_feature_cols)}")

# Check non-numeric features
non_numeric_features = [c for c in ml_feature_cols if not pd.api.types.is_numeric_dtype(df_feat[c])]
print(f"Non-numeric features in ML feature set: {non_numeric_features}")

# Missing and Inf values
n_missing = int(df_feat[ml_feature_cols].isnull().sum().sum())
n_inf = int(np.isinf(df_feat[ml_feature_cols].values).sum())
print(f"Missing values in ML features: {n_missing}")
print(f"Infinite values in ML features: {n_inf}")

# Verify code uses same feature set
with open(IF_CODE_PATH, "r", encoding="utf-8") as f:
    if_code = f.read()
with open(LOF_CODE_PATH, "r", encoding="utf-8") as f:
    lof_code = f.read()

if_feat_code_check = 'feature_cols = [c for c in df.columns if c != "invoice_no"]' in if_code
lof_feat_code_check = 'feature_cols = [c for c in df.columns if c != "invoice_no"]' in lof_code
print(f"IF extracts features using all columns except invoice_no: {if_feat_code_check}")
print(f"LOF extracts features using all columns except invoice_no: {lof_feat_code_check}")
print(f"Same feature set used by both models: {if_feat_code_check and lof_feat_code_check}")

# =====================================================================
# 2. VERIFY ISOLATION FOREST IMPLEMENTATION & RESULTS
# =====================================================================
print("\n--- 2. ISOLATION FOREST VERIFICATION ---")
print(f"StandardScaler in IF code: {'StandardScaler' in if_code}")
print(f"IsolationForest in IF code: {'IsolationForest(' in if_code}")
print(f"n_estimators in IF code: {'n_estimators=IF_N_ESTIMATORS' in if_code or 'n_estimators = 200' in if_code}")
print(f"contamination in IF code: {'contamination=IF_CONTAMINATION' in if_code or 'contamination = 0.05' in if_code}")
print(f"random_state in IF code: {'random_state=IF_RANDOM_STATE' in if_code or 'random_state = 42' in if_code}")
print(f"n_jobs in IF code: {'n_jobs=IF_N_JOBS' in if_code or 'n_jobs = -1' in if_code}")

# Check ground_truth usage during training
# In run_isolation_forest, does it take or touch ground truth?
train_func_def = if_code[if_code.find("def run_isolation_forest"):if_code.find("def save_results")]
print(f"Ground truth touched in run_isolation_forest(): {'ground_truth' in train_func_def or 'actual_anomaly' in train_func_def}")

df_if = pd.read_csv(IF_RESULTS_PATH)
print(f"\nresults/isolation_forest_results.csv:")
print(f"  Shape: {df_if.shape}")
print(f"  Columns: {list(df_if.columns)}")
print(f"  Duplicate invoice_no: {df_if['invoice_no'].duplicated().sum()}")
print(f"  Missing values: {df_if.isnull().sum().to_dict()}")

# Check column naming (if_anomaly vs isolation_anomaly)
if_anom_col = "isolation_anomaly" if "isolation_anomaly" in df_if.columns else "if_anomaly"
if_pred_col = "isolation_prediction" if "isolation_prediction" in df_if.columns else "if_prediction"
if_score_col = "isolation_score" if "isolation_score" in df_if.columns else "if_score"

n_if_anom = int(df_if[if_anom_col].sum())
n_if_norm = int((df_if[if_anom_col] == 0).sum())
print(f"  Predicted normal ({if_anom_col}==0): {n_if_norm}")
print(f"  Predicted anomalous ({if_anom_col}==1): {n_if_anom}")

# =====================================================================
# 3. EVALUATE ISOLATION FOREST
# =====================================================================
print("\n--- 3. EVALUATE ISOLATION FOREST ---")
df_gt = pd.read_csv(GT_PATH)
merged_if = df_if.merge(df_gt, on="invoice_no")

y_true = merged_if["actual_anomaly"].values
y_pred_if = merged_if[if_anom_col].values

tp_if = int(((y_true == 1) & (y_pred_if == 1)).sum())
tn_if = int(((y_true == 0) & (y_pred_if == 0)).sum())
fp_if = int(((y_true == 0) & (y_pred_if == 1)).sum())
fn_if = int(((y_true == 1) & (y_pred_if == 0)).sum())

acc_if = (tp_if + tn_if) / len(y_true)
prec_if = tp_if / (tp_if + fp_if) if (tp_if + fp_if) > 0 else 0.0
rec_if = tp_if / (tp_if + fn_if) if (tp_if + fn_if) > 0 else 0.0
f1_if = (2 * prec_if * rec_if / (prec_if + rec_if)) if (prec_if + rec_if) > 0 else 0.0

print(f"Confusion Matrix:")
print(f"  TP: {tp_if}, TN: {tn_if}, FP: {fp_if}, FN: {fn_if}")
print(f"Accuracy : {acc_if:.4f} ({acc_if*100:.2f}%)")
print(f"Precision: {prec_if:.4f}")
print(f"Recall   : {rec_if:.4f}")
print(f"F1 Score : {f1_if:.4f}")

print("\nIF Anomaly Detection Rate by Anomaly Type:")
if_rates = {}
for atype in sorted(df_gt[df_gt["actual_anomaly"] == 1]["anomaly_type"].unique()):
    sub = merged_if[merged_if["anomaly_type"] == atype]
    tot = len(sub)
    det = int(sub[if_anom_col].sum())
    r = det / tot if tot > 0 else 0.0
    if_rates[atype] = (tot, det, r)
    print(f"  {atype:<30} | {tot:>5} | {det:>8} | {r:>6.1%}")

# =====================================================================
# 4. VERIFY LOF IMPLEMENTATION & RESULTS
# =====================================================================
print("\n--- 4. LOF VERIFICATION ---")
print(f"StandardScaler in LOF code: {'StandardScaler' in lof_code}")
print(f"LocalOutlierFactor in LOF code: {'LocalOutlierFactor(' in lof_code}")
print(f"n_neighbors in LOF code: {'n_neighbors=LOF_N_NEIGHBORS' in lof_code or 'n_neighbors = 20' in lof_code}")
print(f"contamination in LOF code: {'contamination=LOF_CONTAMINATION' in lof_code or 'contamination = 0.05' in lof_code}")

train_func_def_lof = lof_code[lof_code.find("def run_lof"):lof_code.find("def save_results")]
print(f"Ground truth touched in run_lof(): {'ground_truth' in train_func_def_lof or 'actual_anomaly' in train_func_def_lof}")

df_lof = pd.read_csv(LOF_RESULTS_PATH)
print(f"\nresults/lof_results.csv:")
print(f"  Shape: {df_lof.shape}")
print(f"  Columns: {list(df_lof.columns)}")
print(f"  Duplicate invoice_no: {df_lof['invoice_no'].duplicated().sum()}")
print(f"  Missing values: {df_lof.isnull().sum().to_dict()}")

lof_anom_col = "lof_anomaly"
lof_pred_col = "lof_prediction"
lof_score_col = "lof_score"

n_lof_anom = int(df_lof[lof_anom_col].sum())
n_lof_norm = int((df_lof[lof_anom_col] == 0).sum())
print(f"  Predicted normal ({lof_anom_col}==0): {n_lof_norm}")
print(f"  Predicted anomalous ({lof_anom_col}==1): {n_lof_anom}")

# =====================================================================
# 5. EVALUATE LOF
# =====================================================================
print("\n--- 5. EVALUATE LOF ---")
merged_lof = df_lof.merge(df_gt, on="invoice_no")

y_pred_lof = merged_lof[lof_anom_col].values

tp_lof = int(((y_true == 1) & (y_pred_lof == 1)).sum())
tn_lof = int(((y_true == 0) & (y_pred_lof == 0)).sum())
fp_lof = int(((y_true == 0) & (y_pred_lof == 1)).sum())
fn_lof = int(((y_true == 1) & (y_pred_lof == 0)).sum())

acc_lof = (tp_lof + tn_lof) / len(y_true)
prec_lof = tp_lof / (tp_lof + fp_lof) if (tp_lof + fp_lof) > 0 else 0.0
rec_lof = tp_lof / (tp_lof + fn_lof) if (tp_lof + fn_lof) > 0 else 0.0
f1_lof = (2 * prec_lof * rec_lof / (prec_lof + rec_lof)) if (prec_lof + rec_lof) > 0 else 0.0

print(f"Confusion Matrix:")
print(f"  TP: {tp_lof}, TN: {tn_lof}, FP: {fp_lof}, FN: {fn_lof}")
print(f"Accuracy : {acc_lof:.4f} ({acc_lof*100:.2f}%)")
print(f"Precision: {prec_lof:.4f}")
print(f"Recall   : {rec_lof:.4f}")
print(f"F1 Score : {f1_lof:.4f}")

print("\nLOF Anomaly Detection Rate by Anomaly Type:")
lof_rates = {}
for atype in sorted(df_gt[df_gt["actual_anomaly"] == 1]["anomaly_type"].unique()):
    sub = merged_lof[merged_lof["anomaly_type"] == atype]
    tot = len(sub)
    det = int(sub[lof_anom_col].sum())
    r = det / tot if tot > 0 else 0.0
    lof_rates[atype] = (tot, det, r)
    print(f"  {atype:<30} | {tot:>5} | {det:>8} | {r:>6.1%}")

# =====================================================================
# 6. COMPARISON TABLES
# =====================================================================
print("\n--- 6. COMPARISON TABLES ---")
print("| Model | Accuracy | Precision | Recall | F1 Score |")
print("|---|---|---|---|---|")
print(f"| Isolation Forest | {acc_if*100:.2f}% | {prec_if:.4f} | {rec_if:.4f} | {f1_if:.4f} |")
print(f"| LOF | {acc_lof*100:.2f}% | {prec_lof:.4f} | {rec_lof:.4f} | {f1_lof:.4f} |")

print("\n| Anomaly Type | Isolation Forest Detection % | LOF Detection % |")
print("|---|---|---|")
for atype in sorted(if_rates.keys()):
    if_pct = if_rates[atype][2] * 100
    lof_pct = lof_rates[atype][2] * 100
    print(f"| {atype} | {if_pct:.1f}% | {lof_pct:.1f}% |")

print("\n[VERIFICATION SCRIPT COMPLETED]")
