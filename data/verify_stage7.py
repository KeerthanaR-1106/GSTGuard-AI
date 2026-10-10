"""
verify_stage7.py
================
Read-only verification script for Stage 7: Ensemble Risk Score and Classification.
DOES NOT MODIFY ANY FILE OR MODEL CODE.
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

IF_PATH = os.path.join(ROOT_DIR, "results", "isolation_forest_results.csv")
LOF_PATH = os.path.join(ROOT_DIR, "results", "lof_results.csv")
Z_PATH = os.path.join(ROOT_DIR, "results", "zscore_results.csv")
ENS_PATH = os.path.join(ROOT_DIR, "results", "ensemble_results.csv")
GT_PATH = os.path.join(ROOT_DIR, "data", "ground_truth.csv")
ENS_CODE_PATH = os.path.join(ROOT_DIR, "models", "ensemble.py")

print("=" * 70)
print("  STAGE 7 READ-ONLY VERIFICATION")
print("=" * 70)

# 1. VERIFY INPUTS
print("\n--- 1. VERIFY INPUTS ---")
df_if = pd.read_csv(IF_PATH)
df_lof = pd.read_csv(LOF_PATH)
df_z = pd.read_csv(Z_PATH)

print(f"  IF results rows : {len(df_if):,}")
print(f"  LOF results rows: {len(df_lof):,}")
print(f"  Z results rows  : {len(df_z):,}")

set_if = set(df_if["invoice_no"])
set_lof = set(df_lof["invoice_no"])
set_z = set(df_z["invoice_no"])

print(f"  Identical invoice_no sets: {set_if == set_lof == set_z}")
print(f"  All unique invoice_no: {df_if['invoice_no'].is_unique and df_lof['invoice_no'].is_unique and df_z['invoice_no'].is_unique}")
print(f"  IF missing values: {df_if.isnull().sum().sum()}")
print(f"  LOF missing values: {df_lof.isnull().sum().sum()}")
print(f"  Z missing values: {df_z.isnull().sum().sum()}")

# 2. VERIFY ENSEMBLE CODE & WEIGHTS
print("\n--- 2. VERIFY ENSEMBLE CODE & WEIGHTS ---")
with open(ENS_CODE_PATH, "r", encoding="utf-8") as f:
    ens_code = f.read()

print(f"  WEIGHT_ISOLATION_FOREST in code: {'WEIGHT_ISOLATION_FOREST = 0.40' in ens_code}")
print(f"  WEIGHT_LOF in code: {'WEIGHT_LOF              = 0.40' in ens_code or 'WEIGHT_LOF = 0.40' in ens_code}")
print(f"  WEIGHT_ZSCORE in code: {'WEIGHT_ZSCORE           = 0.20' in ens_code or 'WEIGHT_ZSCORE = 0.20' in ens_code}")
print(f"  Weights sum check in code: {'assert abs((WEIGHT_ISOLATION_FOREST + WEIGHT_LOF + WEIGHT_ZSCORE) - 1.0) < 1e-6' in ens_code}")

# 3. VERIFY SCORE NORMALIZATION & FINAL RISK SCORE
print("\n--- 3. VERIFY SCORE NORMALIZATION & FINAL RISK SCORE ---")
df_ens = pd.read_csv(ENS_PATH)
print(f"  Minimum risk_score : {df_ens['risk_score'].min():.4f}")
print(f"  Maximum risk_score : {df_ens['risk_score'].max():.4f}")
print(f"  Average risk_score : {df_ens['risk_score'].mean():.4f}")
print(f"  Missing risk_score : {df_ens['risk_score'].isnull().sum()}")
print(f"  Infinite risk_score: {np.isinf(df_ens['risk_score']).sum()}")
print(f"  Risk scores strictly within [0, 100]: {(df_ens['risk_score'] >= 0.0).all() and (df_ens['risk_score'] <= 100.0).all()}")

# 4. VERIFY RISK CLASSIFICATION
print("\n--- 4. VERIFY RISK CLASSIFICATION ---")
level_counts = df_ens["risk_level"].value_counts().to_dict()
print(f"  Low Risk count   : {level_counts.get('Low Risk', 0):,}")
print(f"  Medium Risk count: {level_counts.get('Medium Risk', 0):,}")
print(f"  High Risk count  : {level_counts.get('High Risk', 0):,}")
print(f"  Valid levels only: {set(df_ens['risk_level'].unique()) <= {'Low Risk', 'Medium Risk', 'High Risk'}}")
print(f"  Total classified : {sum(level_counts.values()):,}")

# Check threshold consistency
low_ok = (df_ens[df_ens["risk_level"] == "Low Risk"]["risk_score"] < 40.0).all()
med_ok = ((df_ens[df_ens["risk_level"] == "Medium Risk"]["risk_score"] >= 40.0) & 
          (df_ens[df_ens["risk_level"] == "Medium Risk"]["risk_score"] < 70.0)).all()
high_ok = (df_ens[df_ens["risk_level"] == "High Risk"]["risk_score"] >= 70.0).all()
print(f"  Threshold compliance: Low (<40): {low_ok}, Med (40-69.99): {med_ok}, High (>=70): {high_ok}")

# 5. VERIFY ANOMALY INFORMATION & AUDIT TERMINOLOGY
print("\n--- 5. VERIFY ANOMALY INFORMATION & AUDIT TERMINOLOGY ---")
fraud_mentions = df_ens["explanation"].str.contains(r"\bfraud\b", case=False).sum()
print(f"  Explanations containing legal accusation 'fraud': {fraud_mentions}")
suspicious_mentions = df_ens["explanation"].str.contains("suspicious|anomalous|review|variance|deviation", case=False).sum()
print(f"  Explanations using compliance audit terminology: {suspicious_mentions} of {len(df_ens)}")

# 6. VERIFY OUTPUT FILE
print("\n--- 6. VERIFY OUTPUT FILE ---")
expected_cols = [
    "invoice_no", "isolation_score", "isolation_anomaly",
    "lof_score", "lof_anomaly", "max_abs_zscore", "zscore_anomaly",
    "risk_score", "risk_level", "anomaly_type", "explanation"
]
print(f"  File: results/ensemble_results.csv exists: {os.path.exists(ENS_PATH)}")
print(f"  Columns match expected exactly: {list(df_ens.columns) == expected_cols}")
print(f"  20,000 records: {len(df_ens) == 20000}")
print(f"  Unique invoice_no: {df_ens['invoice_no'].is_unique}")

# 7. SAMPLE RECORDS VERIFICATION
print("\n--- 7. SAMPLE RECORDS VERIFICATION ---")
# 1 Low Risk, 1 Medium Risk, 1 High Risk, 1 IF flagged, 1 LOF flagged, 1 Z flagged
s_low = df_ens[df_ens["risk_level"] == "Low Risk"].iloc[0]
s_med = df_ens[df_ens["risk_level"] == "Medium Risk"].iloc[0]
s_high = df_ens[df_ens["risk_level"] == "High Risk"].iloc[0]
s_if = df_ens[(df_ens["isolation_anomaly"] == 1) & (df_ens["lof_anomaly"] == 0)].iloc[0]
s_lof = df_ens[(df_ens["lof_anomaly"] == 1) & (df_ens["isolation_anomaly"] == 0)].iloc[0]
s_z = df_ens[(df_ens["zscore_anomaly"] == 1) & (df_ens["isolation_anomaly"] == 0) & (df_ens["lof_anomaly"] == 0)].iloc[0]

samples = [
    ("Low Risk Sample", s_low),
    ("Medium Risk Sample", s_med),
    ("High Risk Sample", s_high),
    ("IF Dominant Sample", s_if),
    ("LOF Dominant Sample", s_lof),
    ("Z-Score Dominant Sample", s_z),
]

for label, s in samples:
    print(f"\n[{label}] Invoice: {s['invoice_no']}")
    print(f"  Isolation Result: anomaly={s['isolation_anomaly']} (score={s['isolation_score']:.4f})")
    print(f"  LOF Result      : anomaly={s['lof_anomaly']} (score={s['lof_score']:.4f})")
    print(f"  Z-score Result  : anomaly={s['zscore_anomaly']} (max_z={s['max_abs_zscore']:.4f})")
    print(f"  Final Risk Score: {s['risk_score']:.2f}")
    print(f"  Risk Level      : {s['risk_level']}")
    print(f"  Anomaly Type    : {s['anomaly_type']}")
    print(f"  Explanation     : {s['explanation'][:100]}...")

# 8. GROUND TRUTH SEPARATION AUDIT
print("\n--- 8. GROUND TRUTH SEPARATION AUDIT ---")
gt_in_code = "ground_truth" in ens_code
act_anom_in_code = "actual_anomaly" in ens_code
print(f"  ground_truth.csv accessed in models/ensemble.py: {gt_in_code}")
print(f"  actual_anomaly used in models/ensemble.py: {act_anom_in_code}")

# 9. OPTIONAL EVALUATION VS GROUND TRUTH
print("\n--- 9. OPTIONAL EVALUATION VS GROUND TRUTH ---")
df_gt = pd.read_csv(GT_PATH)
m_eval = df_ens.merge(df_gt, on="invoice_no")

# Consider Medium + High Risk as predicted anomaly in an audit workflow (score >= 40)
y_true = m_eval["actual_anomaly"].values
y_pred_med_high = (m_eval["risk_score"] >= 40.0).astype(int).values

tp = int(((y_true == 1) & (y_pred_med_high == 1)).sum())
tn = int(((y_true == 0) & (y_pred_med_high == 0)).sum())
fp = int(((y_true == 0) & (y_pred_med_high == 1)).sum())
fn = int(((y_true == 1) & (y_pred_med_high == 0)).sum())

prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
acc = (tp + tn) / len(y_true)

print(f"  Ensemble Flag (Score >= 40: Medium + High Risk):")
print(f"    TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"    Accuracy : {acc*100:.2f}%")
print(f"    Precision: {prec:.4f}")
print(f"    Recall   : {rec:.4f} ({tp}/{y_true.sum()} anomalies flagged)")
print(f"    F1 Score : {f1:.4f}")

# High Risk only (Score >= 70)
y_pred_high = (m_eval["risk_score"] >= 70.0).astype(int).values
tp_h = int(((y_true == 1) & (y_pred_high == 1)).sum())
fp_h = int(((y_true == 0) & (y_pred_high == 1)).sum())
prec_h = tp_h / (tp_h + fp_h) if (tp_h + fp_h) > 0 else 0.0
print(f"\n  High Risk Only (Score >= 70):")
print(f"    TP: {tp_h}, FP: {fp_h}")
print(f"    Precision: {prec_h*100:.2f}% (High confidence tier)")

print("\n[VERIFICATION COMPLETED]")
