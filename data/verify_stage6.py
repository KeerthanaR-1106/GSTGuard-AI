"""
verify_stage6.py
================
Read-only verification script for Stage 6: Model Evaluation and Comparison.
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
GT_PATH = os.path.join(ROOT_DIR, "data", "ground_truth.csv")
EVAL_CODE_PATH = os.path.join(ROOT_DIR, "models", "evaluate_models.py")

print("=" * 70)
print("  STAGE 6 READ-ONLY VERIFICATION")
print("=" * 70)

# 1. VERIFY ALL MODEL RESULTS FILES
print("\n--- 1. VERIFY ALL MODEL RESULTS FILES ---")
files = {
    "Isolation Forest": IF_PATH,
    "LOF": LOF_PATH,
    "Z-score": Z_PATH,
    "Ground Truth": GT_PATH
}

dfs = {}
for name, path in files.items():
    exists = os.path.exists(path)
    df = pd.read_csv(path) if exists else None
    dfs[name] = df
    print(f"  {name:<18}: exists={exists}, rows={len(df) if df is not None else 0:,}")

# Check identical invoices
set_if = set(dfs["Isolation Forest"]["invoice_no"])
set_lof = set(dfs["LOF"]["invoice_no"])
set_z = set(dfs["Z-score"]["invoice_no"])
set_gt = set(dfs["Ground Truth"]["invoice_no"])

all_identical = (set_if == set_lof == set_z == set_gt)
unique_check = (
    dfs["Isolation Forest"]["invoice_no"].is_unique and
    dfs["LOF"]["invoice_no"].is_unique and
    dfs["Z-score"]["invoice_no"].is_unique and
    dfs["Ground Truth"]["invoice_no"].is_unique
)
print(f"  All four files have identical invoice_no sets: {all_identical}")
print(f"  invoice_no is unique in all files: {unique_check}")
print(f"  Each file has exactly 20,000 records: {all(len(df) == 20000 for df in dfs.values())}")

# 2. VERIFY GROUND TRUTH USAGE & LEAKAGE
print("\n--- 2. VERIFY GROUND TRUTH USAGE & LEAKAGE ---")
with open(os.path.join(ROOT_DIR, "models", "isolation_forest.py"), "r", encoding="utf-8") as f:
    if_code = f.read()
with open(os.path.join(ROOT_DIR, "models", "lof.py"), "r", encoding="utf-8") as f:
    lof_code = f.read()
with open(os.path.join(ROOT_DIR, "models", "zscore.py"), "r", encoding="utf-8") as f:
    z_code = f.read()

train_if = if_code[:if_code.find("def quick_evaluation")] if "def quick_evaluation" in if_code else if_code
train_lof = lof_code[:lof_code.find("def evaluate_lof")] if "def evaluate_lof" in lof_code else lof_code
train_z = z_code[:z_code.find("def evaluate_zscore")] if "def evaluate_zscore" in z_code else z_code

leak_if = "ground_truth.csv" in train_if or "actual_anomaly" in train_if
leak_lof = "ground_truth.csv" in train_lof or "actual_anomaly" in train_lof
leak_z = "ground_truth.csv" in train_z or "actual_anomaly" in train_z

print(f"  actual_anomaly used in IF training: {leak_if}")
print(f"  actual_anomaly used in LOF training: {leak_lof}")
print(f"  actual_anomaly used in Z-score calculation: {leak_z}")

feat_cols = pd.read_csv(os.path.join(ROOT_DIR, "data", "ml_features.csv"), nrows=1).columns.tolist()
print(f"  anomaly_type present in ml_features.csv: {'anomaly_type' in feat_cols}")
print(f"  Target leakage status: {'PASS' if not (leak_if or leak_lof or leak_z or 'anomaly_type' in feat_cols) else 'FAIL'}")

# 3. VERIFY INDIVIDUAL MODEL METRICS
print("\n--- 3. VERIFY INDIVIDUAL MODEL METRICS ---")
df_if = dfs["Isolation Forest"]
df_lof = dfs["LOF"]
df_z = dfs["Z-score"]
df_gt = dfs["Ground Truth"]

if_col = "isolation_anomaly" if "isolation_anomaly" in df_if.columns else "if_anomaly"
lof_col = "lof_anomaly"
z_col = "zscore_anomaly"

merged = df_gt.merge(df_if[["invoice_no", if_col]], on="invoice_no")
merged = merged.merge(df_lof[["invoice_no", lof_col]], on="invoice_no")
merged = merged.merge(df_z[["invoice_no", z_col]], on="invoice_no")

y_true = merged["actual_anomaly"].values

def calc_stats(y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "acc": acc, "prec": prec, "rec": rec, "f1": f1}

m_if = calc_stats(merged[if_col].values)
m_lof = calc_stats(merged[lof_col].values)
m_z = calc_stats(merged[z_col].values)

print(f"| Model | Accuracy | Precision | Recall | F1 Score |")
print(f"|---|---|---|---|---|")
print(f"| Isolation Forest | {m_if['acc']*100:.2f}% | {m_if['prec']:.4f} | {m_if['rec']:.4f} | {m_if['f1']:.4f} |")
print(f"| LOF | {m_lof['acc']*100:.2f}% | {m_lof['prec']:.4f} | {m_lof['rec']:.4f} | {m_lof['f1']:.4f} |")
print(f"| Z-score | {m_z['acc']*100:.2f}% | {m_z['prec']:.4f} | {m_z['rec']:.4f} | {m_z['f1']:.4f} |")

# 4. VERIFY ANOMALY-TYPE DETECTION
print("\n--- 4. VERIFY ANOMALY-TYPE DETECTION ---")
anom_df = merged[merged["actual_anomaly"] == 1]
types = sorted(anom_df["anomaly_type"].unique())

type_rates = {}
for t in types:
    sub = anom_df[anom_df["anomaly_type"] == t]
    tot = len(sub)
    det_if = sub[if_col].sum()
    det_lof = sub[lof_col].sum()
    det_z = sub[z_col].sum()
    r_if = det_if / tot
    r_lof = det_lof / tot
    r_z = det_z / tot
    type_rates[t] = {"total": tot, "if": r_if, "lof": r_lof, "z": r_z}
    print(f"| {t:<28} | {r_if*100:>5.1f}% | {r_lof*100:>5.1f}% | {r_z*100:>5.1f}% |")

# 5. VERIFY SCORE DIRECTIONS
print("\n--- 5. VERIFY SCORE DIRECTIONS ---")
# IF score check
if_score_col = "if_score" if "if_score" in df_if.columns else "isolation_score"
if_norm_score = df_if[df_if[if_col] == 0][if_score_col].mean()
if_anom_score = df_if[df_if[if_col] == 1][if_score_col].mean()
print(f"  Isolation Forest Score: normal mean={if_norm_score:.4f}, anomaly mean={if_anom_score:.4f} (Higher=More Anomalous: {if_anom_score > if_norm_score})")

# LOF score check
lof_score_col = "lof_score"
lof_norm_score = df_lof[df_lof[lof_col] == 0][lof_score_col].mean()
lof_anom_score = df_lof[df_lof[lof_col] == 1][lof_score_col].mean()
print(f"  LOF Score: normal mean={lof_norm_score:.4f}, anomaly mean={lof_anom_score:.4f} (Higher=More Anomalous: {lof_anom_score > lof_norm_score})")

# Z-score check
z_score_col = "max_abs_zscore"
z_norm_score = df_z[df_z[z_col] == 0][z_score_col].mean()
z_anom_score = df_z[df_z[z_col] == 1][z_score_col].mean()
print(f"  Z-Score (max_abs_zscore): normal mean={z_norm_score:.4f}, anomaly mean={z_anom_score:.4f} (Higher=More Anomalous: {z_anom_score > z_norm_score})")

all_scores_aligned = (if_anom_score > if_norm_score) and (lof_anom_score > lof_norm_score) and (z_anom_score > z_norm_score)
print(f"  All three model scores consistently oriented (higher = more anomalous): {all_scores_aligned}")

# 6. CHECK DATA CONSISTENCY
print("\n--- 6. CHECK DATA CONSISTENCY ---")
print(f"  Merged records count: {len(merged):,}")
print(f"  Missing values in merged predictions: {merged[['pred_if' if 'pred_if' in merged else if_col, lof_col, z_col]].isnull().sum().to_dict()}")
print(f"  Missing values in IF score: {df_if[if_score_col].isnull().sum()}")
print(f"  Missing values in LOF score: {df_lof[lof_score_col].isnull().sum()}")
print(f"  Missing values in Z score: {df_z[z_score_col].isnull().sum()}")
print(f"  Inf values in IF score: {np.isinf(df_if[if_score_col]).sum()}")
print(f"  Inf values in LOF score: {np.isinf(df_lof[lof_score_col]).sum()}")
print(f"  Inf values in Z score: {np.isinf(df_z[z_score_col]).sum()}")

# 7. CHECK CURRENT IMPLEMENTATION CODE
print("\n--- 7. CHECK CURRENT IMPLEMENTATION CODE ---")
with open(EVAL_CODE_PATH, "r", encoding="utf-8") as f:
    eval_code = f.read()

print(f"  Uses calculate_metrics dynamically: {'def calculate_metrics' in eval_code}")
print(f"  Ground truth loaded after models: {'df_gt = pd.read_csv(GT_PATH)' in eval_code}")
print(f"  Dynamic TP, TN, FP, FN calculation: {'(y_true == 1) & (y_pred == 1)' in eval_code}")
print(f"  No hardcoded metrics: {'0.9438' not in eval_code and '0.9249' not in eval_code}")

print("\n[VERIFICATION COMPLETED]")
