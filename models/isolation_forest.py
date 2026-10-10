"""
isolation_forest.py
===================
Stage 3: Unsupervised anomaly detection using Isolation Forest.

What is Isolation Forest?
--------------------------
Isolation Forest works by randomly selecting a feature and then randomly
selecting a split value between the minimum and maximum of that feature.
Anomalous points are "isolated" (separated from the rest) in fewer splits
because they are rare and different.  Normal points require many more splits
to isolate, so they have a deeper average path length in the trees.

The anomaly score is the average path length needed to isolate a point across
all trees.  A shorter path = more anomalous.

Key parameter choices:
  - n_estimators=200  : 200 trees gives stable, reproducible scores
  - contamination=0.05: we expect ~5% anomalies (matches our dataset design)
  - random_state=42   : ensures reproducibility across runs
  - n_jobs=-1         : use all CPU cores for speed

Output convention:
  predict() returns -1 for anomaly, +1 for normal
  score_samples() returns negative scores; more negative = more anomalous
  We convert: anomaly_score = -score_samples() so higher = more anomalous

Reads  : data/ml_features.csv
Outputs: results/isolation_forest_results.csv

Run:
    python models/isolation_forest.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(BASE_DIR)
FEATURES_PATH = os.path.join(ROOT_DIR, "data", "ml_features.csv")
RESULTS_PATH  = os.path.join(ROOT_DIR, "results", "isolation_forest_results.csv")

# ── Hyperparameters (all in one place for easy tuning) ───────────────────────
IF_N_ESTIMATORS  = 200    # number of isolation trees
IF_CONTAMINATION = 0.05   # expected fraction of anomalies (~5%)
IF_RANDOM_STATE  = 42     # fixed seed for reproducibility
IF_N_JOBS        = -1     # use all CPU cores


# =============================================================================
def load_features(path: str):
    """
    Load ml_features.csv, separate invoice_no from numeric ML features,
    and apply StandardScaler.

    Why StandardScaler?
    Isolation Forest is based on random splits between min and max of each
    feature.  Features with very large value ranges (e.g. taxable_value in
    crores) would dominate the splits if left unscaled.  Scaling brings all
    features to the same mean=0, std=1 range so each feature has equal
    opportunity to contribute to the isolation.
    """
    print("[1/4] Loading feature matrix ...")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(path)
    print(f"      Loaded {len(df):,} rows x {len(df.columns)} columns")

    # Separate identifier from features
    invoice_nos = df["invoice_no"]
    feature_cols = [c for c in df.columns if c != "invoice_no"]
    X = df[feature_cols].values   # numpy array for sklearn

    print(f"      Feature columns used: {len(feature_cols)}")

    # Scale features
    print("[2/4] Scaling features with StandardScaler ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"      Scaling done. Mean ~0, Std ~1 per feature.")

    return invoice_nos, feature_cols, X_scaled


# =============================================================================
def run_isolation_forest(X_scaled: np.ndarray):
    """
    Train and predict using Isolation Forest.

    Returns
    -------
    predictions   : array of -1 (anomaly) or +1 (normal)
    anomaly_flags : array of 1 (anomaly) or 0 (normal)
    anomaly_scores: array of positive floats; higher = more anomalous
    """
    print(f"[3/4] Training Isolation Forest ...")
    print(f"      n_estimators  = {IF_N_ESTIMATORS}")
    print(f"      contamination = {IF_CONTAMINATION}")
    print(f"      random_state  = {IF_RANDOM_STATE}")

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=IF_N_JOBS,
    )

    # fit_predict returns -1 for anomaly, +1 for normal
    predictions = model.fit_predict(X_scaled)

    # score_samples returns negative values; more negative = more anomalous
    # We negate so that higher score = more anomalous (easier to interpret)
    raw_scores    = model.score_samples(X_scaled)
    anomaly_scores = -raw_scores   # now: higher value = more anomalous

    # Binary flag: 1 = anomaly (prediction == -1), 0 = normal
    anomaly_flags = (predictions == -1).astype(int)

    n_anomalies = anomaly_flags.sum()
    print(f"      Detected anomalies : {n_anomalies:,} ({n_anomalies/len(predictions)*100:.2f}%)")
    print(f"      Score range        : {anomaly_scores.min():.4f} to {anomaly_scores.max():.4f}")

    return predictions, anomaly_flags, anomaly_scores


# =============================================================================
def save_results(invoice_nos, predictions, anomaly_flags, anomaly_scores):
    """Save results to CSV with one row per invoice."""

    results_df = pd.DataFrame({
        "invoice_no"       : invoice_nos.values,
        "if_prediction"    : predictions,       # -1=anomaly, +1=normal
        "if_anomaly"       : anomaly_flags,     # 1=anomaly, 0=normal
        "if_score"         : anomaly_scores,    # higher = more suspicious
    })

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"[4/4] Results saved to: {RESULTS_PATH}")

    return results_df


# =============================================================================
def print_summary(results_df: pd.DataFrame):
    """Print a summary report of the Isolation Forest results."""

    print("\n" + "="*60)
    print("  Isolation Forest -- Results Summary")
    print("="*60)

    total     = len(results_df)
    n_anom    = results_df["if_anomaly"].sum()
    n_normal  = total - n_anom

    print(f"  Total invoices   : {total:,}")
    print(f"  Flagged normal   : {n_normal:,} ({n_normal/total*100:.2f}%)")
    print(f"  Flagged anomalous: {n_anom:,}  ({n_anom/total*100:.2f}%)")

    print(f"\n  Anomaly score statistics:")
    print(f"    Min    : {results_df['if_score'].min():.4f}")
    print(f"    Max    : {results_df['if_score'].max():.4f}")
    print(f"    Mean   : {results_df['if_score'].mean():.4f}")
    print(f"    Median : {results_df['if_score'].median():.4f}")

    print(f"\n  Top 5 most suspicious invoices:")
    top5 = results_df.nlargest(5, "if_score")[["invoice_no", "if_score", "if_anomaly"]]
    print(top5.to_string(index=False))

    print(f"\n  Score distribution (quantiles):")
    for q in [0.50, 0.75, 0.90, 0.95, 0.99, 1.00]:
        print(f"    p{q*100:5.1f}: {results_df['if_score'].quantile(q):.4f}")


# =============================================================================
def quick_evaluation(results_df: pd.DataFrame):
    """
    Quick sanity check against ground truth (if available).
    Ground truth is ONLY used here for a sanity peek -- never used during
    model training above.
    """
    gt_path = os.path.join(ROOT_DIR, "data", "ground_truth.csv")
    if not os.path.exists(gt_path):
        print("\n  [INFO] ground_truth.csv not found -- skipping evaluation.")
        return

    gt = pd.read_csv(gt_path)
    merged = results_df.merge(gt, on="invoice_no")

    y_true = merged["actual_anomaly"].values
    y_pred = merged["if_anomaly"].values

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy  = (TP + TN) / len(y_true)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print("\n" + "="*60)
    print("  Quick Evaluation vs Ground Truth")
    print("  (Ground truth used ONLY here -- NOT during model training)")
    print("="*60)
    print(f"  Confusion Matrix:")
    print(f"    TP (correctly flagged anomalies): {TP}")
    print(f"    TN (correctly flagged normal)    : {TN}")
    print(f"    FP (false alarms)                : {FP}")
    print(f"    FN (missed anomalies)            : {FN}")
    print(f"\n  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    # Detection by anomaly type
    print(f"\n  Detection rate by anomaly type:")
    print(f"  {'Anomaly Type':<35} {'Total':>6} {'Detected':>9} {'Rate':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*9} {'-'*7}")
    for atype in sorted(merged[merged["actual_anomaly"]==1]["anomaly_type"].unique()):
        sub     = merged[merged["anomaly_type"] == atype]
        total   = len(sub)
        det     = sub["if_anomaly"].sum()
        rate    = det / total if total > 0 else 0
        print(f"  {atype:<35} {total:>6} {int(det):>9} {rate:>7.1%}")


# =============================================================================
def main():
    print("="*60)
    print("  GST Fraud Detection -- Stage 3: Isolation Forest")
    print("="*60)

    invoice_nos, feature_cols, X_scaled = load_features(FEATURES_PATH)
    predictions, anomaly_flags, anomaly_scores = run_isolation_forest(X_scaled)
    results_df = save_results(invoice_nos, predictions, anomaly_flags, anomaly_scores)
    print_summary(results_df)
    quick_evaluation(results_df)

    print("\n[DONE] Stage 3 complete.")


if __name__ == "__main__":
    main()
