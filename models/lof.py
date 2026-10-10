"""
lof.py
======
Stage 4: Unsupervised anomaly detection using Local Outlier Factor (LOF).

What is Local Outlier Factor (LOF)?
-----------------------------------
Local Outlier Factor (Breunig et al., 2000) is a density-based unsupervised
outlier detection algorithm. While Isolation Forest isolates points based on
global tree partitioning, LOF compares the local density of a data point to
the local densities of its k-nearest neighbors (k = n_neighbors).

Mathematical Concept:
  1. k-distance of p: The distance between point p and its k-th nearest neighbor.
  2. Reachability Distance: reach_dist_k(p, o) = max(k_dist(o), dist(p, o))
  3. Local Reachability Density (lrd): Inverse of average reachability distance
     of point p from its k neighbors.
  4. LOF Score: Average ratio of the lrd of neighbors to the lrd of point p.
     - LOF ≈ 1.0  : Point has density similar to its neighbors (Normal/Inlier).
     - LOF < 1.0  : Point is in a denser region than its neighbors (Inlier).
     - LOF >> 1.0 : Point has substantially lower density than its neighbors (Outlier).

Why LOF Complements Isolation Forest:
-------------------------------------
1. Global vs. Local Perspective:
   - Isolation Forest is a global partitioning algorithm. It separates points
     far away from the center of mass or along extreme projections.
   - LOF is a local density algorithm. A point might not have extreme absolute values,
     but if it falls into a sparse pocket between dense normal clusters (e.g. unusual
     combinations of GST rate and product category or subtle tax split deviations),
     LOF flags it because its local neighborhood is sparse.
2. Sensitivity to Micro-Clusters:
   - LOF can identify outliers with respect to local clusters with varying densities.
3. Multi-Model Defense in Compliance:
   - Combining a tree-based partitioning model (Isolation Forest) with a neighbor-based
     density model (LOF) creates robust ensemble coverage against both global magnitude
     fraud (huge amounts) and localized structural inconsistencies.

Configuration:
  - n_neighbors   = 20
  - contamination = 0.05
  - n_jobs        = -1

Reads   : data/ml_features.csv
Outputs : results/lof_results.csv

Run:
    python models/lof.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(BASE_DIR)
FEATURES_PATH = os.path.join(ROOT_DIR, "data", "ml_features.csv")
RESULTS_PATH  = os.path.join(ROOT_DIR, "results", "lof_results.csv")

# ── Hyperparameters ───────────────────────────────────────────────────────────
LOF_N_NEIGHBORS   = 20    # Number of neighbors to consider for local density
LOF_CONTAMINATION = 0.05  # Expected anomaly rate (~5%)
LOF_N_JOBS        = -1    # Parallel processing using all CPU cores


# =============================================================================
def load_and_scale_features(path: str):
    """
    Load ml_features.csv and scale numeric features using StandardScaler.
    Feature scaling is crucial for distance-based algorithms like LOF because
    unscaled features with large ranges (e.g., total_amount in millions) would
    completely dominate the Euclidean distance calculation over smaller
    percentages (e.g., gst_rate or discount_percentage).
    """
    print("[1/4] Loading feature matrix for LOF ...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(path)
    print(f"      Loaded {len(df):,} records with {len(df.columns)} total columns")

    invoice_nos = df["invoice_no"]
    feature_cols = [c for c in df.columns if c != "invoice_no"]
    X = df[feature_cols].values

    print(f"      Features to scale: {len(feature_cols)}")
    print("[2/4] Applying StandardScaler (critical for Euclidean distance calculation) ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("      Features standardized (mean=0, std=1).")

    return invoice_nos, feature_cols, X_scaled


# =============================================================================
def run_lof(X_scaled: np.ndarray):
    """
    Fit Local Outlier Factor and compute predictions and scores.

    scikit-learn note:
      - fit_predict() returns -1 for outlier, +1 for inlier.
      - negative_outlier_factor_ contains the opposite LOF values (<= -1).
        We calculate: lof_score = -negative_outlier_factor_
        so that lof_score >= 1.0, where ~1.0 is normal and >1.5 is anomalous.
    """
    print(f"[3/4] Fitting Local Outlier Factor (LOF) ...")
    print(f"      n_neighbors   = {LOF_N_NEIGHBORS}")
    print(f"      contamination = {LOF_CONTAMINATION}")
    print(f"      metric        = minkowski (p=2, Euclidean)")

    model = LocalOutlierFactor(
        n_neighbors=LOF_N_NEIGHBORS,
        contamination=LOF_CONTAMINATION,
        n_jobs=LOF_N_JOBS
    )

    # predictions: -1 (anomaly), 1 (normal)
    predictions = model.fit_predict(X_scaled)

    # Convert negative_outlier_factor_ to standard LOF score (higher = more anomalous)
    # negative_outlier_factor_ is negative (e.g. -1.05, -3.2)
    # Opposite is the true LOF: ~1.0 = normal density, >1.5 = low density / outlier
    raw_scores = -model.negative_outlier_factor_

    anomaly_flags = (predictions == -1).astype(int)

    n_anomalies = anomaly_flags.sum()
    print(f"      LOF Anomaly detection completed.")
    print(f"      Flagged anomalous : {n_anomalies:,} ({n_anomalies/len(predictions)*100:.2f}%)")
    print(f"      LOF score range   : min={raw_scores.min():.4f}, max={raw_scores.max():.4f}, mean={raw_scores.mean():.4f}")

    return predictions, anomaly_flags, raw_scores


# =============================================================================
def save_results(invoice_nos, predictions, anomaly_flags, lof_scores):
    """Save LOF results to results/lof_results.csv."""
    results_df = pd.DataFrame({
        "invoice_no"    : invoice_nos.values,
        "lof_prediction": predictions,      # -1=anomaly, +1=normal
        "lof_anomaly"   : anomaly_flags,    # 1=anomaly, 0=normal
        "lof_score"     : lof_scores,       # Standard LOF score (>= 1.0, higher = more outlier)
    })

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"[4/4] Saved LOF results to: {RESULTS_PATH}")
    return results_df


# =============================================================================
def evaluate_lof(results_df: pd.DataFrame):
    """
    Perform a post-run evaluation of LOF against ground truth.
    Ground truth is strictly isolated for evaluation metrics.
    """
    gt_path = os.path.join(ROOT_DIR, "data", "ground_truth.csv")
    if not os.path.exists(gt_path):
        print("  [WARN] ground_truth.csv not found; skipping evaluation.")
        return

    gt = pd.read_csv(gt_path)
    merged = results_df.merge(gt, on="invoice_no")

    y_true = merged["actual_anomaly"].values
    y_pred = merged["lof_anomaly"].values

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())

    acc  = (TP + TN) / len(y_true)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    print("\n" + "="*62)
    print("  Local Outlier Factor (LOF) -- Performance Evaluation")
    print("="*62)
    print(f"  Confusion Matrix:")
    print(f"    True Positives (TP)  : {TP}")
    print(f"    True Negatives (TN)  : {TN}")
    print(f"    False Positives (FP) : {FP}")
    print(f"    False Negatives (FN) : {FN}")
    print(f"\n  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

    print("\n  Detection rate by anomaly type:")
    print(f"  {'Anomaly Type':<35} {'Total':>6} {'Detected':>9} {'Rate':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*9} {'-'*7}")

    anom_rows = merged[merged["actual_anomaly"] == 1]
    for atype in sorted(anom_rows["anomaly_type"].unique()):
        sub   = anom_rows[anom_rows["anomaly_type"] == atype]
        tot   = len(sub)
        det   = sub["lof_anomaly"].sum()
        r     = det / tot if tot > 0 else 0.0
        print(f"  {atype:<35} {tot:>6} {int(det):>9} {r:>7.1%}")


# =============================================================================
def main():
    print("="*62)
    print("  GST Fraud Detection -- Stage 4: Local Outlier Factor (LOF)")
    print("="*62)

    invoice_nos, feature_cols, X_scaled = load_and_scale_features(FEATURES_PATH)
    predictions, anomaly_flags, lof_scores = run_lof(X_scaled)
    results_df = save_results(invoice_nos, predictions, anomaly_flags, lof_scores)
    evaluate_lof(results_df)
    print("\n[DONE] Stage 4 complete.")


if __name__ == "__main__":
    main()
