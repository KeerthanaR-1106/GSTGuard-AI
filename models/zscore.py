"""
zscore.py
=========
Stage 5: Statistical Z-Score Anomaly Detection.

What is Z-Score Anomaly Detection?
----------------------------------
The standard score (Z-score) measures how many standard deviations an observation
x is above or below the mean (μ) of a feature:

    Z = (x - μ) / σ

Where:
  x = observed feature value
  μ = feature mean
  σ = feature standard deviation

Under Chebyshev's inequality and the empirical rule for normal distributions:
  - ~68.2% of data points fall within |Z| < 1
  - ~95.4% of data points fall within |Z| < 2
  - ~99.7% of data points fall within |Z| < 3

Observations with |Z| >= 3.0 are extreme statistical outliers lying in the top 0.3%
tails of the univariate distribution.

Why Z-Score Complements Isolation Forest & LOF:
------------------------------------------------
1. Parametric & Univariate Interpretability:
   - Isolation Forest builds multi-dimensional random split hyperplanes.
   - LOF computes multi-dimensional density ratios over k-nearest neighbors.
   - Z-score inspects every individual feature dimension directly. If a transaction
     has a normal LOF score but contains an extreme 15-sigma spike in `total_amount_difference`,
     Z-score catches it directly and tells the auditor exactly which feature blew up.
2. Fast & Deterministic:
   - Z-scores are computed via closed-form matrix vectorization without random sampling
     or neighbor searches.
3. Third Pillar of Defense in the Ensemble:
   - Ensemble combines:
     * Tree-based partitioning (Isolation Forest)
     * Neighborhood-density estimation (LOF)
     * Parametric univariate dispersion (Z-score)

Configuration:
  - Z_THRESHOLD = 3.0 (configurable)

Reads   : data/ml_features.csv
Outputs : results/zscore_results.csv

Run:
    python models/zscore.py
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(BASE_DIR)
FEATURES_PATH = os.path.join(ROOT_DIR, "data", "ml_features.csv")
RESULTS_PATH  = os.path.join(ROOT_DIR, "results", "zscore_results.csv")

# ── Configurable Parameters ──────────────────────────────────────────────────
# Standard empirical threshold for statistical outliers (|Z| >= 3.0 is in the extreme tail)
Z_THRESHOLD = 3.0


# =============================================================================
def load_data(path: str):
    """Load ml_features.csv and separate invoice_no from ML features."""
    print("[1/4] Loading feature matrix for Z-score analysis ...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(path)
    invoice_nos = df["invoice_no"]
    feature_cols = [c for c in df.columns if c != "invoice_no"]
    X = df[feature_cols].copy()

    print(f"      Loaded {len(df):,} records with {len(feature_cols)} features.")
    return invoice_nos, feature_cols, X


# =============================================================================
def calculate_zscores(X: pd.DataFrame, threshold: float = Z_THRESHOLD):
    """
    Compute Z-score matrix:
      Z_ij = (X_ij - mean_j) / std_j

    Returns
    -------
    max_abs_zscore       : Maximum absolute Z-score across all 43 features for each row
    extreme_feature_count: Count of features where |Z| >= threshold for each row
    anomaly_flags        : 1 if max_abs_zscore >= threshold, else 0
    top_feature_names    : Name of the feature contributing the maximum Z-score for explanation
    """
    print(f"[2/4] Computing statistical Z-scores (Threshold: |Z| >= {threshold}) ...")

    # Compute mean and standard deviation per feature
    means = X.mean()
    stds  = X.std().replace(0, np.nan)  # avoid division by zero for constant features

    # Z-score matrix
    Z_df = (X - means) / stds
    Z_abs = Z_df.abs()

    # Maximum absolute Z-score for each invoice
    max_abs_zscore = Z_abs.max(axis=1)

    # Number of features with |Z| >= threshold
    extreme_feature_count = (Z_abs >= threshold).sum(axis=1)

    # Anomaly flag: 1 if at least one feature exceeds threshold
    anomaly_flags = (max_abs_zscore >= threshold).astype(int)

    # Feature that caused the maximum absolute Z-score (useful for explanation)
    top_feature_names = Z_abs.idxmax(axis=1)

    n_anomalies = anomaly_flags.sum()
    print(f"      Z-score calculation complete.")
    print(f"      Flagged anomalous : {n_anomalies:,} ({n_anomalies/len(X)*100:.2f}%)")
    print(f"      Max |Z| range     : min={max_abs_zscore.min():.4f}, max={max_abs_zscore.max():.4f}, median={max_abs_zscore.median():.4f}")

    return max_abs_zscore, extreme_feature_count, anomaly_flags, top_feature_names


# =============================================================================
def save_results(invoice_nos, max_abs_zscore, extreme_feature_count, anomaly_flags, top_feature_names):
    """Save Z-score results to results/zscore_results.csv."""
    results_df = pd.DataFrame({
        "invoice_no"           : invoice_nos.values,
        "max_abs_zscore"       : max_abs_zscore.round(4).values,
        "extreme_feature_count": extreme_feature_count.values,
        "zscore_anomaly"       : anomaly_flags.values,
        "dominant_feature"     : top_feature_names.values,
    })

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"[3/4] Saved Z-score results to: {RESULTS_PATH}")
    return results_df


# =============================================================================
def evaluate_zscore(results_df: pd.DataFrame):
    """
    Evaluate Z-score detection capability against ground truth.
    Ground truth is strictly isolated for evaluation metrics.
    """
    gt_path = os.path.join(ROOT_DIR, "data", "ground_truth.csv")
    if not os.path.exists(gt_path):
        print("  [WARN] ground_truth.csv not found; skipping evaluation.")
        return

    gt = pd.read_csv(gt_path)
    merged = results_df.merge(gt, on="invoice_no")

    y_true = merged["actual_anomaly"].values
    y_pred = merged["zscore_anomaly"].values

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())

    acc  = (TP + TN) / len(y_true)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    print("\n" + "="*62)
    print("  Statistical Z-Score -- Performance Evaluation")
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
        det   = sub["zscore_anomaly"].sum()
        r     = det / tot if tot > 0 else 0.0
        print(f"  {atype:<35} {tot:>6} {int(det):>9} {r:>7.1%}")


# =============================================================================
def main():
    print("="*62)
    print("  GST Fraud Detection -- Stage 5: Statistical Z-Score")
    print("="*62)

    invoice_nos, feature_cols, X = load_data(FEATURES_PATH)
    max_abs_zscore, extreme_count, anomaly_flags, dominant_feats = calculate_zscores(X, Z_THRESHOLD)
    results_df = save_results(invoice_nos, max_abs_zscore, extreme_count, anomaly_flags, dominant_feats)
    evaluate_zscore(results_df)

    print("\n[DONE] Stage 5 complete.")


if __name__ == "__main__":
    main()
