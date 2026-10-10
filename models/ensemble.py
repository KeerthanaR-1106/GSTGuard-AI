"""
ensemble.py
===========
Stage 7: Ensemble Risk Scoring and Risk Classification.

This module unifies the outputs of all three unsupervised models:
  1. Isolation Forest (Tree-based partitioning)  - Default Weight: 40%
  2. Local Outlier Factor (Density estimation)    - Default Weight: 40%
  3. Statistical Z-Score (Parametric dispersion)   - Default Weight: 20%

Transparent Weighted Approach:
  Each model's raw score is normalized to a 0–100 scale:
    - norm_if: Scaled based on tree isolation path score
    - norm_lof: Scaled based on local outlier reachability density ratio
    - norm_z: Scaled based on maximum absolute standard score deviation

  Weighted Risk Score (0 to 100):
    Risk Score = (W_IF * norm_if) + (W_LOF * norm_lof) + (W_Z * norm_z)
    + Multi-model consensus calibration

Risk Level Classification:
  - Low Risk    : 0 – 39   (Standard transaction, no anomalous flags)
  - Medium Risk : 40 – 69  (Single model detection or moderate statistical divergence)
  - High Risk   : 70 – 100 (Multi-model consensus or extreme feature deviation)

Explainability:
  Synthesizes model consensus and dominant feature deviations to generate
  human-readable explanations for compliance officers (e.g. "High risk because
  Isolation Forest and Z-Score flagged this invoice due to extreme unit price").
  Does NOT claim legal fraud; uses audit terminology ("suspicious", "requires review").

Reads   : results/isolation_forest_results.csv
          results/lof_results.csv
          results/zscore_results.csv
          data/ml_features.csv
Outputs : results/ensemble_results.csv

Run:
    python models/ensemble.py
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(BASE_DIR)
IF_PATH       = os.path.join(ROOT_DIR, "results", "isolation_forest_results.csv")
LOF_PATH      = os.path.join(ROOT_DIR, "results", "lof_results.csv")
Z_PATH        = os.path.join(ROOT_DIR, "results", "zscore_results.csv")
FEAT_PATH     = os.path.join(ROOT_DIR, "data", "ml_features.csv")
RESULTS_PATH  = os.path.join(ROOT_DIR, "results", "ensemble_results.csv")

# ── Configurable Weights & Thresholds ─────────────────────────────────────────
# Documented project design choice: Can be adjusted based on MSME compliance priorities
WEIGHT_ISOLATION_FOREST = 0.40  # 40% weight to tree partitioning
WEIGHT_LOF              = 0.40  # 40% weight to neighborhood density
WEIGHT_ZSCORE           = 0.20  # 20% weight to parametric dispersion

# Verify weights sum to 1.0
assert abs((WEIGHT_ISOLATION_FOREST + WEIGHT_LOF + WEIGHT_ZSCORE) - 1.0) < 1e-6, "Weights must sum to 1.0"

# Risk classification boundaries
THRESHOLD_LOW_MAX    = 39.99
THRESHOLD_MEDIUM_MAX = 69.99
# High Risk is >= 70.0


# =============================================================================
def load_all_results():
    """Load results from all 3 models and raw features for explanation generation."""
    print("[1/5] Loading model outputs and features ...")
    for p in [IF_PATH, LOF_PATH, Z_PATH, FEAT_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    df_if = pd.read_csv(IF_PATH)
    df_lof = pd.read_csv(LOF_PATH)
    df_z = pd.read_csv(Z_PATH)
    df_feat = pd.read_csv(FEAT_PATH)

    # Standardize column naming
    if_score_col = "isolation_score" if "isolation_score" in df_if.columns else "if_score"
    if_anom_col  = "isolation_anomaly" if "isolation_anomaly" in df_if.columns else "if_anomaly"

    # Merge into a single dataframe
    df = df_if[["invoice_no", if_score_col, if_anom_col]].rename(columns={
        if_score_col: "isolation_score",
        if_anom_col:  "isolation_anomaly"
    })
    df = df.merge(df_lof[["invoice_no", "lof_score", "lof_anomaly"]], on="invoice_no")
    df = df.merge(df_z[["invoice_no", "max_abs_zscore", "zscore_anomaly", "dominant_feature"]], on="invoice_no")

    # Merge key feature signals for human-readable audit explanations
    key_signals = [
        "invoice_no", "quantity", "unit_price", "discount_percentage",
        "taxable_value", "total_tax", "total_amount", "gst_difference",
        "total_amount_difference", "cgst_sgst_split_difference",
        "igst_difference", "future_date_flag", "duplicate_transaction_flag",
        "supplier_daily_frequency", "buyer_daily_frequency"
    ]
    df = df.merge(df_feat[key_signals], on="invoice_no")

    print(f"      Loaded and aligned {len(df):,} invoices across all 3 models.")
    return df


# =============================================================================
def normalize_scores(df: pd.DataFrame):
    """
    Normalize individual raw anomaly scores onto a consistent 0 to 100 scale:
      - Isolation Forest: min-max normalization over empirical score distribution
      - LOF: Inliers cluster around 1.0 (mapped to ~0-10); outliers >= 2.0 scaled to 70-100
      - Z-Score: Inliers are |Z| < 3.0 (mapped to 0-35); |Z| >= 3.0 scaled into 50-100
    """
    print("[2/5] Normalizing model scores to 0-100 scale ...")

    # 1. Isolation Forest normalization
    # Scores range ~0.33 to 0.79. Min-max maps lowest (normal) to 0, highest (extreme anomaly) to 100.
    if_min = df["isolation_score"].min()
    if_max = df["isolation_score"].max()
    norm_if = ((df["isolation_score"] - if_min) / (if_max - if_min)) * 100.0

    # 2. LOF normalization
    # Normal points have LOF ≈ 1.0 (range 0.94 - 1.20).
    # Outliers start at LOF > 1.2 and can reach 27.0.
    # We apply robust piecewise scaling:
    #   LOF <= 1.0 -> 0 to 10
    #   LOF between 1.0 and 2.5 -> scales linearly from 10 to 80
    #   LOF > 2.5 -> scales from 80 to 100 (capped at 100)
    lof = df["lof_score"].values
    norm_lof = np.where(
        lof <= 1.0,
        np.clip((lof - 0.90) / 0.10 * 10.0, 0.0, 10.0),
        np.where(
            lof <= 2.5,
            10.0 + ((lof - 1.0) / 1.5) * 70.0,
            80.0 + np.clip((lof - 2.5) / 5.0 * 20.0, 0.0, 20.0)
        )
    )

    # 3. Z-Score normalization
    # |Z| < 3.0 is statistically normal (mapped to 0 - 35).
    # |Z| = 3.0 is threshold (mapped to 50).
    # |Z| >= 6.0 is extreme tail (mapped to 80 - 100).
    z = df["max_abs_zscore"].values
    norm_z = np.where(
        z < 3.0,
        (z / 3.0) * 35.0,
        np.clip(50.0 + ((z - 3.0) / 7.0) * 50.0, 50.0, 100.0)
    )

    df["norm_if"]    = np.round(norm_if, 2)
    df["norm_lof"]   = np.round(norm_lof, 2)
    df["norm_z"]     = np.round(norm_z, 2)

    print("      Score normalization completed.")
    return df


# =============================================================================
def compute_ensemble_risk(df: pd.DataFrame):
    """
    Calculate the combined risk score (0 to 100) using transparent weights
    and multi-model consensus calibration.
    """
    print("[3/5] Calculating weighted ensemble risk score (40% IF, 40% LOF, 20% Z-Score) ...")

    # Base weighted sum of normalized scores
    base_score = (
        WEIGHT_ISOLATION_FOREST * df["norm_if"] +
        WEIGHT_LOF              * df["norm_lof"] +
        WEIGHT_ZSCORE           * df["norm_z"]
    )

    # Consensus count: How many of the 3 models flagged an anomaly?
    consensus_count = (
        df["isolation_anomaly"].astype(int) +
        df["lof_anomaly"].astype(int) +
        df["zscore_anomaly"].astype(int)
    )

    # Consensus adjustment:
    # - If all 3 models flag an anomaly: strong consensus, guaranteed High Risk (boosted to >= 75)
    # - If 2 models flag an anomaly: moderate consensus, boosted to at least 55-70
    # - If 0 models flag an anomaly: capped below 35 to prevent false alarms
    calibrated_score = base_score.copy()
    calibrated_score = np.where(consensus_count == 3, np.maximum(calibrated_score, 75.0), calibrated_score)
    calibrated_score = np.where(consensus_count == 2, np.maximum(calibrated_score, 55.0), calibrated_score)
    calibrated_score = np.where(consensus_count == 0, np.minimum(calibrated_score, 35.0), calibrated_score)

    # Ensure strictly bounded within [0.0, 100.0]
    final_score = np.clip(calibrated_score, 0.0, 100.0).round(2)
    df["risk_score"] = final_score

    # Assign Risk Level: Low (0-39), Medium (40-69), High (70-100)
    conditions = [
        (df["risk_score"] <= THRESHOLD_LOW_MAX),
        (df["risk_score"] >= 40.0) & (df["risk_score"] <= THRESHOLD_MEDIUM_MAX),
        (df["risk_score"] >= 70.0)
    ]
    choices = ["Low Risk", "Medium Risk", "High Risk"]
    df["risk_level"] = np.select(conditions, choices, default="Low Risk")

    print(f"      Ensemble scoring complete.")
    return df


# =============================================================================
def generate_explanations(df: pd.DataFrame):
    """
    Generate professional, non-accusatory compliance audit explanations.
    Synthesizes model consensus and specific detected discrepancies.
    """
    print("[4/5] Generating compliance explanations and audit classifications ...")

    explanations = []
    detected_types = []

    for _, row in df.iterrows():
        level = row["risk_level"]
        score = row["risk_score"]

        # Models that flagged this invoice
        flagged_models = []
        if row["isolation_anomaly"] == 1:
            flagged_models.append("Isolation Forest")
        if row["lof_anomaly"] == 1:
            flagged_models.append("Local Outlier Factor")
        if row["zscore_anomaly"] == 1:
            flagged_models.append("Z-Score")

        # Specific feature discrepancies detected
        reasons = []
        dtype = "Normal"

        if row["duplicate_transaction_flag"] == 1:
            reasons.append("duplicate transaction signature (identical line-items with distinct invoice numbers)")
            dtype = "Duplicate Transaction"
        elif row["future_date_flag"] == 1:
            reasons.append("post-dated invoice outside the authorized compliance period")
            dtype = "Future Invoice Date"
        elif row["total_amount_difference"] > 100.0:
            reasons.append(f"major total invoice amount mismatch (discrepancy of Rs.{row['total_amount_difference']:,.2f})")
            dtype = "Extreme Invoice Amount"
        elif row["unit_price"] > 100_000.0:
            reasons.append(f"unit price (Rs.{row['unit_price']:,.2f}) exceeds normal category thresholds by several orders of magnitude")
            dtype = "Extreme Unit Price"
        elif row["quantity"] > 1_000.0:
            reasons.append(f"unusual quantity ({row['quantity']:,.1f} units) substantially deviating from historical distribution")
            dtype = "Unusual Quantity"
        elif row["discount_percentage"] > 50.0:
            reasons.append(f"abnormally high trade discount of {row['discount_percentage']:.1f}%")
            dtype = "Unusual Discount"
        elif row["gst_difference"] > 50.0 or abs(row["total_tax"] - (row["taxable_value"] * 0.18)) > 1000.0 and row["gst_difference"] > 1.0:
            reasons.append(f"tax calculation inconsistency (calculated GST deviates by Rs.{row['gst_difference']:,.2f})")
            dtype = "Tax Calculation Mismatch"
        elif row["cgst_sgst_split_difference"] > 50.0:
            reasons.append(f"asymmetrical CGST/SGST tax split (deviation of Rs.{row['cgst_sgst_split_difference']:,.2f})")
            dtype = "Incorrect CGST SGST Split"
        elif row["igst_difference"] > 50.0:
            reasons.append(f"inter-state IGST calculation mismatch (deviation of Rs.{row['igst_difference']:,.2f})")
            dtype = "Incorrect IGST"
        elif row["supplier_daily_frequency"] > 20:
            reasons.append(f"unusual burst activity: supplier issued {int(row['supplier_daily_frequency'])} invoices on a single date")
            dtype = "Unusual Supplier Activity"
        elif row["buyer_daily_frequency"] > 20:
            reasons.append(f"unusual burst activity: buyer recorded {int(row['buyer_daily_frequency'])} transactions on a single date")
            dtype = "Unusual Buyer Activity"

        if level != "Low Risk" and dtype == "Normal":
            dtype = "Statistical Model Outlier"

        # Build clean audit narrative
        if level == "Low Risk":
            if len(reasons) == 0:
                exp = "Transaction parameters and tax calculations align with expected MSME compliance benchmarks. No significant anomalies detected."
            else:
                exp = f"Minor statistical variance noted ({reasons[0]}), but overall risk score ({score:.1f}/100) remains within acceptable compliance tolerance."
        elif level == "Medium Risk":
            model_str = ", ".join(flagged_models) if flagged_models else "Statistical analysis"
            reason_str = "; ".join(reasons) if reasons else f"elevated deviation in {row['dominant_feature'].replace('_', ' ')}"
            exp = f"Potentially anomalous transaction flagged by {model_str}. Review recommended due to {reason_str}."
        else: # High Risk
            model_str = ", ".join(flagged_models) if flagged_models else "Multi-model consensus"
            reason_str = "; ".join(reasons) if reasons else f"extreme statistical deviation in {row['dominant_feature'].replace('_', ' ')}"
            exp = f"High-risk anomaly identified by {model_str} (Risk Score: {score:.1f}/100). Requires priority compliance review due to {reason_str}."

        explanations.append(exp)
        detected_types.append(dtype)

    df["anomaly_type"] = detected_types
    df["explanation"] = explanations

    return df


# =============================================================================
def save_ensemble_results(df: pd.DataFrame):
    """Save final aligned output matching the SQLite database schema exactly."""
    print("[5/5] Saving results to results/ensemble_results.csv ...")

    OUTPUT_COLUMNS = [
        "invoice_no",
        "isolation_score",
        "isolation_anomaly",
        "lof_score",
        "lof_anomaly",
        "max_abs_zscore",
        "zscore_anomaly",
        "risk_score",
        "risk_level",
        "anomaly_type",
        "explanation"
    ]

    out_df = df[OUTPUT_COLUMNS].copy()
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    out_df.to_csv(RESULTS_PATH, index=False)
    print(f"      Saved: {RESULTS_PATH}")
    return out_df


# =============================================================================
def print_summary(out_df: pd.DataFrame):
    """Print executive distribution summary."""
    print("\n" + "=" * 70)
    print("  STAGE 7: ENSEMBLE RISK SCORE & CLASSIFICATION SUMMARY")
    print("=" * 70)

    total = len(out_df)
    counts = out_df["risk_level"].value_counts()

    print(f"  Total Invoices Evaluated : {total:,}")
    print("\n  Risk Level Distribution:")
    for level in ["Low Risk", "Medium Risk", "High Risk"]:
        cnt = counts.get(level, 0)
        pct = cnt / total * 100
        print(f"    {level:<15} : {cnt:>6,} ({pct:>5.2f}%)")

    print(f"\n  Risk Score Statistics (0 - 100):")
    print(f"    Minimum : {out_df['risk_score'].min():.2f}")
    print(f"    Maximum : {out_df['risk_score'].max():.2f}")
    print(f"    Mean    : {out_df['risk_score'].mean():.2f}")
    print(f"    Median  : {out_df['risk_score'].median():.2f}")

    print("\n  Sample High-Risk Explanations:")
    sample_high = out_df[out_df["risk_level"] == "High Risk"].head(3)
    for _, r in sample_high.iterrows():
        print(f"    [{r['invoice_no']}] Score: {r['risk_score']:.1f} | Type: {r['anomaly_type']}")
        print(f"      {r['explanation']}")


# =============================================================================
def main():
    print("=" * 70)
    print("  STAGE 7: ENSEMBLE MODEL & COMPLIANCE RISK SCORING")
    print("=" * 70)

    df = load_all_results()
    df = normalize_scores(df)
    df = compute_ensemble_risk(df)
    df = generate_explanations(df)
    out_df = save_ensemble_results(df)
    print_summary(out_df)

    print("\n[DONE] Stage 7 complete.")


if __name__ == "__main__":
    main()
