"""
evaluate_models.py
==================
Stage 6: Comprehensive Model Evaluation and Comparison.

This module evaluates all three unsupervised anomaly detection models:
  1. Isolation Forest (Tree-based partitioning)
  2. Local Outlier Factor (Local density estimation)
  3. Statistical Z-Score (Parametric dispersion)

IMPORTANT DESIGN RULE:
  - Ground truth (actual_anomaly, anomaly_type) is strictly isolated and ONLY
    loaded in this evaluation stage.
  - Models were trained/fitted without any access to ground truth.

Outputs:
  - Formatted console comparison tables
  - results/model_comparison.csv
  - results/model_evaluation_metrics.json (consumed by Flask /api/model-performance)

Run:
    python models/evaluate_models.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR        = os.path.dirname(BASE_DIR)
GT_PATH         = os.path.join(ROOT_DIR, "data", "ground_truth.csv")
IF_PATH         = os.path.join(ROOT_DIR, "results", "isolation_forest_results.csv")
LOF_PATH        = os.path.join(ROOT_DIR, "results", "lof_results.csv")
Z_PATH          = os.path.join(ROOT_DIR, "results", "zscore_results.csv")
OUT_CSV_PATH    = os.path.join(ROOT_DIR, "results", "model_comparison.csv")
OUT_JSON_PATH   = os.path.join(ROOT_DIR, "results", "model_evaluation_metrics.json")


def load_datasets():
    """Load ground truth and model predictions."""
    print("[1/4] Loading model prediction files and ground truth ...")
    for p, name in [(GT_PATH, "ground_truth.csv"),
                    (IF_PATH, "isolation_forest_results.csv"),
                    (LOF_PATH, "lof_results.csv"),
                    (Z_PATH, "zscore_results.csv")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    df_gt = pd.read_csv(GT_PATH)
    df_if = pd.read_csv(IF_PATH)
    df_lof = pd.read_csv(LOF_PATH)
    df_z = pd.read_csv(Z_PATH)

    # Standardize column naming for evaluation
    if_col = "isolation_anomaly" if "isolation_anomaly" in df_if.columns else "if_anomaly"
    lof_col = "lof_anomaly"
    z_col = "zscore_anomaly"

    # Merge all on invoice_no
    merged = df_gt.merge(df_if[["invoice_no", if_col]], on="invoice_no")
    merged = merged.merge(df_lof[["invoice_no", lof_col]], on="invoice_no")
    merged = merged.merge(df_z[["invoice_no", z_col]], on="invoice_no")

    # Rename to clean keys
    merged.rename(columns={
        if_col: "pred_if",
        lof_col: "pred_lof",
        z_col: "pred_zscore"
    }, inplace=True)

    print(f"      Successfully merged {len(merged):,} records across all models.")
    return merged


def calculate_metrics(y_true, y_pred):
    """Compute standard classification metrics from boolean / binary arrays."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def evaluate_all_models(merged: pd.DataFrame):
    """Evaluate each individual model and multi-model consensus."""
    print("[2/4] Computing confusion matrices and classification metrics ...")

    y_true = merged["actual_anomaly"].values

    models = {
        "Isolation Forest": merged["pred_if"].values,
        "Local Outlier Factor (LOF)": merged["pred_lof"].values,
        "Statistical Z-Score": merged["pred_zscore"].values,
    }

    # Also compute consensus signals
    vote_sum = merged["pred_if"] + merged["pred_lof"] + merged["pred_zscore"]
    models["At Least 1 Model (Union)"] = (vote_sum >= 1).astype(int).values
    models["At Least 2 Models (Majority)"] = (vote_sum >= 2).astype(int).values
    models["All 3 Models (Intersection)"] = (vote_sum == 3).astype(int).values

    overall_metrics = {}
    for name, y_pred in models.items():
        overall_metrics[name] = calculate_metrics(y_true, y_pred)

    return overall_metrics, models


def evaluate_by_anomaly_type(merged: pd.DataFrame):
    """Compute detection rate for every anomaly type across models."""
    print("[3/4] Analyzing detection rates across all 11 anomaly categories ...")

    anom_rows = merged[merged["actual_anomaly"] == 1]
    anomaly_types = sorted(anom_rows["anomaly_type"].unique())

    type_results = []
    for atype in anomaly_types:
        sub = anom_rows[anom_rows["anomaly_type"] == atype]
        total = len(sub)

        det_if = int(sub["pred_if"].sum())
        det_lof = int(sub["pred_lof"].sum())
        det_z = int(sub["pred_zscore"].sum())

        union_det = int(((sub["pred_if"] == 1) | (sub["pred_lof"] == 1) | (sub["pred_zscore"] == 1)).sum())
        maj_det = int(((sub["pred_if"] + sub["pred_lof"] + sub["pred_zscore"]) >= 2).sum())

        type_results.append({
            "anomaly_type": atype,
            "total": total,
            "if_detected": det_if,
            "if_rate": round(det_if / total, 4) if total > 0 else 0.0,
            "lof_detected": det_lof,
            "lof_rate": round(det_lof / total, 4) if total > 0 else 0.0,
            "zscore_detected": det_z,
            "zscore_rate": round(det_z / total, 4) if total > 0 else 0.0,
            "union_detected": union_det,
            "union_rate": round(union_det / total, 4) if total > 0 else 0.0,
            "majority_detected": maj_det,
            "majority_rate": round(maj_det / total, 4) if total > 0 else 0.0,
        })

    return pd.DataFrame(type_results)


def print_reports(overall_metrics, df_type_eval):
    """Print clean, publication-ready ASCII comparison tables."""
    print("\n" + "=" * 80)
    print("  GST INVOICE ANOMALY DETECTION: MODEL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<30} | {'Accuracy':>9} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'TP':>5} | {'FP':>5}")
    print("-" * 88)
    for model_name, m in overall_metrics.items():
        print(f"{model_name:<30} | {m['accuracy']*100:>8.2f}% | {m['precision']:>9.4f} | {m['recall']:>9.4f} | {m['f1']:>9.4f} | {m['tp']:>5} | {m['fp']:>5}")

    print("\n" + "=" * 80)
    print("  DETECTION RATE BY ANOMALY TYPE (PERCENTAGE CAPTURED)")
    print("=" * 80)
    print(f"{'Anomaly Category':<27} | {'Total':>5} | {'IF':>7} | {'LOF':>7} | {'Z-Score':>7} | {'Combined (>=1)':>14}")
    print("-" * 80)
    for _, row in df_type_eval.iterrows():
        print(f"{row['anomaly_type']:<27} | {row['total']:>5} | {row['if_rate']*100:>6.1f}% | {row['lof_rate']*100:>6.1f}% | {row['zscore_rate']*100:>6.1f}% | {row['union_rate']*100:>13.1f}%")

    # Overall detection summary across all 1,000 anomalies
    tot_anomalies = df_type_eval["total"].sum()
    if_tot = df_type_eval["if_detected"].sum()
    lof_tot = df_type_eval["lof_detected"].sum()
    z_tot = df_type_eval["zscore_detected"].sum()
    u_tot = df_type_eval["union_detected"].sum()
    print("-" * 80)
    print(f"{'TOTAL OVERALL CAPTURE':<27} | {tot_anomalies:>5} | {if_tot/tot_anomalies*100:>6.1f}% | {lof_tot/tot_anomalies*100:>6.1f}% | {z_tot/tot_anomalies*100:>6.1f}% | {u_tot/tot_anomalies*100:>13.1f}%")


def save_evaluation_artifacts(overall_metrics, df_type_eval):
    """Save metrics to JSON and CSV for integration with Flask API and Dashboard."""
    print("\n[4/4] Saving evaluation artifacts ...")
    os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)

    # Save type comparison CSV
    df_type_eval.to_csv(OUT_CSV_PATH, index=False)
    print(f"      Saved: {OUT_CSV_PATH}")

    # Build comprehensive JSON artifact for Flask API /api/model-performance
    export_payload = {
        "model_metrics": overall_metrics,
        "anomaly_type_breakdown": df_type_eval.to_dict(orient="records"),
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_invoices": 20000,
        "total_anomalies": 1000,
        "total_normal": 19000
    }

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"      Saved: {OUT_JSON_PATH}")


def main():
    print("=" * 80)
    print("  STAGE 6: MODEL EVALUATION & MULTI-MODEL ANALYSIS")
    print("=" * 80)

    merged = load_datasets()
    overall_metrics, _ = evaluate_all_models(merged)
    df_type_eval = evaluate_by_anomaly_type(merged)
    print_reports(overall_metrics, df_type_eval)
    save_evaluation_artifacts(overall_metrics, df_type_eval)

    print("\n[DONE] Stage 6 complete.")


if __name__ == "__main__":
    main()
