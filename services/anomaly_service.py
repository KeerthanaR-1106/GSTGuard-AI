"""
services/anomaly_service.py
===========================
Centralized Anomaly Detection Service.
Refactors and encapsulates the multi-model inference pipeline:
  - Feature extraction (43 features)
  - Isolation Forest inference
  - Local Outlier Factor inference
  - Statistical Z-Score inference
  - Multi-model consensus calibration & risk scoring
  - Audit explanation synthesis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from config import Config

from preprocessing.preprocess import (
    compute_consistency_features,
    compute_date_features,
    compute_frequency_features,
    compute_duplicate_features,
    build_feature_matrix
)


def calculate_weighted_anomaly_index(if_scores, lof_scores, max_z):
    """Combine batch-scaled detector outputs into a transparent 0–100 index."""
    def batch_scale(scores):
        scores = np.asarray(scores, dtype=float)
        minimum = np.min(scores)
        maximum = np.max(scores)
        if maximum <= minimum:
            return np.zeros_like(scores, dtype=float)
        return np.clip((scores - minimum) / (maximum - minimum) * 100.0, 0.0, 100.0)

    weight_total = (
        Config.WEIGHT_ISOLATION_FOREST + Config.WEIGHT_LOF + Config.WEIGHT_ZSCORE
    )
    isolation_contribution = (
        batch_scale(if_scores) * Config.WEIGHT_ISOLATION_FOREST / weight_total
    )
    lof_contribution = batch_scale(lof_scores) * Config.WEIGHT_LOF / weight_total
    zscore_contribution = (
        np.clip(np.asarray(max_z, dtype=float) / 8.0 * 100.0, 0.0, 100.0)
        * Config.WEIGHT_ZSCORE / weight_total
    )
    risk_score = np.clip(
        isolation_contribution + lof_contribution + zscore_contribution,
        0.0,
        100.0,
    ).round(2)
    return isolation_contribution, lof_contribution, zscore_contribution, risk_score


def classify_risk_level(score: float) -> str:
    """Map the displayed screening index to its existing category thresholds."""
    if score < 40.0:
        return "Low Risk"
    if score < 70.0:
        return "Medium Risk"
    return "High Risk"


def run_anomaly_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Execute end-to-end unsupervised ML anomaly detection on raw invoice DataFrame.
    Returns DataFrame matching analysis table schema.
    """
    df = df_raw.copy()
    if len(df) < 2:
        raise ValueError("At least two invoices are required for anomaly analysis.")

    # Numeric validation & date parsing
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    if df["invoice_date"].isnull().any():
        median_date = df["invoice_date"].dropna().median()
        df["invoice_date"] = df["invoice_date"].fillna(median_date)

    num_cols = [
        "quantity", "unit_price", "discount_percentage", "discount_amount",
        "taxable_value", "gst_rate", "cgst", "sgst", "igst",
        "cess_rate", "cess_amount", "total_tax", "total_amount"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    # 1. Feature Engineering
    df_prep = compute_consistency_features(df)
    df_prep = compute_date_features(df_prep)
    df_prep = compute_frequency_features(df_prep)
    df_prep = compute_duplicate_features(df_prep)
    feat_df, feature_cols = build_feature_matrix(df_prep)

    # Scaled feature matrix
    X = feat_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_samples = len(df)

    # 2. Model 1: Isolation Forest
    if_model = IsolationForest(
        n_estimators=100 if n_samples < 5000 else 200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
    if_preds = if_model.fit_predict(X_scaled)
    if_scores = -if_model.score_samples(X_scaled)
    if_anom = (if_preds == -1).astype(int)

    # 3. Model 2: Local Outlier Factor
    n_neighbors = min(20, n_samples - 1)
    lof_model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination="auto",
        n_jobs=-1
    )
    lof_preds = lof_model.fit_predict(X_scaled)
    lof_scores = -lof_model.negative_outlier_factor_
    lof_anom = (lof_preds == -1).astype(int)

    # 4. Model 3: Statistical Z-Score
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0
    Z = np.abs((X - means) / stds)
    max_z = np.max(Z, axis=1)
    z_anom = (max_z >= 3.0).astype(int)
    dom_idx = np.argmax(Z, axis=1)
    dom_feats = [feature_cols[i] for i in dom_idx]

    # 5. Score Normalization (0 - 100)
    isolation_contribution, lof_contribution, zscore_contribution, risk_score = (
        calculate_weighted_anomaly_index(if_scores, lof_scores, max_z)
    )
    risk_levels = [classify_risk_level(float(score)) for score in risk_score]

    # 7. Audit Explanations & Anomaly Types
    explanations = []
    detected_types = []
    for i in range(n_samples):
        lvl = risk_levels[i]
        scr = risk_score[i]
        models = []
        if if_anom[i]: models.append("Isolation Forest")
        if lof_anom[i]: models.append("Local Outlier Factor")
        if z_anom[i]: models.append("Z-Score")

        reasons = []
        dtype = "Normal"

        if df_prep["duplicate_transaction_flag"].iloc[i] == 1:
            reasons.append("duplicate transaction signature")
            dtype = "Duplicate Transaction"
        elif df_prep["future_date_flag"].iloc[i] == 1:
            reasons.append("post-dated invoice outside compliance period")
            dtype = "Future Invoice Date"
        elif df_prep["total_amount_difference"].iloc[i] > 100.0:
            reasons.append(f"major invoice total amount discrepancy (₹{df_prep['total_amount_difference'].iloc[i]:,.2f})")
            dtype = "Extreme Invoice Amount"
        elif df["unit_price"].iloc[i] > 100_000.0:
            reasons.append(f"unit price (₹{df['unit_price'].iloc[i]:,.2f}) significantly exceeds category benchmark")
            dtype = "Extreme Unit Price"
        elif df["quantity"].iloc[i] > 1_000.0:
            reasons.append(f"unusual bulk quantity ({df['quantity'].iloc[i]:,.1f} units)")
            dtype = "Unusual Quantity"
        elif df["discount_percentage"].iloc[i] > 50.0:
            reasons.append(f"abnormally high trade discount ({df['discount_percentage'].iloc[i]:.1f}%)")
            dtype = "Unusual Discount"
        elif df_prep["gst_difference"].iloc[i] > 50.0:
            reasons.append(f"tax calculation inconsistency (₹{df_prep['gst_difference'].iloc[i]:,.2f} variance)")
            dtype = "Tax Calculation Mismatch"
        elif df_prep["cgst_sgst_split_difference"].iloc[i] > 50.0:
            reasons.append("asymmetrical CGST/SGST tax split")
            dtype = "Incorrect CGST SGST Split"
        elif df_prep["igst_difference"].iloc[i] > 50.0:
            reasons.append("inter-state IGST calculation mismatch")
            dtype = "Incorrect IGST"

        if lvl != "Low Risk" and dtype == "Normal":
            dtype = "Statistical Model Outlier"

        if lvl == "Low Risk":
            exp = "No large anomaly was identified relative to this analysis batch. This is not a legal compliance finding."
        elif lvl == "Medium Risk":
            m_str = ", ".join(models) if models else "Statistical model"
            r_str = "; ".join(reasons) if reasons else f"variance in {dom_feats[i].replace('_', ' ')}"
            exp = f"Medium batch-relative anomaly index flagged by {m_str}. Verification is recommended due to {r_str}."
        else:
            m_str = ", ".join(models) if models else "Multi-model consensus"
            r_str = "; ".join(reasons) if reasons else f"extreme variance in {dom_feats[i].replace('_', ' ')}"
            exp = f"High batch-relative anomaly index ({scr:.1f}/100), flagged by {m_str}. Requires verification due to {r_str}; this is not proof of fraud."

        explanations.append(exp)
        detected_types.append(dtype)

    # 8. Aligned Analysis DataFrame
    analysis_df = pd.DataFrame({
        "invoice_no": df["invoice_no"].values,
        "isolation_score": if_scores.round(4),
        "isolation_anomaly": if_anom,
        "lof_score": lof_scores.round(4),
        "lof_anomaly": lof_anom,
        "max_abs_zscore": max_z.round(4),
        "zscore_anomaly": z_anom,
        "isolation_contribution": isolation_contribution.round(4),
        "lof_contribution": lof_contribution.round(4),
        "zscore_contribution": zscore_contribution.round(4),
        "risk_score": risk_score,
        "risk_level": risk_levels,
        "anomaly_type": detected_types,
        "explanation": explanations
    })

    return analysis_df
