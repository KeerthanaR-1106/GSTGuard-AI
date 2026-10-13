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

import os
from functools import lru_cache
from typing import NamedTuple

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


class _TrainingModels(NamedTuple):
    reference_frame: pd.DataFrame
    feature_cols: list[str]
    scaler: StandardScaler
    isolation_forest: IsolationForest
    lof: LocalOutlierFactor
    means: np.ndarray
    stds: np.ndarray
    isolation_range: tuple[float, float]
    lof_range: tuple[float, float]


def calculate_weighted_anomaly_index(
    if_scores,
    lof_scores,
    max_z,
    isolation_range=None,
    lof_range=None,
):
    """Combine detector outputs into a transparent 0–100 index."""
    def scale(scores, reference_range=None):
        scores = np.asarray(scores, dtype=float)
        if reference_range is None:
            minimum, maximum = np.min(scores), np.max(scores)
        else:
            minimum, maximum = reference_range
        if maximum <= minimum:
            return np.zeros_like(scores, dtype=float)
        return np.clip((scores - minimum) / (maximum - minimum) * 100.0, 0.0, 100.0)

    weight_total = (
        Config.WEIGHT_ISOLATION_FOREST + Config.WEIGHT_LOF + Config.WEIGHT_ZSCORE
    )
    isolation_contribution = (
        scale(if_scores, isolation_range)
        * Config.WEIGHT_ISOLATION_FOREST / weight_total
    )
    lof_contribution = (
        scale(lof_scores, lof_range) * Config.WEIGHT_LOF / weight_total
    )
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


def _build_features(
    df_raw: pd.DataFrame,
    duplicate_context_start: int | None = None,
):
    """Prepare raw invoice rows using the same 43 features for training and scoring."""
    df = df_raw.copy()
    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"], errors="coerce", format="mixed"
    )
    if df["invoice_date"].isna().any():
        raise ValueError("Training or upload data contains an invalid invoice_date.")

    numeric_columns = [
        "quantity", "unit_price", "discount_percentage", "discount_amount",
        "taxable_value", "gst_rate", "cgst", "sgst", "igst",
        "cess_rate", "cess_amount", "total_tax", "total_amount"
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        values = df[column].to_numpy(dtype=float)
        if df[column].isna().any() or not np.isfinite(values).all():
            raise ValueError(f"Training or upload data has invalid '{column}' values.")
        df[column] = df[column].clip(lower=0.0)

    prepared = compute_consistency_features(df)
    prepared = compute_date_features(prepared)
    prepared = compute_frequency_features(prepared)
    prepared = compute_duplicate_features(
        prepared,
        duplicate_context_start=duplicate_context_start,
    )
    feature_frame, feature_cols = build_feature_matrix(prepared)
    return df, prepared, feature_frame[feature_cols].to_numpy(dtype=float), feature_cols


@lru_cache(maxsize=1)
def _fit_training_models(training_path: str, _modified_ns: int) -> _TrainingModels:
    """Fit and cache detectors against the fixed training CSV, never the upload."""
    if os.stat(training_path).st_mtime_ns != _modified_ns:
        raise RuntimeError("The training CSV changed while preparing model training.")
    training = pd.read_csv(training_path)
    if len(training) < 2:
        raise ValueError("The training CSV must contain at least two invoices.")

    _, _, x_train, feature_cols = _build_features(training)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
    isolation_forest.fit(x_train_scaled)

    lof = LocalOutlierFactor(
        n_neighbors=min(20, len(training) - 1),
        contamination="auto",
        novelty=True,
        n_jobs=-1
    )
    lof.fit(x_train_scaled)

    isolation_scores = -isolation_forest.score_samples(x_train_scaled)
    lof_scores = -lof.negative_outlier_factor_
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    stds[stds == 0] = 1.0

    trained_models = _TrainingModels(
        reference_frame=training,
        feature_cols=feature_cols,
        scaler=scaler,
        isolation_forest=isolation_forest,
        lof=lof,
        means=means,
        stds=stds,
        isolation_range=(
            float(np.min(isolation_scores)),
            float(np.max(isolation_scores)),
        ),
        lof_range=(float(np.min(lof_scores)), float(np.max(lof_scores))),
    )
    if os.stat(training_path).st_mtime_ns != _modified_ns:
        raise RuntimeError("The training CSV changed while fitting the detectors.")
    return trained_models


def _get_training_models() -> _TrainingModels:
    training_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "gst_invoices.csv",
    )
    if not os.path.exists(training_path):
        raise FileNotFoundError(f"Training CSV not found: {training_path}")
    modified_ns = os.stat(training_path).st_mtime_ns
    return _fit_training_models(training_path, modified_ns)


def run_anomaly_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Score uploaded invoices against detectors fitted on the training CSV.
    Returns analysis records for uploaded invoices only.
    """
    if len(df_raw) < 2:
        raise ValueError("At least two invoices are required for anomaly analysis.")

    training_models = _get_training_models()
    training_count = len(training_models.reference_frame)
    combined_frame = pd.concat(
        [training_models.reference_frame, df_raw],
        ignore_index=True,
    )
    all_df, all_df_prep, all_X, feature_cols = _build_features(
        combined_frame,
        duplicate_context_start=training_count,
    )
    if feature_cols != training_models.feature_cols:
        raise ValueError("Uploaded CSV features do not match the training model.")
    df = all_df.iloc[training_count:].reset_index(drop=True)
    df_prep = all_df_prep.iloc[training_count:].reset_index(drop=True)
    X = all_X[training_count:]
    X_scaled = training_models.scaler.transform(X)
    n_samples = len(df)

    # 2. Apply Isolation Forest trained on the reference CSV.
    if_preds = training_models.isolation_forest.predict(X_scaled)
    if_scores = -training_models.isolation_forest.score_samples(X_scaled)
    if_anom = (if_preds == -1).astype(int)

    # 3. Apply novelty-mode LOF trained on the reference CSV.
    lof_preds = training_models.lof.predict(X_scaled)
    lof_scores = -training_models.lof.score_samples(X_scaled)
    lof_anom = (lof_preds == -1).astype(int)

    # 4. Statistical Z-Score uses statistics fitted on the training CSV.
    Z = np.abs((X - training_models.means) / training_models.stds)
    max_z = np.max(Z, axis=1)
    z_anom = (max_z >= 3.0).astype(int)
    dom_idx = np.argmax(Z, axis=1)
    dom_feats = [feature_cols[i] for i in dom_idx]

    isolation_contribution, lof_contribution, zscore_contribution, risk_score = (
        calculate_weighted_anomaly_index(
            if_scores,
            lof_scores,
            max_z,
            isolation_range=training_models.isolation_range,
            lof_range=training_models.lof_range,
        )
    )
    score_levels = [classify_risk_level(float(score)) for score in risk_score]
    risk_levels = []

    # 7. Audit Explanations & Anomaly Types
    explanations = []
    detected_types = []
    for i in range(n_samples):
        lvl = score_levels[i]
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

        if dtype == "Normal" and lvl != "Low Risk":
            dtype = "Statistical Model Outlier"

        severity = lvl if dtype != "Normal" else "Normal"
        risk_levels.append(severity)

        if severity == "Normal":
            exp = "No large anomaly was identified relative to the training baseline. This is not a legal compliance finding."
        elif severity == "Low Risk":
            m_str = ", ".join(models) if models else "Rule-based screening"
            r_str = "; ".join(reasons) if reasons else f"variance in {dom_feats[i].replace('_', ' ')}"
            exp = f"Low-severity anomaly identified by {m_str}. Review is recommended due to {r_str}."
        elif severity == "Medium Risk":
            m_str = ", ".join(models) if models else "Statistical model"
            r_str = "; ".join(reasons) if reasons else f"variance in {dom_feats[i].replace('_', ' ')}"
            exp = f"Medium training-baseline anomaly index flagged by {m_str}. Verification is recommended due to {r_str}."
        else:
            m_str = ", ".join(models) if models else "Multi-model consensus"
            r_str = "; ".join(reasons) if reasons else f"extreme variance in {dom_feats[i].replace('_', ' ')}"
            exp = f"High training-baseline anomaly index ({scr:.1f}/100), flagged by {m_str}. Requires verification due to {r_str}; this is not proof of fraud."

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
