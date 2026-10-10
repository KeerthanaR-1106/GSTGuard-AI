"""
preprocess.py
=============
Stage 2: Data Validation + Feature Engineering

Reads   : data/gst_invoices.csv
Outputs : data/ml_features.csv   (ML-ready feature matrix, NO labels)

Steps:
  1. Load & basic validation (missing, types, duplicates, dates, negatives)
  2. GST / tax / amount consistency checks  --> consistency error features
  3. Frequency-based features               --> supplier/buyer activity
  4. Duplicate detection feature
  5. Date-based features
  6. Save feature matrix (numeric columns only, no strings, no labels)

Run:
    python preprocessing/preprocess.py
"""

import os
import sys
from typing import Optional
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(BASE_DIR)
INVOICES_PATH = os.path.join(ROOT_DIR, "data", "gst_invoices.csv")
OUTPUT_PATH   = os.path.join(ROOT_DIR, "data", "ml_features.csv")

# ── Parameters ────────────────────────────────────────────────────────────────
DATASET_END = pd.Timestamp("2024-03-31")   # invoices after this = future date


# =============================================================================
#  SECTION 1 – Load & Basic Validation
# =============================================================================

def load_and_validate(path: str) -> pd.DataFrame:
    """Load CSV and run basic validation checks. Raises on critical errors."""

    print("[1/6] Loading raw invoice data ...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Invoice file not found: {path}")

    df = pd.read_csv(path, parse_dates=["invoice_date"])
    print(f"      Loaded {len(df):,} rows x {len(df.columns)} columns")

    # ── 1a. Required columns ──────────────────────────────────────────────────
    REQUIRED = [
        "invoice_no", "invoice_date", "supplier_id", "supplier_gstin",
        "buyer_id", "buyer_gstin", "product_category", "hsn_code",
        "quantity", "unit_price", "discount_percentage", "discount_amount",
        "taxable_value", "gst_rate", "cgst", "sgst", "igst",
        "cess_rate", "cess_amount", "total_tax", "total_amount",
        "payment_method",
    ]
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ── 1b. Missing values ────────────────────────────────────────────────────
    missing_counts = df.isnull().sum()
    has_missing = missing_counts[missing_counts > 0]
    if len(has_missing) > 0:
        print(f"      WARNING: Missing values detected:\n{has_missing}")
        # Fill numeric NaNs with column median; string NaNs with 'Unknown'
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        str_cols = df.select_dtypes(include=["object"]).columns
        df[str_cols] = df[str_cols].fillna("Unknown")
        print("      Missing values filled (median for numeric, 'Unknown' for strings).")
    else:
        print("      No missing values found.")

    # ── 1c. Parse invoice_date if not already datetime ────────────────────────
    if not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    bad_dates = df["invoice_date"].isna().sum()
    if bad_dates > 0:
        print(f"      WARNING: {bad_dates} unparseable dates. Filling with median date.")
        median_date = df["invoice_date"].dropna().median()
        df["invoice_date"] = df["invoice_date"].fillna(median_date)

    # ── 1d. Numeric sanity: clip negatives that shouldn't be negative ─────────
    NON_NEG = ["quantity", "unit_price", "discount_percentage",
               "discount_amount", "taxable_value", "gst_rate",
               "cgst", "sgst", "igst", "cess_rate", "cess_amount",
               "total_tax", "total_amount"]
    for col in NON_NEG:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"      WARNING: {neg_count} negative values in '{col}'. Clipping to 0.")
            df[col] = df[col].clip(lower=0)

    print(f"      Validation complete. {len(df):,} rows retained.")
    return df


# =============================================================================
#  SECTION 2 – GST Consistency Features
# =============================================================================

def compute_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive features that measure deviation from correct GST calculations.
    A value of 0 means perfect consistency; non-zero reveals anomalies.
    """
    print("[2/6] Computing GST / tax / amount consistency features ...")

    df = df.copy()

    # ── Gross value: quantity × unit_price ────────────────────────────────────
    df["calculated_gross_value"] = (df["quantity"] * df["unit_price"]).round(2)

    # ── Discount ─────────────────────────────────────────────────────────────
    df["calculated_discount"] = (
        df["calculated_gross_value"] * df["discount_percentage"] / 100
    ).round(2)
    # How much does the stored discount differ from the mathematically correct one?
    df["discount_difference"] = (
        df["discount_amount"] - df["calculated_discount"]
    ).abs().round(4)

    # ── Taxable value: gross − discount ───────────────────────────────────────
    df["calculated_taxable_value"] = (
        df["calculated_gross_value"] - df["discount_amount"]
    ).round(2)
    df["taxable_value_difference"] = (
        df["taxable_value"] - df["calculated_taxable_value"]
    ).abs().round(4)

    # ── Expected GST: taxable × rate / 100 ───────────────────────────────────
    df["expected_gst"] = (df["taxable_value"] * df["gst_rate"] / 100).round(2)

    # Determine transaction type from state codes in GSTINs
    # Intra-state: same first 2 chars in supplier_gstin and buyer_gstin
    df["_sup_state"] = df["supplier_gstin"].str[:2]
    df["_buy_state"] = df["buyer_gstin"].str[:2]
    df["_is_intra"]  = (df["_sup_state"] == df["_buy_state"]).astype(int)

    # ── CGST / SGST / IGST consistency ───────────────────────────────────────
    # For intra-state: CGST = SGST = GST/2, IGST = 0
    # For inter-state: CGST = 0, SGST = 0, IGST = GST
    df["expected_cgst"] = np.where(
        df["_is_intra"] == 1, (df["expected_gst"] / 2).round(2), 0
    )
    df["expected_sgst"] = np.where(
        df["_is_intra"] == 1, (df["expected_gst"] / 2).round(2), 0
    )
    df["expected_igst"] = np.where(
        df["_is_intra"] == 0, df["expected_gst"].round(2), 0
    )

    # CGST-SGST split difference: |CGST - SGST| should be near 0 for intra-state
    # (inter-state rows will both be 0, so diff is also 0 there -- correct)
    df["cgst_sgst_split_difference"] = (df["cgst"] - df["sgst"]).abs().round(4)

    # IGST difference
    df["igst_difference"] = (df["igst"] - df["expected_igst"]).abs().round(4)

    # Calculated total GST = CGST + SGST + IGST
    df["calculated_gst"] = (df["cgst"] + df["sgst"] + df["igst"]).round(2)
    df["gst_difference"]  = (df["calculated_gst"] - df["expected_gst"]).abs().round(4)

    # ── Cess ─────────────────────────────────────────────────────────────────
    df["calculated_cess"] = (df["taxable_value"] * df["cess_rate"] / 100).round(2)

    # ── Total tax = CGST + SGST + IGST + cess ────────────────────────────────
    df["calculated_total_tax"] = (
        df["cgst"] + df["sgst"] + df["igst"] + df["cess_amount"]
    ).round(2)
    df["total_tax_difference"] = (
        df["total_tax"] - df["calculated_total_tax"]
    ).abs().round(4)

    # ── Total amount = taxable + total_tax ───────────────────────────────────
    df["calculated_total_amount"] = (
        df["taxable_value"] + df["total_tax"]
    ).round(2)
    df["total_amount_difference"] = (
        df["total_amount"] - df["calculated_total_amount"]
    ).abs().round(4)

    # ── Amount-to-taxable ratio: flags extreme total_amount inflation ─────────
    # Normal invoices: ratio ≈ 1.05 to 1.35 (taxable + 5-28% GST + cess)
    # Extreme Invoice Amount anomaly: ratio = 10-50
    df["amount_to_taxable_ratio"] = (
        df["total_amount"] / df["taxable_value"].replace(0, np.nan)
    ).round(4)
    # Cap ratio to avoid inf/NaN from near-zero taxable values
    df["amount_to_taxable_ratio"] = df["amount_to_taxable_ratio"].clip(upper=500).fillna(1.2)

    print(f"      Consistency features computed.")
    return df


# =============================================================================
#  SECTION 3 – Date Features
# =============================================================================

def compute_date_features(df: pd.DataFrame, allowed_end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Derive date-based features."""
    print("[3/6] Computing date-based features ...")

    df = df.copy()

    # Only flag dates beyond a caller-supplied policy cutoff. The synthetic
    # dataset boundary is not a live compliance-period boundary.
    df["future_date_flag"] = (
        (df["invoice_date"] > allowed_end).astype(int)
        if allowed_end is not None else 0
    )
    df["days_beyond_allowed_end"] = (
        (df["invoice_date"] - allowed_end).dt.days.clip(lower=0)
        if allowed_end is not None else 0
    )

    # Invoice month (1-12) -- useful as a frequency context feature
    df["invoice_month"] = df["invoice_date"].dt.month

    # Day of week (0=Monday ... 6=Sunday)
    df["invoice_day_of_week"] = df["invoice_date"].dt.dayofweek

    print(f"      Date features computed. Future date rows: {df['future_date_flag'].sum()}")
    return df


# =============================================================================
#  SECTION 4 – Frequency / Activity Features
# =============================================================================

def compute_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute how often a supplier, buyer, or supplier-buyer pair appears.
    High frequency can indicate unusual activity.
    """
    print("[4/6] Computing frequency / activity features ...")

    df = df.copy()

    # ── Overall supplier & buyer invoice counts ───────────────────────────────
    supplier_freq = df.groupby("supplier_id")["invoice_no"].transform("count")
    buyer_freq    = df.groupby("buyer_id")["invoice_no"].transform("count")

    df["supplier_frequency"] = supplier_freq
    df["buyer_frequency"]    = buyer_freq

    # ── Supplier-Buyer pair frequency ────────────────────────────────────────
    df["_pair"] = df["supplier_id"].astype(str) + "_" + df["buyer_id"].astype(str)
    pair_freq = df.groupby("_pair")["invoice_no"].transform("count")
    df["supplier_buyer_frequency"] = pair_freq

    # ── Daily frequency features ──────────────────────────────────────────────
    # "How many invoices did this supplier raise on the same date?"
    df["_date_str"] = df["invoice_date"].dt.date.astype(str)

    sup_daily = (
        df.groupby(["supplier_id", "_date_str"])["invoice_no"]
        .transform("count")
    )
    buy_daily = (
        df.groupby(["buyer_id", "_date_str"])["invoice_no"]
        .transform("count")
    )
    all_daily = (
        df.groupby("_date_str")["invoice_no"].transform("count")
    )

    df["supplier_daily_frequency"] = sup_daily
    df["buyer_daily_frequency"]    = buy_daily
    df["daily_transaction_frequency"] = all_daily

    # Drop helper columns
    df.drop(columns=["_pair", "_date_str"], inplace=True)

    print("      Frequency features computed.")
    print(f"      Max supplier daily freq : {df['supplier_daily_frequency'].max()}")
    print(f"      Max buyer daily freq    : {df['buyer_daily_frequency'].max()}")
    return df


# =============================================================================
#  SECTION 5 – Duplicate Detection Feature
# =============================================================================

def compute_duplicate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect near-duplicate transactions.
    A duplicate is defined as another row with the same:
      supplier_gstin, buyer_gstin, quantity, unit_price, gst_rate, taxable_value
    but a different invoice_no.
    """
    print("[5/6] Computing duplicate transaction features ...")

    df = df.copy()

    # Signature: key fields that identify the transaction content
    DUPE_KEYS = [
        "supplier_gstin", "buyer_gstin",
        "quantity", "unit_price",
        "gst_rate", "taxable_value",
    ]

    # Round numeric keys to avoid floating-point mismatches
    df["_q_r"]  = df["quantity"].round(1)
    df["_up_r"] = df["unit_price"].round(1)
    df["_tv_r"] = df["taxable_value"].round(1)
    df["_sig"]  = (
        df["supplier_gstin"].astype(str) + "|" +
        df["buyer_gstin"].astype(str)    + "|" +
        df["_q_r"].astype(str)           + "|" +
        df["_up_r"].astype(str)          + "|" +
        df["gst_rate"].astype(str)       + "|" +
        df["_tv_r"].astype(str)
    )

    # Count how many times each signature appears
    sig_counts = df.groupby("_sig")["invoice_no"].transform("count")
    df["duplicate_transaction_frequency"] = sig_counts

    # Flag: 1 if there is at least one other row with the same signature
    df["duplicate_transaction_flag"] = (sig_counts > 1).astype(int)

    # Drop helper columns
    df.drop(columns=["_q_r", "_up_r", "_tv_r", "_sig"], inplace=True)

    n_dup = df["duplicate_transaction_flag"].sum()
    print(f"      Duplicate transactions flagged: {n_dup}")
    return df


# =============================================================================
#  SECTION 6 – Assemble Final ML Feature Matrix
# =============================================================================

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only numeric ML features.
    Excludes:
      - Identifier columns (invoice_no, supplier_id, buyer_id, GSTINs)
      - String/categorical columns
      - Labels (actual_anomaly, anomaly_type) -- NOT present in this file
      - Internal helper columns (prefixed with _)
    """
    print("[6/6] Building ML feature matrix ...")

    # These are the ML feature columns in the order they will be saved
    FEATURE_COLS = [
        # -- Raw numeric invoice fields --
        "quantity",
        "unit_price",
        "discount_percentage",
        "discount_amount",
        "taxable_value",
        "gst_rate",
        "cgst",
        "sgst",
        "igst",
        "cess_rate",
        "cess_amount",
        "total_tax",
        "total_amount",

        # -- Derived consistency features --
        "calculated_gross_value",
        "calculated_discount",
        "discount_difference",
        "calculated_taxable_value",
        "taxable_value_difference",
        "expected_gst",
        "calculated_gst",
        "gst_difference",
        "expected_cgst",
        "expected_sgst",
        "expected_igst",
        "cgst_sgst_split_difference",
        "igst_difference",
        "calculated_total_tax",
        "total_tax_difference",
        "calculated_total_amount",
        "total_amount_difference",
        "amount_to_taxable_ratio",

        # -- Date features --
        "future_date_flag",
        "days_beyond_allowed_end",
        "invoice_month",
        "invoice_day_of_week",

        # -- Frequency / activity features --
        "supplier_frequency",
        "buyer_frequency",
        "supplier_buyer_frequency",
        "supplier_daily_frequency",
        "buyer_daily_frequency",
        "daily_transaction_frequency",

        # -- Duplicate features --
        "duplicate_transaction_frequency",
        "duplicate_transaction_flag",
    ]

    # Verify all expected columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Build output: invoice_no (for joining back to results) + features
    out = df[["invoice_no"] + FEATURE_COLS].copy()

    # Ensure no NaN or Inf remains
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_counts = out.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"      WARNING: Remaining NaNs -- filling with column median:")
        print(nan_counts[nan_counts > 0].to_string())
        num_cols = out.select_dtypes(include=[np.number]).columns
        out[num_cols] = out[num_cols].fillna(out[num_cols].median())

    print(f"      Feature matrix: {len(out):,} rows x {len(FEATURE_COLS)} features")
    return out, FEATURE_COLS


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  GST Fraud Detection -- Stage 2: Preprocessing")
    print("=" * 60)

    # Step 1 – Load & validate
    df = load_and_validate(INVOICES_PATH)

    # Step 2 – GST consistency features
    df = compute_consistency_features(df)

    # Step 3 – Date features
    df = compute_date_features(df, allowed_end=DATASET_END)

    # Step 4 – Frequency features
    df = compute_frequency_features(df)

    # Step 5 – Duplicate detection
    df = compute_duplicate_features(df)

    # Step 6 – Build & save feature matrix
    feature_df, feature_cols = build_feature_matrix(df)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    # ── Summary report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Input rows          : {len(df):,}")
    print(f"  Output rows         : {len(feature_df):,}")
    print(f"  Feature columns     : {len(feature_cols)}")
    print(f"  Output file         : {OUTPUT_PATH}")
    print()
    print("  Feature columns:")
    for i, col in enumerate(feature_cols, 1):
        vmin = feature_df[col].min()
        vmax = feature_df[col].max()
        vmean = feature_df[col].mean()
        print(f"    {i:2}. {col:<40} min={vmin:>12.2f}  max={vmax:>14.2f}  mean={vmean:>12.2f}")

    print()
    print("  Consistency check summary (on full dataset):")
    for feat in ["discount_difference", "taxable_value_difference",
                 "gst_difference", "total_tax_difference",
                 "total_amount_difference"]:
        nonzero = (feature_df[feat] > 0.05).sum()
        print(f"    {feat:<40}: {nonzero:>5} rows with deviation > Rs0.05")

    print()
    print(f"  Future-date invoices flagged : {feature_df['future_date_flag'].sum()}")
    print(f"  Duplicate transactions flagged: {feature_df['duplicate_transaction_flag'].sum()}")
    print()
    print("[DONE] Stage 2 complete. ml_features.csv saved.")


if __name__ == "__main__":
    main()
