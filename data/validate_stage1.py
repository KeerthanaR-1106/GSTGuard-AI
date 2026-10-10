"""
validate_stage1.py
==================
Comprehensive validation of Stage 1 output files:
  - data/gst_invoices.csv
  - data/ground_truth.csv

Prints a PASS / FAIL report for every check.
Does NOT modify any file.

Run:
    python data/validate_stage1.py
"""

import os, re, sys
import pandas as pd
import numpy as np

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
INVOICES_PATH    = os.path.join(BASE_DIR, "gst_invoices.csv")
GT_PATH          = os.path.join(BASE_DIR, "ground_truth.csv")

DATASET_START    = pd.Timestamp("2023-04-01")
DATASET_END      = pd.Timestamp("2024-03-31")

TOLERANCE        = 0.05   # Rs 0.05 rounding tolerance for tax/amount checks
# Real GSTIN format: SS AAAAA NNNN A N Z A (15 characters total)
# SS=2-digit state code, AAAAA=5 letters (PAN chars 1-5),
# NNNN=4 digits (PAN chars 6-9), A=1 letter (PAN char 10),
# N=1 registration digit (1-9), Z=literal, A=1 alphanumeric checksum
GSTIN_RE         = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z]\dZ[A-Z0-9]$")

PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0

results = []


def check(label, condition, detail="", warn=False):
    global PASS_COUNT, FAIL_COUNT, WARN_COUNT
    status = "PASS" if condition else ("WARN" if warn else "FAIL")
    if condition:
        PASS_COUNT += 1
    elif warn:
        WARN_COUNT += 1
    else:
        FAIL_COUNT += 1
    tag = f"[{status:4}]"
    line = f"{tag}  {label}"
    if detail:
        line += f"\n        {detail}"
    results.append((status, label, detail))
    print(line)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =============================================================================
print("\n" + "="*60)
print("  GST Fraud Detection -- Stage 1 Validation Report")
print("="*60)

# ── Load files ────────────────────────────────────────────────────────────────
section("0. FILE EXISTENCE")

inv_exists = os.path.exists(INVOICES_PATH)
gt_exists  = os.path.exists(GT_PATH)
check("gst_invoices.csv exists", inv_exists, INVOICES_PATH)
check("ground_truth.csv exists", gt_exists,  GT_PATH)

if not inv_exists or not gt_exists:
    print("\n[ABORT] One or both CSV files are missing. Cannot continue.")
    sys.exit(1)

df  = pd.read_csv(INVOICES_PATH)
df_gt = pd.read_csv(GT_PATH)

# ── Convert types early ───────────────────────────────────────────────────────
df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

# =============================================================================
section("1. ROW COUNTS")

check("gst_invoices.csv has exactly 20,000 rows",
      len(df) == 20_000,
      f"Actual: {len(df):,}")

check("ground_truth.csv has exactly 20,000 rows",
      len(df_gt) == 20_000,
      f"Actual: {len(df_gt):,}")

# =============================================================================
section("2. COLUMNS")

REQUIRED_INV_COLS = [
    "invoice_no", "invoice_date", "supplier_id", "supplier_gstin",
    "buyer_id", "buyer_gstin", "product_category", "hsn_code",
    "quantity", "unit_price", "discount_percentage", "discount_amount",
    "taxable_value", "gst_rate", "cgst", "sgst", "igst",
    "cess_rate", "cess_amount", "total_tax", "total_amount",
    "payment_method",
]
REQUIRED_GT_COLS = ["invoice_no", "actual_anomaly", "anomaly_type"]

missing_inv = [c for c in REQUIRED_INV_COLS if c not in df.columns]
extra_inv   = [c for c in df.columns if c not in REQUIRED_INV_COLS]

check("gst_invoices.csv has all 22 required columns",
      len(missing_inv) == 0,
      f"Missing: {missing_inv}" if missing_inv else "All 22 columns present")

check("gst_invoices.csv has NO extra columns (no ML features leaked)",
      len(extra_inv) == 0,
      f"Extra: {extra_inv}" if extra_inv else "No extra columns")

missing_gt = [c for c in REQUIRED_GT_COLS if c not in df_gt.columns]
check("ground_truth.csv has all 3 required columns",
      len(missing_gt) == 0,
      f"Missing: {missing_gt}" if missing_gt else "All 3 columns present")

print(f"\n  Actual columns in gst_invoices.csv ({len(df.columns)}):")
for i, c in enumerate(df.columns, 1):
    print(f"    {i:2}. {c}")

# =============================================================================
section("3. NORMAL / ANOMALOUS SPLIT")

normal_gt  = (df_gt["actual_anomaly"] == 0).sum()
anomaly_gt = (df_gt["actual_anomaly"] == 1).sum()
pct_anomaly = anomaly_gt / len(df_gt) * 100

check("Normal invoices = 19,000 (95%)",
      normal_gt == 19_000,
      f"Actual: {normal_gt:,}")

check("Anomalous invoices = 1,000 (5%)",
      anomaly_gt == 1_000,
      f"Actual: {anomaly_gt:,}")

check("Anomaly fraction is ~5%",
      abs(pct_anomaly - 5.0) < 0.1,
      f"Actual: {pct_anomaly:.3f}%")

# =============================================================================
section("4. ANOMALY TYPE DISTRIBUTION")

EXPECTED_TYPES = [
    "Extreme Invoice Amount",
    "Extreme Unit Price",
    "Unusual Quantity",
    "Unusual Discount",
    "Tax Calculation Mismatch",
    "Incorrect CGST SGST Split",
    "Incorrect IGST",
    "Duplicate Transaction",
    "Unusual Supplier Activity",
    "Unusual Buyer Activity",
    "Future Invoice Date",
]

anom_dist = df_gt[df_gt["actual_anomaly"] == 1]["anomaly_type"].value_counts()
print("\n  Anomaly type distribution:")
for atype, cnt in anom_dist.items():
    print(f"    {atype:<35}: {cnt}")

all_types_present = all(t in anom_dist.index for t in EXPECTED_TYPES)
check("All 11 anomaly types are present", all_types_present,
      f"Missing: {[t for t in EXPECTED_TYPES if t not in anom_dist.index]}"
      if not all_types_present else "")

counts_reasonable = all(80 <= anom_dist.get(t, 0) <= 120 for t in EXPECTED_TYPES)
check("Each anomaly type has 80-120 samples (approx. balanced)",
      counts_reasonable,
      {t: int(anom_dist.get(t, 0)) for t in EXPECTED_TYPES
       if not (80 <= anom_dist.get(t, 0) <= 120)} or "All within range")

# =============================================================================
section("5. MISSING VALUES")

missing_counts = df.isnull().sum()
total_missing  = missing_counts.sum()
check("No missing values in gst_invoices.csv",
      total_missing == 0,
      f"Missing per column:\n{missing_counts[missing_counts>0].to_string()}"
      if total_missing > 0 else "0 missing values")

missing_gt_counts = df_gt.isnull().sum()
check("No missing values in ground_truth.csv",
      missing_gt_counts.sum() == 0,
      f"Missing: {missing_gt_counts[missing_gt_counts>0].to_string()}"
      if missing_gt_counts.sum() > 0 else "0 missing values")

# =============================================================================
section("6. DUPLICATE INVOICE NUMBERS")

dup_in_inv = df["invoice_no"].duplicated().sum()
check("No duplicate invoice_no in gst_invoices.csv",
      dup_in_inv == 0,
      f"Duplicates found: {dup_in_inv}")

dup_in_gt = df_gt["invoice_no"].duplicated().sum()
check("No duplicate invoice_no in ground_truth.csv",
      dup_in_gt == 0,
      f"Duplicates found: {dup_in_gt}")

# ── invoice_no alignment ──────────────────────────────────────────────────────
inv_set = set(df["invoice_no"])
gt_set  = set(df_gt["invoice_no"])
only_in_inv = inv_set - gt_set
only_in_gt  = gt_set  - inv_set
check("invoice_no in both files are identical sets",
      len(only_in_inv) == 0 and len(only_in_gt) == 0,
      f"Only in invoices: {len(only_in_inv)}, Only in GT: {len(only_in_gt)}")

# =============================================================================
section("7. DATE VALIDATION")

bad_dates = df["invoice_date"].isna().sum()
check("All invoice_date values are parseable dates",
      bad_dates == 0,
      f"Unparseable dates: {bad_dates}")

# Normal invoices should fall within dataset range
normal_inv_nos = df_gt[df_gt["actual_anomaly"] == 0]["invoice_no"]
df_normal = df[df["invoice_no"].isin(normal_inv_nos)].copy()

out_of_range_normal = df_normal[
    (df_normal["invoice_date"] < DATASET_START) |
    (df_normal["invoice_date"] > DATASET_END)
]
check("Normal invoices: all dates within 2023-04-01 to 2024-03-31",
      len(out_of_range_normal) == 0,
      f"Out-of-range normal dates: {len(out_of_range_normal)}")

# Future-date anomalies should have dates > DATASET_END
future_inv_nos = df_gt[df_gt["anomaly_type"] == "Future Invoice Date"]["invoice_no"]
df_future = df[df["invoice_no"].isin(future_inv_nos)]
all_future = (df_future["invoice_date"] > DATASET_END).all()
check("'Future Invoice Date' anomalies: all dates are after 2024-03-31",
      all_future,
      f"Rows with date <= dataset end: {(~(df_future['invoice_date'] > DATASET_END)).sum()}")

# =============================================================================
section("8. NEGATIVE / ZERO VALUE CHECKS")

neg_qty    = (df["quantity"]   <= 0).sum()
neg_price  = (df["unit_price"] <= 0).sum()
neg_taxable= (df["taxable_value"] < 0).sum()
neg_gstrate= (df["gst_rate"]   <= 0).sum()
neg_ttax   = (df["total_tax"]  < 0).sum()
neg_tamt   = (df["total_amount"]< 0).sum()

check("No zero/negative quantity",        neg_qty    == 0, f"{neg_qty} rows")
check("No zero/negative unit_price",      neg_price  == 0, f"{neg_price} rows")
check("No negative taxable_value",        neg_taxable== 0, f"{neg_taxable} rows")
check("No zero/negative gst_rate",        neg_gstrate== 0, f"{neg_gstrate} rows")
check("No negative total_tax",            neg_ttax   == 0, f"{neg_ttax} rows")
check("No negative total_amount",         neg_tamt   == 0, f"{neg_tamt} rows")

neg_cgst   = (df["cgst"] < 0).sum()
neg_sgst   = (df["sgst"] < 0).sum()
neg_igst   = (df["igst"] < 0).sum()
check("No negative cgst",  neg_cgst == 0, f"{neg_cgst} rows")
check("No negative sgst",  neg_sgst == 0, f"{neg_sgst} rows")
check("No negative igst",  neg_igst == 0, f"{neg_igst} rows")

# =============================================================================
section("9. GSTIN FORMAT VALIDATION")

def is_valid_gstin(g):
    if not isinstance(g, str):
        return False
    return bool(GSTIN_RE.match(g))

invalid_sup = (~df["supplier_gstin"].apply(is_valid_gstin)).sum()
invalid_buy = (~df["buyer_gstin"].apply(is_valid_gstin)).sum()

check("All supplier_gstin match GSTIN format",
      invalid_sup == 0,
      f"Invalid count: {invalid_sup}  Sample: {df[~df['supplier_gstin'].apply(is_valid_gstin)]['supplier_gstin'].head(3).tolist()}"
      if invalid_sup else "")

check("All buyer_gstin match GSTIN format",
      invalid_buy == 0,
      f"Invalid count: {invalid_buy}  Sample: {df[~df['buyer_gstin'].apply(is_valid_gstin)]['buyer_gstin'].head(3).tolist()}"
      if invalid_buy else "")

# =============================================================================
section("10. GST RATE VALIDITY")

valid_rates = {5, 12, 18, 28}
invalid_rates = df[~df["gst_rate"].isin(valid_rates)]["gst_rate"].unique()
check("All gst_rate values are in {5, 12, 18, 28}",
      len(invalid_rates) == 0,
      f"Invalid rates found: {invalid_rates}")

# =============================================================================
section("11. MATHEMATICAL CONSISTENCY -- NORMAL INVOICES")
# Check on normal invoices only (anomalies deliberately violate these)

df_n = df[df["invoice_no"].isin(normal_inv_nos)].copy()
print(f"\n  Checking {len(df_n):,} normal invoices for math consistency ...")

# 11a. discount_amount = quantity * unit_price * discount_percentage / 100
df_n["calc_gross"]    = (df_n["quantity"] * df_n["unit_price"]).round(2)
df_n["calc_disc"]     = (df_n["calc_gross"] * df_n["discount_percentage"] / 100).round(2)
df_n["disc_diff"]     = (df_n["discount_amount"] - df_n["calc_disc"]).abs()
bad_disc = (df_n["disc_diff"] > TOLERANCE).sum()
check("Normal invoices: discount_amount = gross * discount% / 100",
      bad_disc == 0,
      f"{bad_disc} rows deviate by more than Rs{TOLERANCE}")

# 11b. taxable_value = gross - discount_amount
df_n["calc_taxable"]  = (df_n["calc_gross"] - df_n["discount_amount"]).round(2)
df_n["taxable_diff"]  = (df_n["taxable_value"] - df_n["calc_taxable"]).abs()
bad_taxable = (df_n["taxable_diff"] > TOLERANCE).sum()
check("Normal invoices: taxable_value = gross_value - discount_amount",
      bad_taxable == 0,
      f"{bad_taxable} rows deviate by more than Rs{TOLERANCE}")

# 11c. expected_gst = taxable_value * gst_rate / 100
df_n["calc_gst"]      = (df_n["taxable_value"] * df_n["gst_rate"] / 100).round(2)

# 11d. CGST+SGST+IGST check
# Intra-state: cgst=sgst=gst/2, igst=0
# Inter-state: cgst=0, sgst=0, igst=gst
df_n["sup_state"] = df_n["supplier_gstin"].str[:2]
df_n["buy_state"] = df_n["buyer_gstin"].str[:2]
df_n["is_intra"]  = df_n["sup_state"] == df_n["buy_state"]

intra = df_n[df_n["is_intra"]].copy()
inter = df_n[~df_n["is_intra"]].copy()

# Intra-state checks
intra_cgst_diff = (intra["cgst"] - (intra["calc_gst"] / 2)).abs()
bad_intra_cgst  = (intra_cgst_diff > TOLERANCE).sum()
check(f"Intra-state ({len(intra):,} rows): cgst = gst/2",
      bad_intra_cgst == 0,
      f"{bad_intra_cgst} rows deviate")

intra_sgst_diff = (intra["sgst"] - (intra["calc_gst"] / 2)).abs()
bad_intra_sgst  = (intra_sgst_diff > TOLERANCE).sum()
check(f"Intra-state ({len(intra):,} rows): sgst = gst/2",
      bad_intra_sgst == 0,
      f"{bad_intra_sgst} rows deviate")

bad_intra_igst = (intra["igst"].abs() > TOLERANCE).sum()
check(f"Intra-state ({len(intra):,} rows): igst = 0",
      bad_intra_igst == 0,
      f"{bad_intra_igst} rows have non-zero igst")

# Inter-state checks
inter_igst_diff = (inter["igst"] - inter["calc_gst"]).abs()
bad_inter_igst  = (inter_igst_diff > TOLERANCE).sum()
check(f"Inter-state ({len(inter):,} rows): igst = taxable * gst_rate / 100",
      bad_inter_igst == 0,
      f"{bad_inter_igst} rows deviate")

bad_inter_cgst = (inter["cgst"].abs() > TOLERANCE).sum()
bad_inter_sgst = (inter["sgst"].abs() > TOLERANCE).sum()
check(f"Inter-state ({len(inter):,} rows): cgst = 0",
      bad_inter_cgst == 0,
      f"{bad_inter_cgst} rows have non-zero cgst")
check(f"Inter-state ({len(inter):,} rows): sgst = 0",
      bad_inter_sgst == 0,
      f"{bad_inter_sgst} rows have non-zero sgst")

# 11e. cess_amount = taxable_value * cess_rate / 100
df_n["calc_cess"]     = (df_n["taxable_value"] * df_n["cess_rate"] / 100).round(2)
cess_diff = (df_n["cess_amount"] - df_n["calc_cess"]).abs()
bad_cess  = (cess_diff > TOLERANCE).sum()
check("Normal invoices: cess_amount = taxable * cess_rate / 100",
      bad_cess == 0,
      f"{bad_cess} rows deviate")

# 11f. total_tax = cgst + sgst + igst + cess_amount
df_n["calc_total_tax"] = (df_n["cgst"] + df_n["sgst"] + df_n["igst"] + df_n["cess_amount"]).round(2)
ttax_diff = (df_n["total_tax"] - df_n["calc_total_tax"]).abs()
bad_ttax  = (ttax_diff > TOLERANCE).sum()
check("Normal invoices: total_tax = cgst+sgst+igst+cess_amount",
      bad_ttax == 0,
      f"{bad_ttax} rows deviate by >{TOLERANCE}")

# 11g. total_amount = taxable_value + total_tax
df_n["calc_total_amt"] = (df_n["taxable_value"] + df_n["total_tax"]).round(2)
tamt_diff = (df_n["total_amount"] - df_n["calc_total_amt"]).abs()
bad_tamt  = (tamt_diff > TOLERANCE).sum()
check("Normal invoices: total_amount = taxable_value + total_tax",
      bad_tamt == 0,
      f"{bad_tamt} rows deviate by >{TOLERANCE}")

# Statistics
print(f"\n  Normal invoice amount statistics:")
print(f"    quantity     : min={df_n['quantity'].min():.2f}, max={df_n['quantity'].max():.2f}, mean={df_n['quantity'].mean():.2f}")
print(f"    unit_price   : min={df_n['unit_price'].min():.2f}, max={df_n['unit_price'].max():.2f}, mean={df_n['unit_price'].mean():.2f}")
print(f"    total_amount : min={df_n['total_amount'].min():.2f}, max={df_n['total_amount'].max():.2f}, mean={df_n['total_amount'].mean():.2f}")
print(f"    discount_pct : min={df_n['discount_percentage'].min():.2f}, max={df_n['discount_percentage'].max():.2f}")
print(f"    Intra-state rows : {len(intra):,}  ({len(intra)/len(df_n)*100:.1f}%)")
print(f"    Inter-state rows : {len(inter):,}  ({len(inter)/len(df_n)*100:.1f}%)")

# =============================================================================
section("12. INTENDED ANOMALIES ACTUALLY CONTAIN THE ANOMALY")

def get_anom_df(atype):
    nos = df_gt[df_gt["anomaly_type"] == atype]["invoice_no"]
    return df[df["invoice_no"].isin(nos)].copy()

# 12a. Extreme Invoice Amount
# The anomaly inflates total_amount while taxable_value+total_tax remain correct.
# Correct detection: total_amount >> taxable_value + total_tax (ratio should be 10-50x).
ea = get_anom_df("Extreme Invoice Amount")
ea["expected_amt"] = ea["taxable_value"] + ea["total_tax"]
ea["inflation_ratio"] = ea["total_amount"] / ea["expected_amt"].replace(0, np.nan)
# Every EIA row must have total_amount at least 5x higher than expected
ea_extreme = (ea["inflation_ratio"] >= 5).sum()
check("'Extreme Invoice Amount': total_amount inflated >= 5x above (taxable+tax)",
      ea_extreme >= len(ea) * 0.95,
      f"{ea_extreme}/{len(ea)} rows have inflation ratio>=5 | "
      f"min ratio={ea['inflation_ratio'].min():.1f}, max={ea['inflation_ratio'].max():.1f}")

# 12b. Extreme Unit Price
eup = get_anom_df("Extreme Unit Price")
normal_max_price = df_n["unit_price"].quantile(0.99)
eup_extreme = (eup["unit_price"] > normal_max_price * 10).sum()
check("'Extreme Unit Price': most rows have unit_price >> 10x 99th-pct of normal",
      eup_extreme >= len(eup) * 0.8,
      f"{eup_extreme}/{len(eup)} rows qualify (threshold: 10x={normal_max_price*10:.0f})")

# 12c. Unusual Quantity
uq = get_anom_df("Unusual Quantity")
normal_max_qty = df_n["quantity"].quantile(0.99)
uq_extreme = (uq["quantity"] > normal_max_qty * 5).sum()
check("'Unusual Quantity': most rows have quantity >> 5x 99th-pct of normal",
      uq_extreme >= len(uq) * 0.8,
      f"{uq_extreme}/{len(uq)} rows qualify (threshold: 5x={normal_max_qty*5:.0f})")

# 12d. Unusual Discount -- disc_pct > 50%
ud = get_anom_df("Unusual Discount")
ud_high = (ud["discount_percentage"] > 50).sum()
check("'Unusual Discount': all rows have discount_percentage > 50%",
      ud_high == len(ud),
      f"{ud_high}/{len(ud)} rows have disc_pct > 50%")

# 12e. Tax Calculation Mismatch -- total_tax deviates from expected by >10%
tcm = get_anom_df("Tax Calculation Mismatch")
tcm["exp_tax"] = (
    tcm["taxable_value"] * tcm["gst_rate"] / 100
    + tcm["taxable_value"] * tcm["cess_rate"] / 100
).round(2)
tcm["tax_deviation_pct"] = ((tcm["total_tax"] - tcm["exp_tax"]).abs() / tcm["exp_tax"].replace(0, np.nan) * 100)
tcm_mismatch = (tcm["tax_deviation_pct"] > 10).sum()
check("'Tax Calculation Mismatch': total_tax deviates >10% from expected",
      tcm_mismatch >= len(tcm) * 0.8,
      f"{tcm_mismatch}/{len(tcm)} rows deviate >10%")

# 12f. Incorrect CGST SGST Split -- split should NOT be 50/50
ics = get_anom_df("Incorrect CGST SGST Split")
ics["gst_total"] = ics["cgst"] + ics["sgst"]
ics["ratio"]     = (ics["cgst"] / ics["gst_total"].replace(0, np.nan)).round(2)
ics_wrong = ((ics["ratio"] - 0.5).abs() > 0.05).sum()
check("'Incorrect CGST SGST Split': CGST/SGST ratio is NOT 50/50",
      ics_wrong >= len(ics) * 0.8,
      f"{ics_wrong}/{len(ics)} rows have non-50/50 split")

# 12g. Incorrect IGST -- igst deviates from expected
iig = get_anom_df("Incorrect IGST")
iig["exp_igst"] = (iig["taxable_value"] * iig["gst_rate"] / 100).round(2)
iig["igst_dev"] = ((iig["igst"] - iig["exp_igst"]).abs() / iig["exp_igst"].replace(0, np.nan) * 100)
iig_wrong = (iig["igst_dev"] > 10).sum()
check("'Incorrect IGST': igst deviates >10% from expected",
      iig_wrong >= len(iig) * 0.8,
      f"{iig_wrong}/{len(iig)} rows deviate >10%")

# 12h. Duplicate Transaction -- find matching rows on key fields
dup = get_anom_df("Duplicate Transaction")
KEY_COLS = ["supplier_gstin", "buyer_gstin", "taxable_value", "gst_rate",
            "quantity", "unit_price"]
# For each dup row, check if there is another row with the same key cols
df_all_key = df[KEY_COLS + ["invoice_no"]].copy()
merged = dup.merge(df_all_key, on=KEY_COLS, suffixes=("_dup","_orig"))
# Count rows where original invoice_no != dup invoice_no (i.e. a match exists)
merged_matches = merged[merged["invoice_no_dup"] != merged["invoice_no_orig"]]
dup_matched = merged_matches["invoice_no_dup"].nunique()
check("'Duplicate Transaction': each row has a matching transaction elsewhere",
      dup_matched == len(dup),
      f"{dup_matched}/{len(dup)} duplicate rows found a match in the dataset")

# 12i. Unusual Supplier Activity -- supplier has >50 invoices in a single day
usa = get_anom_df("Unusual Supplier Activity")
if "invoice_date" not in usa.columns or usa["invoice_date"].isna().all():
    usa_ok = False
    usa_detail = "invoice_date not usable"
else:
    usa["date_str"] = usa["invoice_date"].astype(str)
    sup_day_counts = usa.groupby(["supplier_id","date_str"]).size()
    usa_ok = (sup_day_counts.max() > 50)
    usa_detail = f"Max supplier-day burst: {sup_day_counts.max()}"
check("'Unusual Supplier Activity': supplier has >50 invoices in one day",
      usa_ok, usa_detail)

# 12j. Unusual Buyer Activity -- buyer has >50 invoices in a single day
uba = get_anom_df("Unusual Buyer Activity")
if "invoice_date" not in uba.columns or uba["invoice_date"].isna().all():
    uba_ok = False
    uba_detail = "invoice_date not usable"
else:
    uba["date_str"] = uba["invoice_date"].astype(str)
    buy_day_counts = uba.groupby(["buyer_id","date_str"]).size()
    uba_ok = (buy_day_counts.max() > 50)
    uba_detail = f"Max buyer-day burst: {buy_day_counts.max()}"
check("'Unusual Buyer Activity': buyer has >50 invoices in one day",
      uba_ok, uba_detail)

# 12k. Future Invoice Date -- dates > DATASET_END
fid = get_anom_df("Future Invoice Date")
if fid["invoice_date"].isna().any():
    fid_ok = False
    fid_detail = "Some dates not parsed"
else:
    fid_ok = (fid["invoice_date"] > DATASET_END).all()
    fid_detail = f"All {len(fid)} rows have date > {DATASET_END.date()}"
check("'Future Invoice Date': all dates > 2024-03-31",
      fid_ok, fid_detail)

# =============================================================================
section("13. ADDITIONAL SANITY CHECKS")

# Unique supplier/buyer counts
uniq_sup = df["supplier_id"].nunique()
uniq_buy = df["buyer_id"].nunique()
check("Dataset has ~500 unique suppliers",
      450 <= uniq_sup <= 500,
      f"Actual: {uniq_sup}")
check("Dataset has ~1000 unique buyers",
      900 <= uniq_buy <= 1000,
      f"Actual: {uniq_buy}")

# Payment methods
valid_payments = {"NEFT","RTGS","UPI","Cheque","Cash","IMPS"}
inv_payments = set(df["payment_method"].unique()) - valid_payments
check("All payment_method values are in expected set",
      len(inv_payments) == 0,
      f"Invalid: {inv_payments}")

# GST rate coverage
rate_dist = df["gst_rate"].value_counts().sort_index()
check("All 4 GST rates (5,12,18,28) appear in dataset",
      set(rate_dist.index) >= {5,12,18,28},
      f"Rates found: {sorted(rate_dist.index.tolist())}")

print(f"\n  GST rate distribution:")
for rate, cnt in rate_dist.items():
    print(f"    {rate:2}%  ->  {cnt:,} invoices")

# =============================================================================
section("SUMMARY")

total_checks = PASS_COUNT + FAIL_COUNT + WARN_COUNT
print(f"\n  Total checks : {total_checks}")
print(f"  PASS         : {PASS_COUNT}")
print(f"  FAIL         : {FAIL_COUNT}")
print(f"  WARN         : {WARN_COUNT}")

if FAIL_COUNT == 0:
    print("\n  *** ALL CHECKS PASSED -- Stage 1 is VALID ***")
    print("  Ready to proceed to Stage 2.")
else:
    print(f"\n  *** {FAIL_COUNT} CHECK(S) FAILED -- Fix Stage 1 before proceeding ***")
    print("\n  Failed checks:")
    for status, label, detail in results:
        if status == "FAIL":
            print(f"    - {label}")
            if detail:
                print(f"      {detail}")
