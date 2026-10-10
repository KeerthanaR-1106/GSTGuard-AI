"""
generate_gst_dataset.py
=======================
Generates ~20,000 synthetic GST invoice records that resemble realistic
Indian GST transaction data.  No real GSTINs, names, or personal data are used.

Output files
------------
  data/gst_invoices.csv   – Raw transaction columns only (no ML features)
  data/ground_truth.csv   – invoice_no, actual_anomaly, anomaly_type
                            (used ONLY for model evaluation, never for training)

Run:
    python data/generate_gst_dataset.py
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Dataset parameters ───────────────────────────────────────────────────────
TOTAL_INVOICES       = 20_000
ANOMALY_FRACTION     = 0.05        # 5 % anomalous
N_ANOMALIES          = int(TOTAL_INVOICES * ANOMALY_FRACTION)   # 1 000
N_NORMAL             = TOTAL_INVOICES - N_ANOMALIES              # 19 000

N_SUPPLIERS = 500
N_BUYERS    = 1_000

# Dataset covers invoices issued in this period
DATASET_START = datetime(2023, 4, 1)
DATASET_END   = datetime(2024, 3, 31)

# ── Output paths ─────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
INVOICES_PATH    = os.path.join(BASE_DIR, "gst_invoices.csv")
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "ground_truth.csv")

# ── Reference data ───────────────────────────────────────────────────────────
GST_RATES = [5, 12, 18, 28]           # percent

CESS_RATES = {                         # cess applies mainly on 28 % bracket
    5: 0,
    12: 0,
    18: 0,
    28: 1,                            # 1 % cess for simplicity
}

PRODUCT_CATEGORIES = [
    "Electronics",
    "Pharmaceuticals",
    "Textiles",
    "Machinery",
    "Food & Beverages",
    "Chemicals",
    "Automobile Parts",
    "Construction Materials",
    "Agricultural Products",
    "IT Services",
]

# HSN codes mapped to categories (realistic but dummy values)
HSN_MAP = {
    "Electronics":            ["8471", "8517", "8528", "8544"],
    "Pharmaceuticals":        ["3004", "3006", "2941", "3002"],
    "Textiles":               ["5208", "5209", "6101", "6203"],
    "Machinery":              ["8413", "8419", "8428", "8479"],
    "Food & Beverages":       ["1905", "2009", "2106", "0402"],
    "Chemicals":              ["2814", "2901", "2915", "3808"],
    "Automobile Parts":       ["8708", "8714", "8716", "8409"],
    "Construction Materials": ["6810", "7214", "7308", "6802"],
    "Agricultural Products":  ["1001", "1006", "1201", "0713"],
    "IT Services":            ["9983", "9984", "9985", "9986"],
}

PAYMENT_METHODS = ["NEFT", "RTGS", "UPI", "Cheque", "Cash", "IMPS"]

# Indian state codes used in GSTIN (first 2 digits)
STATE_CODES = [
    "01","02","03","04","05","06","07","08","09","10",
    "11","12","13","14","15","16","17","18","19","20",
    "21","22","23","24","27","29","30","32","33","34","36"
]


# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_gstin(entity_num: int, entity_type: str = "supplier") -> str:
    """
    Create a realistic-looking but completely dummy GSTIN.
    Format: <2-digit state> <10-char PAN-like> <1 digit> Z <1 alphanum>
    Real GSTINs follow: SS AAAAA NNNN A Z N
    We use fictional entity numbers so no real GSTIN is reproduced.
    """
    state_code = STATE_CODES[entity_num % len(STATE_CODES)]
    # Build a fake 10-char PAN-like string
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # deliberately excludes I, O
    pan_letters = (
        letters[entity_num % 24]
        + letters[(entity_num // 24) % 24]
        + letters[(entity_num // 576) % 24]
        + letters[(entity_num // 13824) % 24]
        + letters[(entity_num // 331776) % 24]
    )
    pan_digits = f"{entity_num % 9999:04d}"
    pan_suffix = letters[(entity_num + 7) % 24]
    pan = pan_letters + pan_digits + pan_suffix   # 10 chars

    reg_num = str((entity_num % 9) + 1)           # 1–9
    last    = letters[(entity_num * 3 + 5) % 24]
    return f"{state_code}{pan}{reg_num}Z{last}"


def random_date(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between start and end."""
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def compute_gst_amounts(taxable_value: float, gst_rate: float,
                        cess_rate: float, is_intrastate: bool):
    """
    Compute all GST amount fields from taxable_value.

    Intra-state:  CGST = SGST = GST/2,  IGST = 0
    Inter-state:  CGST = 0, SGST = 0,   IGST = GST
    """
    total_gst     = round(taxable_value * gst_rate / 100, 2)
    cess_amount   = round(taxable_value * cess_rate / 100, 2)

    if is_intrastate:
        cgst = round(total_gst / 2, 2)
        sgst = round(total_gst / 2, 2)
        igst = 0.0
    else:
        cgst = 0.0
        sgst = 0.0
        igst = total_gst

    total_tax    = round(cgst + sgst + igst + cess_amount, 2)
    total_amount = round(taxable_value + total_tax, 2)

    return cgst, sgst, igst, cess_amount, total_tax, total_amount


def build_normal_invoice(invoice_no: str, supplier_id: int, buyer_id: int,
                         sup_gstin: str, buy_gstin: str,
                         inv_date: datetime) -> dict:
    """
    Generate one mathematically consistent, realistic normal invoice row.
    Returns a dict matching the gst_invoices.csv column schema.
    """
    category   = random.choice(PRODUCT_CATEGORIES)
    hsn_code   = random.choice(HSN_MAP[category])
    gst_rate   = random.choice(GST_RATES)
    cess_rate  = CESS_RATES[gst_rate]
    payment    = random.choice(PAYMENT_METHODS)

    quantity   = round(random.uniform(1, 500), 2)
    unit_price = round(random.uniform(50, 50_000), 2)

    # discount: 0–20 %
    disc_pct  = round(random.uniform(0, 20), 2)
    gross_val = round(quantity * unit_price, 2)
    disc_amt  = round(gross_val * disc_pct / 100, 2)
    taxable   = round(gross_val - disc_amt, 2)

    # Determine transaction type by state codes in GSTINs
    sup_state = sup_gstin[:2]
    buy_state = buy_gstin[:2]
    is_intra  = (sup_state == buy_state)

    cgst, sgst, igst, cess_amt, total_tax, total_amt = compute_gst_amounts(
        taxable, gst_rate, cess_rate, is_intra
    )

    return {
        "invoice_no":          invoice_no,
        "invoice_date":        inv_date.strftime("%Y-%m-%d"),
        "supplier_id":         f"SUP{supplier_id:04d}",
        "supplier_gstin":      sup_gstin,
        "buyer_id":            f"BUY{buyer_id:04d}",
        "buyer_gstin":         buy_gstin,
        "product_category":    category,
        "hsn_code":            hsn_code,
        "quantity":            quantity,
        "unit_price":          unit_price,
        "discount_percentage": disc_pct,
        "discount_amount":     disc_amt,
        "taxable_value":       taxable,
        "gst_rate":            gst_rate,
        "cgst":                cgst,
        "sgst":                sgst,
        "igst":                igst,
        "cess_rate":           cess_rate,
        "cess_amount":         cess_amt,
        "total_tax":           total_tax,
        "total_amount":        total_amt,
        "payment_method":      payment,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-build supplier / buyer master data
# ─────────────────────────────────────────────────────────────────────────────

supplier_ids  = list(range(1, N_SUPPLIERS + 1))
buyer_ids     = list(range(1, N_BUYERS + 1))

supplier_gstin_map = {sid: generate_gstin(sid, "supplier") for sid in supplier_ids}
buyer_gstin_map    = {bid: generate_gstin(bid + N_SUPPLIERS, "buyer") for bid in buyer_ids}


# ─────────────────────────────────────────────────────────────────────────────
#  Stage A – Generate normal invoices
# ─────────────────────────────────────────────────────────────────────────────

print(f"[INFO] Generating {N_NORMAL:,} normal invoices ...")

normal_rows = []
for i in range(1, N_NORMAL + 1):
    inv_no   = f"INV{i:06d}"
    sup_id   = random.choice(supplier_ids)
    buy_id   = random.choice(buyer_ids)
    inv_date = random_date(DATASET_START, DATASET_END)

    row = build_normal_invoice(
        inv_no,
        sup_id, buy_id,
        supplier_gstin_map[sup_id],
        buyer_gstin_map[buy_id],
        inv_date,
    )
    normal_rows.append(row)

# ─────────────────────────────────────────────────────────────────────────────
#  Stage B – Generate anomalous invoices
# ─────────────────────────────────────────────────────────────────────────────
#
#  11 anomaly types, each ~91 records (≈1 000 total).
#  Anomalies are injected AFTER normal invoice base values are created so that
#  only specific fields deviate — this makes them realistic rather than purely
#  random garbage.

ANOMALY_TYPES = [
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

# Distribute 1 000 anomalies evenly across 11 types
anomaly_counts = {}
per_type = N_ANOMALIES // len(ANOMALY_TYPES)
remainder = N_ANOMALIES % len(ANOMALY_TYPES)
for idx, atype in enumerate(ANOMALY_TYPES):
    anomaly_counts[atype] = per_type + (1 if idx < remainder else 0)

print(f"[INFO] Generating {N_ANOMALIES:,} anomalous invoices ...")
print("       Distribution:", {k: v for k, v in anomaly_counts.items()})

anomaly_rows       = []
ground_truth_rows  = []
invoice_counter    = N_NORMAL + 1   # continue numbering after normal invoices

# Pick a handful of suppliers/buyers for burst anomalies
BURST_SUPPLIERS = random.sample(supplier_ids, 10)
BURST_BUYERS    = random.sample(buyer_ids,    10)

def next_inv_no():
    global invoice_counter
    inv_no = f"INV{invoice_counter:06d}"
    invoice_counter += 1
    return inv_no


def make_anomaly_base(sup_id=None, buy_id=None, inv_date=None):
    """Return a base normal row that we will then corrupt."""
    sup_id   = sup_id   or random.choice(supplier_ids)
    buy_id   = buy_id   or random.choice(buyer_ids)
    inv_date = inv_date or random_date(DATASET_START, DATASET_END)
    inv_no   = next_inv_no()
    row = build_normal_invoice(
        inv_no, sup_id, buy_id,
        supplier_gstin_map[sup_id],
        buyer_gstin_map[buy_id],
        inv_date,
    )
    return row


# ── 1. Extreme Invoice Amount ────────────────────────────────────────────────
for _ in range(anomaly_counts["Extreme Invoice Amount"]):
    row = make_anomaly_base()
    # Inflate total_amount by a massive factor (10–50×)
    multiplier = random.uniform(10, 50)
    row["total_amount"] = round(row["total_amount"] * multiplier, 2)
    # Keep taxable_value / tax untouched so the mismatch is visible
    anomaly_rows.append(row)
    ground_truth_rows.append({
        "invoice_no":    row["invoice_no"],
        "actual_anomaly": 1,
        "anomaly_type":  "Extreme Invoice Amount",
    })

# ── 2. Extreme Unit Price ────────────────────────────────────────────────────
for _ in range(anomaly_counts["Extreme Unit Price"]):
    row = make_anomaly_base()
    # Unit price is 100–500× the normal range ceiling
    row["unit_price"] = round(random.uniform(5_000_000, 25_000_000), 2)
    # Recalculate downstream amounts so total_amount is also extreme
    gross = round(row["quantity"] * row["unit_price"], 2)
    disc  = round(gross * row["discount_percentage"] / 100, 2)
    taxable = round(gross - disc, 2)
    row["discount_amount"] = disc
    row["taxable_value"]   = taxable
    sup_state = row["supplier_gstin"][:2]
    buy_state = row["buyer_gstin"][:2]
    is_intra  = (sup_state == buy_state)
    cgst, sgst, igst, cess_amt, ttax, tamt = compute_gst_amounts(
        taxable, row["gst_rate"], row["cess_rate"], is_intra
    )
    row.update(cgst=cgst, sgst=sgst, igst=igst,
               cess_amount=cess_amt, total_tax=ttax, total_amount=tamt)
    anomaly_rows.append(row)
    ground_truth_rows.append({
        "invoice_no":    row["invoice_no"],
        "actual_anomaly": 1,
        "anomaly_type":  "Extreme Unit Price",
    })

# ── 3. Unusual Quantity ──────────────────────────────────────────────────────
for _ in range(anomaly_counts["Unusual Quantity"]):
    row = make_anomaly_base()
    # Quantity 5 000–50 000 (normal max is 500)
    row["quantity"] = round(random.uniform(5_000, 50_000), 2)
    gross   = round(row["quantity"] * row["unit_price"], 2)
    disc    = round(gross * row["discount_percentage"] / 100, 2)
    taxable = round(gross - disc, 2)
    row["discount_amount"] = disc
    row["taxable_value"]   = taxable
    sup_state = row["supplier_gstin"][:2]
    buy_state = row["buyer_gstin"][:2]
    is_intra  = (sup_state == buy_state)
    cgst, sgst, igst, cess_amt, ttax, tamt = compute_gst_amounts(
        taxable, row["gst_rate"], row["cess_rate"], is_intra
    )
    row.update(cgst=cgst, sgst=sgst, igst=igst,
               cess_amount=cess_amt, total_tax=ttax, total_amount=tamt)
    anomaly_rows.append(row)
    ground_truth_rows.append({
        "invoice_no":    row["invoice_no"],
        "actual_anomaly": 1,
        "anomaly_type":  "Unusual Quantity",
    })

# ── 4. Unusual Discount ──────────────────────────────────────────────────────
for _ in range(anomaly_counts["Unusual Discount"]):
    row = make_anomaly_base()
    # Discount 60–95 % (normal max is 20 %)
    row["discount_percentage"] = round(random.uniform(60, 95), 2)
    gross   = round(row["quantity"] * row["unit_price"], 2)
    disc    = round(gross * row["discount_percentage"] / 100, 2)
    taxable = max(round(gross - disc, 2), 0.01)   # avoid zero/negative
    row["discount_amount"] = disc
    row["taxable_value"]   = taxable
    sup_state = row["supplier_gstin"][:2]
    buy_state = row["buyer_gstin"][:2]
    is_intra  = (sup_state == buy_state)
    cgst, sgst, igst, cess_amt, ttax, tamt = compute_gst_amounts(
        taxable, row["gst_rate"], row["cess_rate"], is_intra
    )
    row.update(cgst=cgst, sgst=sgst, igst=igst,
               cess_amount=cess_amt, total_tax=ttax, total_amount=tamt)
    anomaly_rows.append(row)
    ground_truth_rows.append({
        "invoice_no":    row["invoice_no"],
        "actual_anomaly": 1,
        "anomaly_type":  "Unusual Discount",
    })

# ── 5. Tax Calculation Mismatch ──────────────────────────────────────────────
for _ in range(anomaly_counts["Tax Calculation Mismatch"]):
    row = make_anomaly_base()
    # Deliberately enter a wrong total_tax (off by 20–80 %)
    error_factor   = random.uniform(0.2, 0.8)
    row["total_tax"] = round(row["total_tax"] * (1 + error_factor), 2)
    # total_amount also wrong
    row["total_amount"] = round(row["taxable_value"] + row["total_tax"], 2)
    anomaly_rows.append(row)
    ground_truth_rows.append({
        "invoice_no":    row["invoice_no"],
        "actual_anomaly": 1,
        "anomaly_type":  "Tax Calculation Mismatch",
    })

# ── 6. Incorrect CGST SGST Split ────────────────────────────────────────────
for _ in range(anomaly_counts["Incorrect CGST SGST Split"]):
    row = make_anomaly_base()
    # Force intra-state GSTIN (same state code) so CGST+SGST should apply
    state = random.choice(STATE_CODES)
    sup_id = random.choice(supplier_ids)
    buy_id = random.choice(buyer_ids)
    # Override GSTINs to same state
    row["supplier_gstin"] = state + row["supplier_gstin"][2:]
    row["buyer_gstin"]    = state + row["buyer_gstin"][2:]
    # Correct tax
    gst = round(row["taxable_value"] * row["gst_rate"] / 100, 2)
    # Wrong split: e.g. 80/20 instead of 50/50
    wrong_ratio = random.uniform(0.1, 0.4)
    row["cgst"] = round(gst * wrong_ratio, 2)
    row["sgst"] = round(gst * (1 - wrong_ratio), 2)
    row["igst"] = 0.0
    anomaly_rows.append(row)
    ground_truth_rows.append({
        "invoice_no":    row["invoice_no"],
        "actual_anomaly": 1,
        "anomaly_type":  "Incorrect CGST SGST Split",
    })

# ── 7. Incorrect IGST ───────────────────────────────────────────────────────
for _ in range(anomaly_counts["Incorrect IGST"]):
    row = make_anomaly_base()
    # Force inter-state (different state codes)
    all_states = list(STATE_CODES)
    s1, s2 = random.sample(all_states, 2)
    row["supplier_gstin"] = s1 + row["supplier_gstin"][2:]
    row["buyer_gstin"]    = s2 + row["buyer_gstin"][2:]
    row["cgst"] = 0.0
    row["sgst"] = 0.0
    # Correct IGST
    correct_igst = round(row["taxable_value"] * row["gst_rate"] / 100, 2)
    # Set wrong IGST (50–200 % of correct)
    wrong_factor = random.choice([0.5, 0.7, 1.5, 2.0])
    row["igst"] = round(correct_igst * wrong_factor, 2)
    row["total_tax"]    = round(row["igst"] + row["cess_amount"], 2)
    row["total_amount"] = round(row["taxable_value"] + row["total_tax"], 2)
    anomaly_rows.append(row)
    ground_truth_rows.append({
        "invoice_no":    row["invoice_no"],
        "actual_anomaly": 1,
        "anomaly_type":  "Incorrect IGST",
    })

# ── 8. Duplicate Transaction ─────────────────────────────────────────────────
#  Pick existing normal invoices and clone them with a new invoice_no
dup_sources = random.sample(normal_rows, anomaly_counts["Duplicate Transaction"])
for src in dup_sources:
    row = src.copy()
    row["invoice_no"] = next_inv_no()   # new number, same transaction details
    anomaly_rows.append(row)
    ground_truth_rows.append({
        "invoice_no":    row["invoice_no"],
        "actual_anomaly": 1,
        "anomaly_type":  "Duplicate Transaction",
    })

# ── 9. Unusual Supplier Activity ────────────────────────────────────────────
#  A small set of suppliers each produce a burst of invoices in a single day
per_burst_supplier = anomaly_counts["Unusual Supplier Activity"]
burst_date = DATASET_START + timedelta(days=random.randint(10, 300))
chosen_burst_sup = random.choice(BURST_SUPPLIERS)
for _ in range(per_burst_supplier):
    row = make_anomaly_base(sup_id=chosen_burst_sup, inv_date=burst_date)
    anomaly_rows.append(row)
    ground_truth_rows.append({
        "invoice_no":    row["invoice_no"],
        "actual_anomaly": 1,
        "anomaly_type":  "Unusual Supplier Activity",
    })

# ── 10. Unusual Buyer Activity ───────────────────────────────────────────────
per_burst_buyer = anomaly_counts["Unusual Buyer Activity"]
burst_date2 = DATASET_START + timedelta(days=random.randint(10, 300))
chosen_burst_buy = random.choice(BURST_BUYERS)
for _ in range(per_burst_buyer):
    row = make_anomaly_base(buy_id=chosen_burst_buy, inv_date=burst_date2)
    anomaly_rows.append(row)
    ground_truth_rows.append({
        "invoice_no":    row["invoice_no"],
        "actual_anomaly": 1,
        "anomaly_type":  "Unusual Buyer Activity",
    })

# ── 11. Future Invoice Date ──────────────────────────────────────────────────
FUTURE_START = DATASET_END + timedelta(days=1)
FUTURE_END   = DATASET_END + timedelta(days=365)
for _ in range(anomaly_counts["Future Invoice Date"]):
    future_date = random_date(FUTURE_START, FUTURE_END)
    row = make_anomaly_base(inv_date=future_date)
    anomaly_rows.append(row)
    ground_truth_rows.append({
        "invoice_no":    row["invoice_no"],
        "actual_anomaly": 1,
        "anomaly_type":  "Future Invoice Date",
    })

# ─────────────────────────────────────────────────────────────────────────────
#  Stage C – Build ground truth for normal invoices
# ─────────────────────────────────────────────────────────────────────────────

for row in normal_rows:
    ground_truth_rows.append({
        "invoice_no":    row["invoice_no"],
        "actual_anomaly": 0,
        "anomaly_type":  "Normal",
    })

# ─────────────────────────────────────────────────────────────────────────────
#  Stage D – Combine, shuffle, and save
# ─────────────────────────────────────────────────────────────────────────────

all_rows = normal_rows + anomaly_rows
df_invoices = pd.DataFrame(all_rows)
df_invoices = df_invoices.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

df_gt = pd.DataFrame(ground_truth_rows)

print(f"\n[INFO] Total invoices generated : {len(df_invoices):,}")
print(f"[INFO] Total ground truth rows  : {len(df_gt):,}")
print(f"\n[INFO] Anomaly type distribution in ground truth:")
print(df_gt[df_gt["actual_anomaly"] == 1]["anomaly_type"].value_counts().to_string())

# Verify column schema – must match spec exactly
REQUIRED_COLS = [
    "invoice_no", "invoice_date", "supplier_id", "supplier_gstin",
    "buyer_id", "buyer_gstin", "product_category", "hsn_code",
    "quantity", "unit_price", "discount_percentage", "discount_amount",
    "taxable_value", "gst_rate", "cgst", "sgst", "igst",
    "cess_rate", "cess_amount", "total_tax", "total_amount",
    "payment_method",
]
missing = [c for c in REQUIRED_COLS if c not in df_invoices.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Save
df_invoices[REQUIRED_COLS].to_csv(INVOICES_PATH, index=False)
df_gt.to_csv(GROUND_TRUTH_PATH, index=False)

print(f"\n[SUCCESS] Saved: {INVOICES_PATH}")
print(f"[SUCCESS] Saved: {GROUND_TRUTH_PATH}")

# ── Quick sanity checks ──────────────────────────────────────────────────────
print("\n-- Sanity checks --")
print(f"  Total rows      : {len(df_invoices)}")
print(f"  Normal invoices : {len(normal_rows)}")
print(f"  Anomalous       : {len(anomaly_rows)}")
print(f"  Anomaly %%      : {len(anomaly_rows)/len(df_invoices)*100:.2f} %%")
print(f"  Unique suppliers: {df_invoices['supplier_id'].nunique()}")
print(f"  Unique buyers   : {df_invoices['buyer_id'].nunique()}")
print(f"  Invoice_no range: {df_invoices['invoice_no'].min()} to {df_invoices['invoice_no'].max()}")
print(f"  Invoice date range: {df_invoices['invoice_date'].min()} to {df_invoices['invoice_date'].max()}")
print("\n  total_amount stats:")
print(df_invoices['total_amount'].describe().to_string())

print("\n[DONE] Stage 1 complete.")
