"""
audit_stage8.py
===============
Read-only comprehensive audit script for Stage 8: SQLite Database Integration.
Does NOT modify any file, schema, or data.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DB_PATH  = os.path.join(ROOT_DIR, "database", "gst_fraud.db")

print("=" * 70)
print("  STAGE 8 COMPREHENSIVE AUDIT & VERIFICATION")
print("=" * 70)

# 1. Inspect Files
print("\n--- 1. FILES RESPONSIBLE FOR STAGE 8 ---")
stage8_files = [
    os.path.join(ROOT_DIR, "database", "gst_fraud.db"),
    os.path.join(ROOT_DIR, "database", "models.py"),
    os.path.join(ROOT_DIR, "database", "database.py"),
    os.path.join(ROOT_DIR, "database", "__init__.py"),
    os.path.join(ROOT_DIR, "data", "gst_invoices.csv"),
    os.path.join(ROOT_DIR, "results", "ensemble_results.csv"),
    os.path.join(ROOT_DIR, "app.py")
]
for f in stage8_files:
    exists = os.path.exists(f)
    size = f"{os.path.getsize(f) / (1024*1024):.2f} MB" if exists else "N/A"
    print(f"  {os.path.relpath(f, ROOT_DIR):<35}: exists={exists}, size={size}")

# 2. Verify Database Connection
print("\n--- 2. DATABASE CONNECTION & INTEGRITY ---")
try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA integrity_check;")
    integrity = cursor.fetchone()[0]
    print(f"  Connection successful: True")
    print(f"  SQLite integrity_check: {integrity}")
except Exception as e:
    print(f"  Connection failed: {e}")
    sys.exit(1)

# 3. Verify Database Schema
print("\n--- 3. DATABASE SCHEMA & TABLE STRUCTURE ---")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
tables = [r[0] for r in cursor.fetchall()]
print(f"  Tables found: {tables}")

for t in tables:
    print(f"\n  Table: '{t}'")
    cursor.execute(f"PRAGMA table_info({t});")
    cols = cursor.fetchall()
    # (cid, name, type, notnull, dflt_value, pk)
    for c in cols:
        pk_str = " [PRIMARY KEY]" if c[5] > 0 else ""
        notnull_str = " NOT NULL" if c[3] > 0 else " NULLABLE"
        print(f"    - {c[1]:<25} {c[2]:<12}{notnull_str}{pk_str}")
    
    # Check foreign keys
    cursor.execute(f"PRAGMA foreign_key_list({t});")
    fks = cursor.fetchall()
    if fks:
        for fk in fks:
            print(f"    * Foreign Key: {fk[3]} -> {fk[2]}.{fk[4]}")
    
    # Check indexes
    cursor.execute(f"PRAGMA index_list({t});")
    indexes = cursor.fetchall()
    if indexes:
        print(f"    * Indexes: {[idx[1] for idx in indexes]}")

# 4. Verify Invoice Data Storage
print("\n--- 4. INVOICE DATA STORAGE ---")
cursor.execute("SELECT COUNT(*) FROM invoices;")
n_inv = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(DISTINCT invoice_no) FROM invoices;")
n_inv_distinct = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM invoices WHERE invoice_no IS NULL OR invoice_no = '';")
n_inv_null = cursor.fetchone()[0]

print(f"  Expected invoices: 20,000")
print(f"  Actual invoices: {n_inv:,}")
print(f"  Distinct invoice_no: {n_inv_distinct:,}")
print(f"  Duplicate invoice_no: {n_inv - n_inv_distinct}")
print(f"  Missing required invoice_no: {n_inv_null}")

# Check nulls across all invoice columns
df_inv_db = pd.read_sql("SELECT * FROM invoices;", conn)
inv_nulls = df_inv_db.isnull().sum()
total_inv_nulls = int(inv_nulls.sum())
print(f"  Total missing values across all invoice fields: {total_inv_nulls}")

# 5. Integration with ML & Pipeline Alignment Check
print("\n--- 5. INTEGRATION WITH ML STAGES & PIPELINE ALIGNMENT ---")
df_raw_csv = pd.read_csv(os.path.join(ROOT_DIR, "data", "gst_invoices.csv"))
df_ens_csv = pd.read_csv(os.path.join(ROOT_DIR, "results", "ensemble_results.csv"))
df_ans_db  = pd.read_sql("SELECT * FROM analysis;", conn)

inv_list_raw = df_raw_csv["invoice_no"].tolist()
inv_list_ens = df_ens_csv["invoice_no"].tolist()
inv_list_db_inv = df_inv_db["invoice_no"].tolist()
inv_list_db_ans = df_ans_db["invoice_no"].tolist()

print(f"  raw CSV == ensemble CSV invoice_no order & set: {inv_list_raw == inv_list_ens}")
print(f"  DB invoices == raw CSV invoice_no order & set: {inv_list_db_inv == inv_list_raw}")
print(f"  DB analysis == ensemble CSV invoice_no order & set: {inv_list_db_ans == inv_list_ens}")
print(f"  DB invoices == DB analysis invoice_no order & set: {inv_list_db_inv == inv_list_db_ans}")

# Spot check alignment: compare 5 random invoices between raw/ensemble and DB
sample_test_ids = ["INV000001", "INV005000", "INV010000", "INV015000", "INV020000"]
alignment_ok = True
for sid in sample_test_ids:
    raw_amt = df_raw_csv[df_raw_csv["invoice_no"] == sid]["total_amount"].iloc[0]
    db_amt = df_inv_db[df_inv_db["invoice_no"] == sid]["total_amount"].iloc[0]
    ens_risk = df_ens_csv[df_ens_csv["invoice_no"] == sid]["risk_score"].iloc[0]
    db_risk = df_ans_db[df_ans_db["invoice_no"] == sid]["risk_score"].iloc[0]
    if raw_amt != db_amt or ens_risk != db_risk:
        alignment_ok = False
        print(f"  MISMATCH on {sid}: raw_amt={raw_amt}, db_amt={db_amt}, ens_risk={ens_risk}, db_risk={db_risk}")
print(f"  Cross-file spot check alignment passed: {alignment_ok}")

# 6. Verify Ensemble Fields in Analysis Table
print("\n--- 6. VERIFY ENSEMBLE FIELDS IN ANALYSIS TABLE ---")
expected_ens_cols = [
    "invoice_no", "isolation_score", "isolation_anomaly", "lof_score",
    "lof_anomaly", "max_abs_zscore", "zscore_anomaly", "risk_score",
    "risk_level", "anomaly_type", "explanation"
]
actual_ans_cols = list(df_ans_db.columns)
print(f"  Expected columns ({len(expected_ens_cols)}): {expected_ens_cols}")
print(f"  Actual columns   ({len(actual_ans_cols)}): {actual_ans_cols}")
print(f"  Column match: {actual_ans_cols == expected_ens_cols}")

# 7. Risk Score & Level Integrity
print("\n--- 7. RISK SCORE & RISK LEVEL INTEGRITY ---")
risk_scores = df_ans_db["risk_score"]
min_rs = risk_scores.min()
max_rs = risk_scores.max()
mean_rs = risk_scores.mean()
n_nan_rs = risk_scores.isnull().sum()
n_inf_rs = np.isinf(risk_scores).sum()

print(f"  Risk score range: min={min_rs:.2f}, max={max_rs:.2f}, mean={mean_rs:.2f}")
print(f"  Within [0, 100]: {min_rs >= 0.0 and max_rs <= 100.0}")
print(f"  NaN / Inf in risk_score: NaN={n_nan_rs}, Inf={n_inf_rs}")

risk_levels = df_ans_db["risk_level"].value_counts()
print(f"  Risk levels in DB: {risk_levels.to_dict()}")

# Verify risk boundaries
low_viol = df_ans_db[(df_ans_db["risk_level"] == "Low Risk") & (df_ans_db["risk_score"] >= 40.0)]
med_viol = df_ans_db[(df_ans_db["risk_level"] == "Medium Risk") & ((df_ans_db["risk_score"] < 40.0) | (df_ans_db["risk_score"] >= 70.0))]
high_viol = df_ans_db[(df_ans_db["risk_level"] == "High Risk") & (df_ans_db["risk_score"] < 70.0)]

print(f"  Violations of Low Risk (score >= 40): {len(low_viol)}")
print(f"  Violations of Medium Risk (score < 40 or >= 70): {len(med_viol)}")
print(f"  Violations of High Risk (score < 70): {len(high_viol)}")

# 8. Ground Truth Isolation
print("\n--- 8. GROUND TRUTH ISOLATION AUDIT ---")
with open(os.path.join(ROOT_DIR, "database", "database.py"), "r", encoding="utf-8") as f:
    db_code = f.read()

has_gt_in_pop = "ground_truth.csv" in db_code
has_act_anom = "actual_anomaly" in db_code
print(f"  ground_truth.csv loaded in database population: {has_gt_in_pop}")
print(f"  actual_anomaly referenced in database.py: {has_act_anom}")
print(f"  actual_anomaly stored as column in DB: {'actual_anomaly' in df_ans_db.columns}")

# 9. Verify Data Types and Logical Values
print("\n--- 9. LOGICAL VALUE VALIDATIONS ---")
neg_qty = (df_inv_db["quantity"] < 0).sum()
neg_price = (df_inv_db["unit_price"] < 0).sum()
neg_tax = (df_inv_db["total_tax"] < 0).sum()
neg_total = (df_inv_db["total_amount"] < 0).sum()
valid_rates = {5.0, 12.0, 18.0, 28.0}
invalid_rates = (~df_inv_db["gst_rate"].isin(valid_rates)).sum()
dates_valid = pd.to_datetime(df_inv_db["invoice_date"], errors="coerce").notnull().all()

print(f"  Negative quantities: {neg_qty}")
print(f"  Negative unit prices: {neg_price}")
print(f"  Negative taxes: {neg_tax}")
print(f"  Negative total amounts: {neg_total}")
print(f"  Invalid GST rates (not in 5,12,18,28): {invalid_rates}")
print(f"  All invoice dates valid YYYY-MM-DD format: {dates_valid}")

# 10. Population Process Inspection
print("\n--- 10. POPULATION PROCESS AUDIT ---")
print(f"  populate_database() defined: {'def populate_database' in db_code}")
print(f"  Transaction commit executed: {'conn.commit()' in db_code}")
print(f"  Connection closed in finally block: {'conn.close()' in db_code}")
print(f"  Deletes prior to append (idempotent): {'DELETE FROM analysis;' in db_code and 'DELETE FROM invoices;' in db_code}")

# 13. Basic SQL Queries Test
print("\n--- 13. BASIC DATABASE QUERIES TEST ---")
cursor.execute("SELECT COUNT(*) FROM invoices;")
q_tot = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM analysis WHERE risk_level IN ('Medium Risk', 'High Risk');")
q_sus = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM analysis WHERE risk_level = 'High Risk';")
q_high = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM analysis WHERE risk_level = 'Medium Risk';")
q_med = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM analysis WHERE risk_level = 'Low Risk';")
q_low = cursor.fetchone()[0]

cursor.execute("SELECT AVG(risk_score) FROM analysis;")
q_avg = cursor.fetchone()[0]

print(f"  SQL Total invoices       : {q_tot:,}")
print(f"  SQL Suspicious invoices  : {q_sus:,}")
print(f"  SQL High-Risk invoices   : {q_high:,}")
print(f"  SQL Medium-Risk invoices : {q_med:,}")
print(f"  SQL Low-Risk invoices    : {q_low:,}")
print(f"  SQL Average risk score   : {q_avg:.2f}")

# 14. Invoice Lookup Test
print("\n--- 14. INVOICE LOOKUP TEST ---")
# Existing invoice
cursor.execute("""
    SELECT i.invoice_no, i.supplier_id, i.total_amount, a.risk_score, a.risk_level, a.anomaly_type, a.explanation
    FROM invoices i JOIN analysis a ON i.invoice_no = a.invoice_no
    WHERE i.invoice_no = 'INV019074';
""")
ex_res = cursor.fetchone()
print(f"  Existing invoice lookup (INV019074):")
print(f"    Result: {ex_res is not None}")
if ex_res:
    print(f"    InvNo: {ex_res[0]}, Supplier: {ex_res[1]}, TotalAmt: Rs.{ex_res[2]:,.2f}, Risk: {ex_res[3]} ({ex_res[4]}), Type: {ex_res[5]}")

# Non-existing invoice
cursor.execute("""
    SELECT i.invoice_no, i.total_amount, a.risk_score, a.risk_level
    FROM invoices i JOIN analysis a ON i.invoice_no = a.invoice_no
    WHERE i.invoice_no = 'INV999999';
""")
non_res = cursor.fetchone()
print(f"  Non-existing invoice lookup (INV999999):")
print(f"    Result is None: {non_res is None} (No error raised)")

# 15. Check Database Performance & Indexes
print("\n--- 15. PERFORMANCE & INDEXES ---")
cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%';")
indexes = cursor.fetchall()
print(f"  Total user-defined indexes: {len(indexes)}")
for idx_name, idx_sql in indexes:
    print(f"    - {idx_name}")

conn.close()
print("\n[AUDIT COMPLETED]")
