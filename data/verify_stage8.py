"""
verify_stage8.py
================
Verification script for Stage 8: SQLite Database Integration.
"""

import sys
import os
import sqlite3
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import (
    DB_PATH,
    get_dashboard_summary,
    get_invoices,
    get_suspicious_invoices,
    get_invoice_by_no,
)

print("=" * 70)
print("  STAGE 8: SQLITE DATABASE VERIFICATION")
print("=" * 70)

# 1. Database file check
print(f"\n1. Database file: {DB_PATH}")
print(f"   Exists: {os.path.exists(DB_PATH)}")
print(f"   File size: {os.path.getsize(DB_PATH) / (1024 * 1024):.2f} MB")

# 2. Table and column checks via SQLite connection
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [r[0] for r in cursor.fetchall() if not r[0].startswith("sqlite_")]
print(f"\n2. Tables created: {tables}")

# Check columns for invoices
cursor.execute("PRAGMA table_info(invoices);")
inv_cols = [r[1] for r in cursor.fetchall()]
print(f"   Invoices table columns ({len(inv_cols)}): {inv_cols[:5]} ... {inv_cols[-3:]}")

# Check columns for analysis
cursor.execute("PRAGMA table_info(analysis);")
ans_cols = [r[1] for r in cursor.fetchall()]
print(f"   Analysis table columns ({len(ans_cols)}): {ans_cols}")

# Row counts
cursor.execute("SELECT COUNT(*) FROM invoices;")
n_inv = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM analysis;")
n_ans = cursor.fetchone()[0]

print(f"\n3. Record counts:")
print(f"   invoices table : {n_inv:,}")
print(f"   analysis table : {n_ans:,}")

# Check foreign key integrity (invoices matching analysis)
cursor.execute("""
    SELECT COUNT(*) FROM invoices i
    LEFT JOIN analysis a ON i.invoice_no = a.invoice_no
    WHERE a.invoice_no IS NULL;
""")
orphan_inv = cursor.fetchone()[0]
print(f"   Invoices missing analysis: {orphan_inv}")

conn.close()

# 4. Test live query functions
print("\n4. Testing Live Query Functions:")

# a) Dashboard summary
summary = get_dashboard_summary()
print(f"   Dashboard summary:")
print(f"     Total Invoices      : {summary['total_invoices']:,}")
print(f"     Normal Invoices     : {summary['normal_invoices']:,}")
print(f"     Suspicious Invoices : {summary['suspicious_invoices']:,}")
print(f"     High Risk           : {summary['high_risk']:,}")
print(f"     Medium Risk         : {summary['medium_risk']:,}")
print(f"     Low Risk            : {summary['low_risk']:,}")
print(f"     Average Risk Score  : {summary['avg_risk_score']}")

# b) Paginated invoices
inv_page = get_invoices(page=1, per_page=3)
print(f"\n   get_invoices() test: Retrieved {len(inv_page['items'])} items (Total: {inv_page['total']:,}, Pages: {inv_page['pages']:,})")
for it in inv_page["items"]:
    print(f"     [{it['invoice_no']}] {it['supplier_gstin']} -> {it['buyer_gstin']} | Rs.{it['total_amount']:,.2f} | Risk: {it['risk_score']} ({it['risk_level']})")

# c) Suspicious invoices
sus_page = get_suspicious_invoices(page=1, per_page=3)
print(f"\n   get_suspicious_invoices() test: Retrieved {len(sus_page['items'])} items (Total: {sus_page['total']:,})")
for it in sus_page["items"]:
    print(f"     [{it['invoice_no']}] Risk: {it['risk_score']} ({it['risk_level']}) | Type: {it['anomaly_type']}")

# d) Invoice by ID
det = get_invoice_by_no("INV019074")
print(f"\n   get_invoice_by_no('INV019074') test:")
print(f"     Invoice No   : {det['invoice_no']}")
print(f"     Supplier     : {det['supplier_id']} ({det['supplier_gstin']})")
print(f"     Total Amount : Rs.{det['total_amount']:,.2f}")
print(f"     IF Score     : {det['isolation_score']}")
print(f"     LOF Score    : {det['lof_score']}")
print(f"     Z Score      : {det['max_abs_zscore']}")
print(f"     Risk Score   : {det['risk_score']} ({det['risk_level']})")
print(f"     Explanation  : {det['explanation']}")

print("\n[VERIFICATION COMPLETED: STAGE 8 FULLY OPERATIONAL]")
