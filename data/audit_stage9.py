"""
audit_stage9.py
===============
Read-only comprehensive audit script for Stage 9: Flask Backend + REST API.
Does NOT modify any file, schema, or data.
"""

import os
import sys
import sqlite3
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import create_app
from config import Config
from database.database import DB_PATH

print("=" * 70)
print("  STAGE 9 COMPREHENSIVE FLASK BACKEND AUDIT")
print("=" * 70)

# 1. Structure check
print("\n--- 1. BACKEND FILES INSPECTION ---")
backend_files = [
    ("Application Entry Point", "app.py"),
    ("Configuration Module", "config.py"),
    ("Requirements Specification", "requirements.txt"),
    ("Dashboard & Summary Routes", "routes/dashboard_routes.py"),
    ("Invoice & Search Routes", "routes/invoice_routes.py"),
    ("Report & Performance Routes", "routes/report_routes.py"),
    ("Upload & Pipeline Routes", "routes/upload_routes.py"),
    ("Database Engine & Helpers", "database/database.py"),
    ("SQLAlchemy ORM Models", "database/models.py"),
    ("SQLite Database File", "database/gst_fraud.db")
]

for label, rel_path in backend_files:
    full_path = os.path.join(ROOT_DIR, rel_path)
    exists = os.path.exists(full_path)
    size = f"{os.path.getsize(full_path)/1024:.1f} KB" if exists else "N/A"
    print(f"  {label:<30} | {rel_path:<28} | exists={exists} | size={size}")

# 2. Database Record Counts BEFORE API Testing
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM invoices;")
count_inv_before = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM analysis;")
count_ans_before = c.fetchone()[0]
conn.close()
print(f"\n--- 2. INITIAL DATABASE STATE ---")
print(f"  Invoices count before API tests: {count_inv_before:,}")
print(f"  Analysis count before API tests: {count_ans_before:,}")

# 3. Initialize Flask App
print("\n--- 3. FLASK APP INITIALIZATION ---")
try:
    app = create_app()
    client = app.test_client()
    print("  Flask app created successfully via factory: True")
    print(f"  App name: {app.name}")
    print(f"  Config loaded: SECRET_KEY={bool(app.config.get('SECRET_KEY'))}, DB_URI={app.config.get('DATABASE_URI')}")
    rules = [rule.rule for rule in app.url_map.iter_rules()]
    print(f"  Registered routes count: {len(rules)}")
except Exception as e:
    print(f"  Failed to create Flask app: {e}")
    sys.exit(1)

# 4. Comprehensive Endpoint Tests
print("\n--- 4. ENDPOINT VERIFICATION TESTS ---")
test_results = []

# Test A: GET /api/dashboard
res = client.get("/api/dashboard")
data = res.get_json() if res.is_json else None
test_results.append({
    "endpoint": "/api/dashboard",
    "method": "GET",
    "test": "Live dashboard statistics",
    "expected_code": 200,
    "actual_code": res.status_code,
    "is_json": res.is_json,
    "valid": res.status_code == 200 and data and data.get("data", {}).get("total_invoices") == 20000
})
print(f"  GET /api/dashboard: Status={res.status_code}, JSON={res.is_json}")
if data:
    d = data.get("data", {})
    print(f"    Total: {d.get('total_invoices')}, Normal: {d.get('normal_invoices')}, Suspicious: {d.get('suspicious_invoices')}, AvgScore: {d.get('avg_risk_score')}")

# Test B: GET /api/invoices (default pagination)
res = client.get("/api/invoices?page=1&per_page=20")
data = res.get_json() if res.is_json else None
test_results.append({
    "endpoint": "/api/invoices",
    "method": "GET",
    "test": "Paginated invoice list",
    "expected_code": 200,
    "actual_code": res.status_code,
    "is_json": res.is_json,
    "valid": res.status_code == 200 and data and len(data.get("data", [])) == 20 and data.get("pagination", {}).get("total") == 20000
})
print(f"  GET /api/invoices: Status={res.status_code}, Items={len(data.get('data', [])) if data else 0}, Total={data.get('pagination', {}).get('total') if data else 0}")

# Test C: GET /api/invoices with search matching
res = client.get("/api/invoices?search=INV019074")
data = res.get_json() if res.is_json else None
match_found = len(data.get("data", [])) == 1 and data["data"][0]["invoice_no"] == "INV019074"
test_results.append({
    "endpoint": "/api/invoices?search=INV019074",
    "method": "GET",
    "test": "Search matching invoice",
    "expected_code": 200,
    "actual_code": res.status_code,
    "is_json": res.is_json,
    "valid": res.status_code == 200 and match_found
})
print(f"  GET /api/invoices?search=INV019074: Status={res.status_code}, Matched={match_found}")

# Test D: GET /api/invoices with search NO MATCH
res = client.get("/api/invoices?search=DOESNOTEXIST999")
data = res.get_json() if res.is_json else None
no_match_ok = len(data.get("data", [])) == 0 and data.get("pagination", {}).get("total") == 0
test_results.append({
    "endpoint": "/api/invoices?search=DOESNOTEXIST999",
    "method": "GET",
    "test": "Search with no matching records",
    "expected_code": 200,
    "actual_code": res.status_code,
    "is_json": res.is_json,
    "valid": res.status_code == 200 and no_match_ok
})
print(f"  GET /api/invoices?search=DOESNOTEXIST999: Status={res.status_code}, No-match handled safely={no_match_ok}")

# Test E: GET /api/invoices with risk_level filter
res = client.get("/api/invoices?risk_level=High%20Risk&per_page=10")
data = res.get_json() if res.is_json else None
high_risk_ok = data.get("pagination", {}).get("total") == 144
test_results.append({
    "endpoint": "/api/invoices?risk_level=High%20Risk",
    "method": "GET",
    "test": "Filter by risk_level",
    "expected_code": 200,
    "actual_code": res.status_code,
    "is_json": res.is_json,
    "valid": res.status_code == 200 and high_risk_ok
})
print(f"  GET /api/invoices?risk_level=High Risk: Status={res.status_code}, Total High Risk={data.get('pagination', {}).get('total') if data else 0}")

# Test F: GET /api/invoices/<invoice_no> (Existing)
res = client.get("/api/invoices/INV019074")
data = res.get_json() if res.is_json else None
inv_ok = False
if res.status_code == 200 and data and data.get("data"):
    inv = data["data"]
    inv_ok = (
        inv.get("invoice_no") == "INV019074" and
        "isolation_score" in inv and
        "lof_score" in inv and
        "max_abs_zscore" in inv and
        "risk_score" in inv and
        "explanation" in inv
    )
test_results.append({
    "endpoint": "/api/invoices/INV019074",
    "method": "GET",
    "test": "Single existing invoice lookup",
    "expected_code": 200,
    "actual_code": res.status_code,
    "is_json": res.is_json,
    "valid": inv_ok
})
print(f"  GET /api/invoices/INV019074: Status={res.status_code}, Contains model scores and explanation={inv_ok}")

# Test G: GET /api/invoices/<invoice_no> (Non-existing)
res = client.get("/api/invoices/INV999999")
data = res.get_json() if res.is_json else None
not_found_ok = res.status_code == 404 and data and data.get("status") == "error"
test_results.append({
    "endpoint": "/api/invoices/INV999999",
    "method": "GET",
    "test": "Single non-existing invoice lookup",
    "expected_code": 404,
    "actual_code": res.status_code,
    "is_json": res.is_json,
    "valid": not_found_ok
})
print(f"  GET /api/invoices/INV999999: Status={res.status_code}, Clean 404 JSON returned={not_found_ok}")

# Test H: GET /api/suspicious-invoices
res = client.get("/api/suspicious-invoices?page=1&per_page=10")
data = res.get_json() if res.is_json else None
sus_total = data.get("pagination", {}).get("total") if data else 0
sus_ok = res.status_code == 200 and sus_total == 886
test_results.append({
    "endpoint": "/api/suspicious-invoices",
    "method": "GET",
    "test": "Suspicious invoice filtering (Med+High)",
    "expected_code": 200,
    "actual_code": res.status_code,
    "is_json": res.is_json,
    "valid": sus_ok
})
print(f"  GET /api/suspicious-invoices: Status={res.status_code}, Total suspicious={sus_total} (Expected: 886)")

# Test I: GET /api/model-performance
res = client.get("/api/model-performance")
data = res.get_json() if res.is_json else None
perf_ok = (
    res.status_code == 200 and
    data and
    "Isolation Forest" in data.get("data", {}).get("model_metrics", {}) and
    "Local Outlier Factor (LOF)" in data.get("data", {}).get("model_metrics", {}) and
    "Statistical Z-Score" in data.get("data", {}).get("model_metrics", {})
)
test_results.append({
    "endpoint": "/api/model-performance",
    "method": "GET",
    "test": "Model performance metrics",
    "expected_code": 200,
    "actual_code": res.status_code,
    "is_json": res.is_json,
    "valid": perf_ok
})
print(f"  GET /api/model-performance: Status={res.status_code}, All 3 models included={perf_ok}")

# Test J: GET /api/export/csv
res = client.get("/api/export/csv?risk_level=High%20Risk")
csv_content = res.data.decode("utf-8") if res.status_code == 200 else ""
export_ok = res.status_code == 200 and res.mimetype == "text/csv" and "Invoice No" in csv_content
test_results.append({
    "endpoint": "/api/export/csv",
    "method": "GET",
    "test": "CSV compliance report export",
    "expected_code": 200,
    "actual_code": res.status_code,
    "is_json": False,
    "valid": export_ok
})
print(f"  GET /api/export/csv: Status={res.status_code}, Mime={res.mimetype}, CSV headers valid={export_ok}")

# 5. Database Record Counts AFTER API Testing (Integrity Check)
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM invoices;")
count_inv_after = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM analysis;")
count_ans_after = c.fetchone()[0]
conn.close()
print(f"\n--- 5. DATABASE INTEGRITY POST-TEST AUDIT ---")
print(f"  Invoices count after API tests: {count_inv_after:,} (Before: {count_inv_before:,})")
print(f"  Analysis count after API tests: {count_ans_after:,} (Before: {count_ans_before:,})")
print(f"  Read-only API calls did not mutate database: {count_inv_after == count_inv_before == 20000 and count_ans_after == count_ans_before == 20000}")

# 6. Check ML Pipeline Isolation & Ground Truth Isolation
print("\n--- 6. ISOLATION AUDIT ---")
with open(os.path.join(ROOT_DIR, "routes", "dashboard_routes.py"), "r", encoding="utf-8") as f:
    code_dash = f.read()
with open(os.path.join(ROOT_DIR, "routes", "invoice_routes.py"), "r", encoding="utf-8") as f:
    code_inv = f.read()

read_apis_gt_free = ("ground_truth.csv" not in code_dash) and ("ground_truth.csv" not in code_inv)
read_apis_ml_free = ("fit_predict" not in code_dash) and ("fit_predict" not in code_inv)
print(f"  Read API routes do NOT load ground_truth.csv: {read_apis_gt_free}")
print(f"  Read API routes do NOT trigger ML model retraining: {read_apis_ml_free}")

# 7. Check Query Safety & SQL Injection Prevention
print("\n--- 7. QUERY SAFETY & PARAMETERIZATION ---")
# Attempt SQL injection in search parameter
injection_payload = "' OR 1=1 --"
res_inj = client.get(f"/api/invoices?search={injection_payload}")
data_inj = res_inj.get_json() if res_inj.is_json else {}
inj_total = data_inj.get("pagination", {}).get("total", 0)
print(f"  SQL injection attempt (' OR 1=1 --): returned {inj_total} records (Safe: total is NOT 20,000)")
sql_injection_safe = (inj_total != 20000 and res_inj.status_code == 200)
print(f"  Parameterized query SQL injection safe: {sql_injection_safe}")

print("\n[STAGE 9 AUDIT COMPLETED]")
