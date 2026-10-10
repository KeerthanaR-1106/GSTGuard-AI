"""
test_stage9_apis.py
===================
Comprehensive test suite for Stage 9 Flask REST APIs:
  - GET /api/dashboard
  - GET /api/invoices
  - GET /api/invoices/<invoice_no> (existing & non-existing)
  - GET /api/suspicious-invoices
  - GET /api/model-performance
  - POST /api/upload (valid CSV, missing columns, empty upload)
"""

import os
import sys
import io
import json
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

print("=" * 70)
print("  STAGE 9: FLASK REST API ENDPOINTS TEST SUITE")
print("=" * 70)

app = create_app()
client = app.test_client()

# ── 1. GET /api/dashboard ─────────────────────────────────────────────────────
print("\n[TEST 1] GET /api/dashboard ...")
res = client.get("/api/dashboard")
print(f"  Status code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
assert data["status"] == "success"
stats = data["data"]
print(f"  Total Invoices     : {stats['total_invoices']:,}")
print(f"  Normal Invoices    : {stats['normal_invoices']:,}")
print(f"  Suspicious Invoices: {stats['suspicious_invoices']:,}")
print(f"  High Risk Invoices : {stats['high_risk']:,}")
print(f"  Medium Risk        : {stats['medium_risk']:,}")
print(f"  Low Risk           : {stats['low_risk']:,}")
print(f"  Avg Risk Score     : {stats['avg_risk_score']}")
assert stats["total_invoices"] == 20000
assert stats["suspicious_invoices"] == stats["medium_risk"] + stats["high_risk"]
print("  -> PASS")

# ── 2. GET /api/invoices ──────────────────────────────────────────────────────
print("\n[TEST 2] GET /api/invoices ...")
res = client.get("/api/invoices?page=1&per_page=10")
print(f"  Status code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200
assert len(data["data"]) == 10
assert data["pagination"]["total"] == 20000
first_inv = data["data"][0]
print(f"  First invoice: {first_inv['invoice_no']} | Risk: {first_inv['risk_score']} ({first_inv['risk_level']})")
print("  -> PASS")

# ── 3. GET /api/invoices with filter ──────────────────────────────────────────
print("\n[TEST 3] GET /api/invoices?risk_level=High%20Risk ...")
res = client.get("/api/invoices?risk_level=High%20Risk&per_page=5")
data = res.get_json()
assert res.status_code == 200
assert data["pagination"]["total"] == stats["high_risk"]
for item in data["data"]:
    assert item["risk_level"] == "High Risk"
    assert item["risk_score"] >= 70.0
print(f"  Filtered items count: {len(data['data'])} (Total High Risk: {data['pagination']['total']})")
print("  -> PASS")

# ── 4. GET /api/invoices/<invoice_no> (Existing) ──────────────────────────────
print("\n[TEST 4] GET /api/invoices/INV019074 (Existing) ...")
res = client.get("/api/invoices/INV019074")
print(f"  Status code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200
inv = data["data"]
print(f"  Invoice: {inv['invoice_no']}")
print(f"  Supplier: {inv['supplier_id']} ({inv['supplier_gstin']})")
print(f"  Total Amount: Rs.{inv['total_amount']:,.2f}")
print(f"  IF Score: {inv['isolation_score']}, LOF: {inv['lof_score']}, Max Z: {inv['max_abs_zscore']}")
print(f"  Risk Score: {inv['risk_score']} ({inv['risk_level']})")
print(f"  Explanation: {inv['explanation'][:80]}...")
assert inv["invoice_no"] == "INV019074"
assert "isolation_score" in inv
assert "lof_score" in inv
assert "max_abs_zscore" in inv
assert "explanation" in inv
print("  -> PASS")

# ── 5. GET /api/invoices/<invoice_no> (Non-existing) ──────────────────────────
print("\n[TEST 5] GET /api/invoices/INV999999 (Non-existing) ...")
res = client.get("/api/invoices/INV999999")
print(f"  Status code: {res.status_code}")
assert res.status_code == 404
data = res.get_json()
assert data["status"] == "error"
print(f"  Error message: {data['message']}")
print("  -> PASS")

# ── 6. GET /api/suspicious-invoices ───────────────────────────────────────────
print("\n[TEST 6] GET /api/suspicious-invoices ...")
res = client.get("/api/suspicious-invoices?page=1&per_page=15")
print(f"  Status code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200
assert data["pagination"]["total"] == stats["suspicious_invoices"]
for item in data["data"]:
    assert item["risk_level"] in ["Medium Risk", "High Risk"]
print(f"  Total suspicious returned: {data['pagination']['total']} (Retrieved: {len(data['data'])})")
print("  -> PASS")

# ── 7. GET /api/model-performance ─────────────────────────────────────────────
print("\n[TEST 7] GET /api/model-performance ...")
res = client.get("/api/model-performance")
print(f"  Status code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200
assert data["metrics_available"] is False
assert data["data"] is None
assert "in-sample" in data["message"]
print(f"  Metrics availability: {data['metrics_available']}")
print(f"  Explanation: {data['message']}")
print("  -> PASS")

# ── 8. POST /api/upload - Empty / Missing File ────────────────────────────────
print("\n[TEST 8] POST /api/upload (Empty / Missing file) ...")
res = client.post("/api/upload")
assert res.status_code == 400
data = res.get_json()
print(f"  Response: {data['message']}")
print("  -> PASS")

# ── 9. POST /api/upload - Invalid File with Missing Columns ───────────────────
print("\n[TEST 9] POST /api/upload (Invalid CSV missing columns) ...")
bad_csv = io.BytesIO(
    b"invoice_no,total_amount\n"
    b"INV999001,5000\n"
    b"INV999002,6000\n"
)
res = client.post(
    "/api/upload",
    data={"file": (bad_csv, "invalid_sample.csv")},
    content_type="multipart/form-data"
)
print(f"  Status code: {res.status_code}")
assert res.status_code == 400
data = res.get_json()
print(f"  Validation error caught: {data['message']}")
assert "Missing required columns" in data["message"]
print("  -> PASS")

# ── 10. POST /api/upload - Valid CSV End-to-End Pipeline ─────────────────────
print("\n[TEST 10] POST /api/upload (Valid batch of 10 test invoices) ...")
raw_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gst_invoices.csv"))
sample_test_df = raw_df.head(10).copy()
sample_test_df["invoice_no"] = [f"TEST_INV_{i:04d}" for i in range(1, 11)]

csv_buffer = io.BytesIO()
sample_test_df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

res = client.post(
    "/api/upload",
    data={"file": (csv_buffer, "batch_test_upload.csv")},
    content_type="multipart/form-data"
)
print(f"  Status code: {res.status_code}")
assert res.status_code == 200
data = res.get_json()
assert data["status"] == "success"
print(f"  Upload response message : {data['message']}")
print(f"  Processed record count  : {data['processed_count']}")
print(f"  Summary generated       : {data['summary']}")

# Verify that the new test invoice exists in the SQLite database
res_verify = client.get("/api/invoices/TEST_INV_0001")
assert res_verify.status_code == 200
test_item = res_verify.get_json()["data"]
print(f"  Database verification: TEST_INV_0001 fetched with Risk Score: {test_item['risk_score']} ({test_item['risk_level']})")
print("  -> PASS")

print("\n" + "=" * 70)
print("  ALL 10 API TEST CASES PASSED SUCCESSFULLY!")
print("=" * 70)
