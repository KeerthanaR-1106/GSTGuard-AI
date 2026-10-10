"""
data/test_stage10_frontend.py
=============================
Integration test suite for Stage 10: Frontend Dashboard UI.
Verifies all templates, routes, static assets, and context variables.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import create_app
from database.database import SessionLocal, Invoice, Analysis, get_invoice_by_no


def run_tests():
    print("=" * 70)
    print("  STAGE 10: FRONTEND DASHBOARD & UI VERIFICATION")
    print("=" * 70)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    passes = 0
    failures = 0

    def assert_test(name, condition, detail=""):
        nonlocal passes, failures
        if condition:
            passes += 1
            print(f"  [PASS] {name}")
        else:
            failures += 1
            print(f"  [FAIL] {name}: {detail}")

    # 1. Static Assets Verification
    print("\n[1] Verifying Static Files Exist on Disk:")
    css_path = os.path.join(BASE_DIR, "static", "css", "style.css")
    js_dash_path = os.path.join(BASE_DIR, "static", "js", "dashboard.js")
    js_charts_path = os.path.join(BASE_DIR, "static", "js", "charts.js")

    assert_test("static/css/style.css exists", os.path.exists(css_path) and os.path.getsize(css_path) > 100)
    assert_test("static/js/dashboard.js exists", os.path.exists(js_dash_path) and os.path.getsize(js_dash_path) > 100)
    assert_test("static/js/charts.js exists", os.path.exists(js_charts_path) and os.path.getsize(js_charts_path) > 100)

    # 2. Template Files Existence
    print("\n[2] Verifying Template Files Exist on Disk:")
    templates = [
        "base.html", "login.html", "dashboard.html", "invoices.html",
        "suspicious.html", "invoice_details.html", "export_reports.html",
        "upload.html", "404.html", "500.html"
    ]
    for t in templates:
        t_path = os.path.join(BASE_DIR, "templates", t)
        assert_test(f"templates/{t} exists", os.path.exists(t_path) and os.path.getsize(t_path) > 50)

    # 3. Route & View Rendering Tests
    print("\n[3] Testing Frontend Template Rendering via Flask Test Client:")

    # GET / -> 302 Redirect
    resp = client.get("/")
    assert_test("GET / returns 302 Redirect", resp.status_code == 302)

    # GET /login -> 200 OK
    resp = client.get("/login")
    assert_test("GET /login returns 200 OK", resp.status_code == 200)
    html = resp.data.decode("utf-8", errors="ignore")
    assert_test("login.html does not disclose demo credentials", "admin123" not in html and "Demo Credentials" not in html)

    # POST /login (Success)
    resp = client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=True)
    assert_test("POST /login with valid credentials redirects to dashboard", resp.status_code == 200)
    html = resp.data.decode("utf-8", errors="ignore")
    assert_test("Dashboard rendered after login", "Executive Dashboard" in html or "Dashboard" in html)

    # GET /dashboard -> 200 OK
    resp = client.get("/dashboard")
    assert_test("GET /dashboard returns 200 OK", resp.status_code == 200)
    html = resp.data.decode("utf-8", errors="ignore")
    assert_test("dashboard.html includes live stat cards", "Total Invoices" in html and "High Risk" in html)
    assert_test("dashboard.html includes canvas elements for charts", "<canvas" in html)
    assert_test("Low Risk and Normal Invoices use matching shield-check icons", html.count('class="bi bi-shield-check"') >= 2)
    assert_test("dashboard anomaly breakdown separates model outliers", "Statistical Model Outlier" in html)
    assert_test("dashboard.html omits the Quick Demo Tour banner", "Quick Demo Tour" not in html and "dash-hero" not in html)

    # GET /invoices -> 200 OK
    resp = client.get("/invoices")
    assert_test("GET /invoices returns 200 OK", resp.status_code == 200)
    html = resp.data.decode("utf-8", errors="ignore")
    assert_test("invoices.html contains invoice table", "<table" in html and "Invoice No" in html)
    assert_test("invoices.html contains pagination controls", "pagination" in html)

    session = SessionLocal()
    high_risk_normal = session.query(Invoice.invoice_no).join(
        Analysis, Invoice.invoice_no == Analysis.invoice_no
    ).filter(Analysis.risk_level == "High Risk", Analysis.anomaly_type == "Normal").first()
    session.close()
    if high_risk_normal:
        corrected_invoice = get_invoice_by_no(high_risk_normal[0])
        assert_test(
            "high-risk invoices with no rule match show a model-outlier pattern",
            corrected_invoice["anomaly_type"] == "Statistical Model Outlier"
        )
    else:
        assert_test("no stored high-risk invoices are mislabeled Normal", True)

    resp = client.get("/invoices?anomaly_type=Statistical%20Model%20Outlier")
    assert_test(
        "model-outlier anomaly filter returns matching invoices",
        resp.status_code == 200 and "Statistical Model Outlier" in resp.data.decode("utf-8", errors="ignore")
    )

    # GET /invoices with filters
    resp = client.get("/invoices?risk_level=High%20Risk")
    assert_test("GET /invoices?risk_level=High%20Risk returns 200 OK", resp.status_code == 200)
    html = resp.data.decode("utf-8", errors="ignore")
    assert_test("Filtered invoices page contains records", "High Risk" in html)

    # GET /suspicious -> 200 OK
    resp = client.get("/suspicious")
    assert_test("GET /suspicious returns 200 OK", resp.status_code == 200)
    html = resp.data.decode("utf-8", errors="ignore")
    assert_test("suspicious.html contains compliance review warning", "Compliance" in html or "Suspicious" in html)

    # GET /invoice/<invoice_no> -> 200 OK
    session = SessionLocal()
    sample_inv = session.query(Invoice.invoice_no).first()
    session.close()

    if sample_inv:
        inv_no = sample_inv[0]
        resp = client.get(f"/invoice/{inv_no}")
        assert_test(f"GET /invoice/{inv_no} returns 200 OK", resp.status_code == 200)
        html = resp.data.decode("utf-8", errors="ignore")
        assert_test("invoice_details.html contains Isolation Forest score", "Isolation Forest" in html)
        assert_test("invoice_details.html contains LOF score", "Local Outlier Factor" in html)
        assert_test("invoice_details.html contains Z-Score", "Z-Score" in html)
        assert_test("invoice_details.html contains Risk Assessment", "Risk" in html)
        assert_test("invoice anomaly badge uses visible translucent styling", "risk-pattern-badge" in html and "rgba(255, 255, 255, 0.16)" in html)
    else:
        assert_test("Sample invoice exists in DB", False, "No invoices found in database")

    # GET /reports -> 200 OK
    resp = client.get("/reports")
    assert_test("GET /reports returns 200 OK", resp.status_code == 200)
    html = resp.data.decode("utf-8", errors="ignore")
    assert_test("reports page is now Export Reports", "Export Reports" in html and "Model Performance" not in html)
    assert_test("export page offers all invoice CSVs", "/api/export/csv" in html)
    assert_test("export page offers each risk-tier CSV", all(f"risk_level={level}%20Risk" in html for level in ["High", "Medium", "Low"]))

    # GET /upload -> 200 OK
    resp = client.get("/upload")
    assert_test("GET /upload returns 200 OK", resp.status_code == 200)
    html = resp.data.decode("utf-8", errors="ignore")
    assert_test("upload.html contains drag-drop zone", "drop-zone" in html or "Drag" in html)

    # Removed informational page
    resp = client.get("/about")
    assert_test("GET /about returns 404 NOT FOUND", resp.status_code == 404)
    assert_test("Sidebar no longer links to /about", "/about" not in client.get("/dashboard").data.decode("utf-8", errors="ignore"))

    # GET 404 handler
    resp = client.get("/non-existent-page-url")
    assert_test("GET invalid page returns 404 NOT FOUND", resp.status_code == 404)
    html = resp.data.decode("utf-8", errors="ignore")
    assert_test("404.html template rendered", "404" in html)

    # POST /api/upload integration test
    print("\n[3b] Testing API Upload handling via client:")
    import io
    import pandas as pd
    df_sample = pd.read_csv(os.path.join(BASE_DIR, "data", "gst_invoices.csv")).head(20)
    df_sample["invoice_no"] = ["STAGE10_SAMPLE_" + str(index) for index in range(len(df_sample))]
    csv_bytes = io.BytesIO()
    df_sample.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    resp = client.post(
        "/api/upload",
        data={"file": (csv_bytes, "test_upload_sample.csv")},
        content_type="multipart/form-data"
    )
    assert_test("POST /api/upload returns 200 OK", resp.status_code == 200)
    data = resp.get_json()
    assert_test("Upload response status is success", data.get("status") == "success")
    assert_test("Upload response contains processed_count", data.get("processed_count") == 20)
    assert_test("Upload response contains summary dict with metrics", "summary" in data and "high_risk" in data["summary"])

    # 4. Static URL Serving Tests
    print("\n[4] Testing Static Asset URL Serving via Flask:")
    for asset in ["/static/css/style.css", "/static/js/dashboard.js", "/static/js/charts.js"]:
        resp = client.get(asset)
        assert_test(f"GET {asset} returns 200 OK", resp.status_code == 200 and len(resp.data) > 50)

    print("\n" + "=" * 70)
    print(f"  STAGE 10 TEST SUMMARY: {passes} PASSED, {failures} FAILED")
    print("=" * 70)

    return failures == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
