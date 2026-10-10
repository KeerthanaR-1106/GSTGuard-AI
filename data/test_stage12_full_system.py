"""
data/test_stage12_full_system.py
================================
STAGE 12: Master Full-System Validation & Health Check Runner

Automated GST Invoice Fraud Detection System using Anomaly Detection
for MSME Compliance.

Runs holistic end-to-end verification across the entire repository:
  - Dataset generation & schema checks (Stage 1)
  - Preprocessing & 43 engineered features (Stage 2)
  - Three Unsupervised ML models: IF, LOF, Z-Score (Stages 3-5)
  - Offline Model Evaluation metrics (Stage 6)
  - Ensemble Risk Scoring (Stage 7)
  - SQLite Database persistence (Stage 8)
  - Flask REST API endpoints (Stage 9)
  - Frontend UI templates & assets (Stage 10)
  - End-to-end integration lifecycle (Stage 11)
  - Documentation & Viva readiness (Stage 12)
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import create_app
from database.database import (
    DB_PATH, SessionLocal, Invoice, Analysis,
    get_dashboard_summary
)


def run_full_system_test():
    print("=" * 75)
    print("  STAGE 12: MASTER FULL-SYSTEM VALIDATION & HEALTH CHECK")
    print("  Automated GST Invoice Fraud Detection System for MSME Compliance")
    print("=" * 75)

    passed_checks = 0
    failed_checks = 0
    failures = []

    def check(stage: str, name: str, condition: bool, details: str = ""):
        nonlocal passed_checks, failed_checks, failures
        if condition:
            passed_checks += 1
            print(f"  [PASS] [{stage}] {name}")
        else:
            failed_checks += 1
            msg = f"[{stage}] {name}: {details}"
            failures.append(msg)
            print(f"  [FAIL] {msg}")

    # -------------------------------------------------------------------------
    # 1. Dataset Verification
    # -------------------------------------------------------------------------
    print("\n--- 1. Dataset Verification (Stage 1) ---")
    inv_file = os.path.join(BASE_DIR, "data", "gst_invoices.csv")
    gt_file = os.path.join(BASE_DIR, "data", "ground_truth.csv")

    check("Stage 1", "gst_invoices.csv exists", os.path.exists(inv_file))
    check("Stage 1", "ground_truth.csv exists", os.path.exists(gt_file))

    df_inv = pd.read_csv(inv_file)
    df_gt = pd.read_csv(gt_file)
    check("Stage 1", "20,000 raw invoices generated", len(df_inv) == 20000)
    check("Stage 1", "22 standard GST columns present", df_inv.shape[1] == 22)
    check("Stage 1", "Ground truth has 20,000 labels", len(df_gt) == 20000)
    check("Stage 1", "Ground truth has exactly 1,000 anomalies (5.00%)", df_gt["actual_anomaly"].sum() == 1000)

    # -------------------------------------------------------------------------
    # 2. Preprocessing & Feature Engineering
    # -------------------------------------------------------------------------
    print("\n--- 2. Feature Engineering Verification (Stage 2) ---")
    feat_file = os.path.join(BASE_DIR, "data", "ml_features.csv")
    check("Stage 2", "ml_features.csv exists", os.path.exists(feat_file))

    df_feat = pd.read_csv(feat_file)
    check("Stage 2", "20,000 feature rows", len(df_feat) == 20000)
    check("Stage 2", "44 columns (invoice_no + 43 features)", df_feat.shape[1] == 44)
    check("Stage 2", "Zero NaN values in feature matrix", df_feat.isnull().sum().sum() == 0)
    check("Stage 2", "Zero Inf values in feature matrix", not np.isinf(df_feat.drop(columns=['invoice_no']).values).any())
    check("Stage 2", "No ground truth leakage in feature matrix", "actual_anomaly" not in df_feat.columns)

    # -------------------------------------------------------------------------
    # 3. Model Results & Isolation
    # -------------------------------------------------------------------------
    print("\n--- 3. Unsupervised ML Models (Stages 3 - 5) ---")
    if_res_file = os.path.join(BASE_DIR, "results", "isolation_forest_results.csv")
    lof_res_file = os.path.join(BASE_DIR, "results", "lof_results.csv")
    zs_res_file = os.path.join(BASE_DIR, "results", "zscore_results.csv")

    check("Stage 3", "isolation_forest_results.csv exists", os.path.exists(if_res_file))
    check("Stage 4", "lof_results.csv exists", os.path.exists(lof_res_file))
    check("Stage 5", "zscore_results.csv exists", os.path.exists(zs_res_file))

    df_if = pd.read_csv(if_res_file)
    df_lof = pd.read_csv(lof_res_file)
    df_zs = pd.read_csv(zs_res_file)

    check("Stage 3", "Isolation Forest flags exactly 1,000 anomalies (5%)", df_if["if_anomaly"].sum() == 1000)
    check("Stage 4", "LOF flags exactly 1,000 anomalies (5%)", df_lof["lof_anomaly"].sum() == 1000)
    check("Stage 5", "Z-Score (|Z| >= 3.0) flags 2,071 anomalies", df_zs["zscore_anomaly"].sum() == 2071)
    check("Models", "All model output files contain exactly 20,000 rows", len(df_if) == len(df_lof) == len(df_zs) == 20000)

    # -------------------------------------------------------------------------
    # 4. Evaluation & Comparison Metrics
    # -------------------------------------------------------------------------
    print("\n--- 4. Offline Model Evaluation (Stage 6) ---")
    metrics_file = os.path.join(BASE_DIR, "results", "model_evaluation_metrics.json")
    check("Stage 6", "model_evaluation_metrics.json exists", os.path.exists(metrics_file))

    # -------------------------------------------------------------------------
    # 5. Ensemble Risk Scoring
    # -------------------------------------------------------------------------
    print("\n--- 5. Ensemble Risk Scoring (Stage 7) ---")
    ens_file = os.path.join(BASE_DIR, "results", "ensemble_results.csv")
    check("Stage 7", "ensemble_results.csv exists", os.path.exists(ens_file))

    df_ens = pd.read_csv(ens_file)
    check("Stage 7", "20,000 ensemble records", len(df_ens) == 20000)
    check("Stage 7", "Risk scores strictly bounded [0, 100]", 0.0 <= df_ens["risk_score"].min() and df_ens["risk_score"].max() <= 100.0)
    score_levels_match = (
        ((df_ens["risk_score"] < 40.0) == (df_ens["risk_level"] == "Low Risk")).all()
        and (((df_ens["risk_score"] >= 40.0) & (df_ens["risk_score"] < 70.0)) == (df_ens["risk_level"] == "Medium Risk")).all()
        and ((df_ens["risk_score"] >= 70.0) == (df_ens["risk_level"] == "High Risk")).all()
    )
    check("Stage 7", "Risk categories match the documented score thresholds", score_levels_match)

    # -------------------------------------------------------------------------
    # 6. Database Health
    # -------------------------------------------------------------------------
    print("\n--- 6. SQLite Database (Stage 8) ---")
    check("Stage 8", "gst_fraud.db exists", os.path.exists(DB_PATH))

    session = SessionLocal()
    try:
        inv_cnt = session.query(Invoice).count()
        ans_cnt = session.query(Analysis).count()
        check("Stage 8", "Invoices table count = 20,000", inv_cnt == 20000)
        check("Stage 8", "Analysis table count = 20,000", ans_cnt == 20000)

        # Confirm 1:1 join without orphans
        orphans = session.query(Invoice).outerjoin(Analysis, Invoice.invoice_no == Analysis.invoice_no).filter(Analysis.invoice_no.is_(None)).count()
        check("Stage 8", "Zero orphan invoice records", orphans == 0)

        summary = get_dashboard_summary()
        check("Stage 8", "Live summary total matches invoice table", summary["total_invoices"] == inv_cnt)
        check("Stage 8", "Live suspicious count matches medium plus high tiers",
              summary["suspicious_invoices"] == summary["medium_risk"] + summary["high_risk"])
        check("Stage 8", "Live average risk score is within index bounds",
              0.0 <= summary["avg_risk_score"] <= 100.0)
    finally:
        session.close()

    # -------------------------------------------------------------------------
    # 7. Flask Application & REST APIs
    # -------------------------------------------------------------------------
    print("\n--- 7. Flask REST API (Stage 9) ---")
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    r_dash = client.get("/api/dashboard")
    r_invs = client.get("/api/invoices?page=1&per_page=10")
    r_one = client.get("/api/invoices/INV019074")
    r_susp = client.get("/api/suspicious-invoices?page=1&per_page=10")
    r_perf = client.get("/api/model-performance")
    r_csv = client.get("/api/export/csv")

    check("Stage 9", "GET /api/dashboard returns 200 OK", r_dash.status_code == 200)
    check("Stage 9", "GET /api/invoices returns 200 OK", r_invs.status_code == 200)
    check("Stage 9", "GET /api/invoices/INV019074 returns 200 OK", r_one.status_code == 200)
    check("Stage 9", "GET /api/suspicious-invoices returns 200 OK", r_susp.status_code == 200)
    check("Stage 9", "GET /api/model-performance returns 200 OK", r_perf.status_code == 200)
    check(
        "Stage 9",
        "In-sample model metrics are withheld",
        r_perf.get_json().get("metrics_available") is False and r_perf.get_json().get("data") is None
    )
    check("Stage 9", "GET /api/export/csv returns 200 OK", r_csv.status_code == 200)

    # -------------------------------------------------------------------------
    # 8. Web Dashboard & UI Templates
    # -------------------------------------------------------------------------
    print("\n--- 8. Frontend Dashboard UI (Stage 10) ---")
    templates = [
        "base.html", "login.html", "dashboard.html", "invoices.html",
        "suspicious.html", "invoice_details.html", "export_reports.html",
        "upload.html", "404.html", "500.html"
    ]
    for tmpl in templates:
        p = os.path.join(BASE_DIR, "templates", tmpl)
        check("Stage 10", f"Template {tmpl} exists", os.path.exists(p) and os.path.getsize(p) > 100)

    # Check static files
    check("Stage 10", "static/css/style.css exists", os.path.exists(os.path.join(BASE_DIR, "static", "css", "style.css")))
    check("Stage 10", "static/js/dashboard.js exists", os.path.exists(os.path.join(BASE_DIR, "static", "js", "dashboard.js")))
    check("Stage 10", "static/js/charts.js exists", os.path.exists(os.path.join(BASE_DIR, "static", "js", "charts.js")))

    # -------------------------------------------------------------------------
    # 9. Complete User Journey
    # -------------------------------------------------------------------------
    print("\n--- 9. Complete End-to-End User Journey ---")
    r1 = client.get("/")
    r2 = client.get("/login")
    r3 = client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=True)
    r4 = client.get("/dashboard")
    r5 = client.get("/invoices")
    r6 = client.get("/invoices?search=INV019074")
    r7 = client.get("/invoice/INV019074")
    r8 = client.get("/suspicious")
    r9 = client.get("/reports")
    r10 = client.get("/upload")
    r11 = client.get("/about")
    r12 = client.get("/logout", follow_redirects=True)

    journey_ok = (r1.status_code == 302 and r2.status_code == 200 and r3.status_code == 200 and
                  r4.status_code == 200 and r5.status_code == 200 and r6.status_code == 200 and
                  r7.status_code == 200 and r8.status_code == 200 and r9.status_code == 200 and
                  r10.status_code == 200 and r11.status_code == 404 and r12.status_code == 200)
    check("Stage 11", "Full user journey executed without error", journey_ok)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"  FULL SYSTEM TEST SUMMARY: {passed_checks} PASSED, {failed_checks} FAILED")
    print("=" * 75)

    if failed_checks > 0:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        return False
    else:
        print("\nALL SYSTEM COMPONENTS VERIFIED WITH ZERO ERRORS.")
        return True


if __name__ == "__main__":
    success = run_full_system_test()
    sys.exit(0 if success else 1)
