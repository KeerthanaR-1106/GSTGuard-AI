"""
data/test_stage11_integration.py
================================
STAGE 11: End-to-End System Integration Test Suite

Automated GST Invoice Fraud Detection System using Anomaly Detection
for MSME Compliance.

This test suite executes comprehensive end-to-end integration tests
verifying the complete workflow across all modules:
  1. Data Pipeline & Artifact Integrity
  2. Database Integration & Schema Consistency
  3. REST API Endpoint Integration
  4. End-to-End Upload, ML Inference & DB Update Lifecycle
  5. UI Template & View Rendering Integration
  6. Edge Cases, Validation & Error Handling
  7. Viva Compliance & Unsupervised Machine Learning Integrity
"""

import os
import sys
import io
import sqlite3
import uuid
import pandas as pd

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import create_app
from database.database import (
    DB_PATH, SessionLocal, Invoice, Analysis,
    get_dashboard_summary
)


class IntegrationTestSuite:
    def __init__(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.failures = []

    def check(self, category: str, test_name: str, condition: bool, details: str = ""):
        """Record and display individual test assertion result."""
        self.total_tests += 1
        if condition:
            self.passed_tests += 1
            print(f"  [PASS] [{category}] {test_name}")
        else:
            self.failed_tests += 1
            error_msg = f"[{category}] {test_name}: {details}"
            self.failures.append(error_msg)
            print(f"  [FAIL] {error_msg}")

    # =========================================================================
    # SUITE 1: Data Pipeline & Artifact Integrity
    # =========================================================================
    def test_pipeline_artifacts(self):
        print("\n" + "=" * 70)
        print("  SUITE 1: DATA PIPELINE & ARTIFACT INTEGRITY")
        print("=" * 70)

        required_files = [
            ("data/gst_invoices.csv", 20000),
            ("data/ground_truth.csv", 20000),
            ("data/ml_features.csv", 20000),
            ("results/isolation_forest_results.csv", 20000),
            ("results/lof_results.csv", 20000),
            ("results/zscore_results.csv", 20000),
            ("results/ensemble_results.csv", 20000),
            ("results/model_evaluation_metrics.json", None),
            ("database/gst_fraud.db", None)
        ]

        for rel_path, expected_rows in required_files:
            full_path = os.path.join(BASE_DIR, rel_path)
            exists = os.path.exists(full_path)
            self.check("Artifacts", f"File exists: {rel_path}", exists)

            if exists and expected_rows is not None:
                try:
                    df = pd.read_csv(full_path)
                    row_ok = len(df) == expected_rows
                    self.check("Artifacts", f"{rel_path} has {expected_rows:,} rows", row_ok, f"Found {len(df):,}")
                except Exception as e:
                    self.check("Artifacts", f"Read {rel_path}", False, str(e))

    # =========================================================================
    # SUITE 2: Database Integration & Schema Consistency
    # =========================================================================
    def test_database_integration(self):
        print("\n" + "=" * 70)
        print("  SUITE 2: DATABASE INTEGRATION & SCHEMA CONSISTENCY")
        print("=" * 70)

        session = SessionLocal()
        try:
            # Table counts
            inv_count = session.query(Invoice).count()
            ans_count = session.query(Analysis).count()

            self.check("DB", "Invoices table count equals 20,000", inv_count == 20000, f"Found {inv_count}")
            self.check("DB", "Analysis table count equals 20,000", ans_count == 20000, f"Found {ans_count}")

            # 1:1 Relationship check
            unmatched = session.query(Invoice).outerjoin(Analysis, Invoice.invoice_no == Analysis.invoice_no).filter(Analysis.invoice_no.is_(None)).count()
            self.check("DB", "Zero orphan invoice records (1:1 mapping)", unmatched == 0, f"Found {unmatched}")

            # Risk levels check
            valid_levels = {"Low Risk", "Medium Risk", "High Risk"}
            distinct_levels = set(r[0] for r in session.query(Analysis.risk_level).distinct().all())
            self.check("DB", "Risk levels match expected taxonomy", distinct_levels == valid_levels, f"Found {distinct_levels}")

            # Score ranges
            min_score = session.query(sqlite3.func.min(Analysis.risk_score) if hasattr(sqlite3, 'func') else Invoice.total_amount).first()
            scores = [r[0] for r in session.query(Analysis.risk_score).all()]
            self.check("DB", "All risk scores between 0.0 and 100.0", min(scores) >= 0.0 and max(scores) <= 100.0, f"Min={min(scores)}, Max={max(scores)}")

            # Summary function check
            summary = get_dashboard_summary()
            self.check("DB", "get_dashboard_summary returns correct total", summary["total_invoices"] == 20000)
            self.check("DB", "get_dashboard_summary risk distribution sum equals 20,000",
                       sum(summary["risk_distribution"].values()) == 20000)
            self.check("DB", "All risk categories sum to total invoices", sum(summary["risk_distribution"].values()) == inv_count)
            self.check("DB", "Suspicious count equals medium plus high categories",
                       summary["suspicious_invoices"] == summary["medium_risk"] + summary["high_risk"])
            self.check("DB", "Average risk score remains within index bounds",
                       0.0 <= summary["avg_risk_score"] <= 100.0, f"Found {summary['avg_risk_score']}")
        finally:
            session.close()

    # =========================================================================
    # SUITE 3: REST API Endpoint Integration
    # =========================================================================
    def test_rest_api_endpoints(self):
        print("\n" + "=" * 70)
        print("  SUITE 3: REST API ENDPOINT INTEGRATION")
        print("=" * 70)

        # 1. /api/dashboard
        resp = self.client.get("/api/dashboard")
        self.check("API", "GET /api/dashboard returns 200", resp.status_code == 200)
        dash_data = resp.get_json()
        self.check("API", "Dashboard API status is success", dash_data.get("status") == "success")
        self.check("API", "Dashboard API data contains 20,000 invoices", dash_data.get("data", {}).get("total_invoices") == 20000)

        # 2. /api/invoices
        resp = self.client.get("/api/invoices?page=1&per_page=25")
        self.check("API", "GET /api/invoices returns 200", resp.status_code == 200)
        inv_data = resp.get_json()
        self.check("API", "Invoices pagination total is 20,000", inv_data.get("pagination", {}).get("total") == 20000)
        self.check("API", "Invoices pagination returns 25 items", len(inv_data.get("data", [])) == 25)

        # 3. /api/invoices/<invoice_no>
        resp = self.client.get("/api/invoices/INV019074")
        self.check("API", "GET /api/invoices/INV019074 returns 200", resp.status_code == 200)
        item = resp.get_json().get("data", {})
        self.check("API", "Invoice detail contains ML scores", "isolation_score" in item and "lof_score" in item and "max_abs_zscore" in item)
        self.check("API", "Invoice detail contains explanation", len(item.get("explanation", "")) > 10)

        # 4. /api/invoices/INVALID
        resp = self.client.get("/api/invoices/INV999999")
        self.check("API", "GET non-existent invoice returns 404", resp.status_code == 404)

        # 5. /api/suspicious-invoices
        resp = self.client.get("/api/suspicious-invoices?page=1&per_page=50")
        self.check("API", "GET /api/suspicious-invoices returns 200", resp.status_code == 200)
        susp_data = resp.get_json()
        self.check(
            "API",
            "Suspicious invoice total matches dashboard summary",
            susp_data.get("pagination", {}).get("total") == dash_data.get("data", {}).get("suspicious_invoices")
        )
        all_susp = all(x.get("risk_level") in ["Medium Risk", "High Risk"] for x in susp_data.get("data", []))
        self.check("API", "All returned records are Medium or High Risk", all_susp)

        # 6. /api/model-performance
        resp = self.client.get("/api/model-performance")
        self.check("API", "GET /api/model-performance returns 200", resp.status_code == 200)
        perf_data = resp.get_json()
        self.check("API", "In-sample model performance is marked unavailable", perf_data.get("metrics_available") is False and perf_data.get("data") is None)
        self.check("API", "Model performance explains evaluation limitation", "in-sample" in perf_data.get("message", ""))

        # 7. /api/export/csv
        resp = self.client.get("/api/export/csv")
        self.check("API", "GET /api/export/csv returns 200", resp.status_code == 200)
        self.check("API", "Export content type is text/csv", "text/csv" in resp.content_type)
        self.check("API", "Export attachment filename header present", "gst_compliance_report" in resp.headers.get("Content-Disposition", ""))

        # 8. Filtered export
        resp = self.client.get("/api/export/csv?risk_level=High%20Risk")
        self.check("API", "GET /api/export/csv?risk_level=High%20Risk returns 200", resp.status_code == 200)
        exported_df = pd.read_csv(io.BytesIO(resp.data))
        self.check(
            "API",
            "Filtered High Risk CSV count matches dashboard summary",
            len(exported_df) == dash_data.get("data", {}).get("high_risk"),
            f"Found {len(exported_df)}"
        )

    # =========================================================================
    # SUITE 4: End-to-End Upload, ML Inference & DB Update Lifecycle
    # =========================================================================
    def test_upload_and_inference_lifecycle(self):
        print("\n" + "=" * 70)
        print("  SUITE 4: END-TO-END UPLOAD, ML INFERENCE & DB UPDATE LIFECYCLE")
        print("=" * 70)

        # Take a slice of 10 real invoices to simulate a user batch upload
        df_raw = pd.read_csv(os.path.join(BASE_DIR, "data", "gst_invoices.csv")).head(10).copy()
        test_prefix = f"STG11_TEST_{uuid.uuid4().hex}"
        test_invoice_ids = [f"{test_prefix}_{index:04d}" for index in range(len(df_raw))]
        df_raw["invoice_no"] = test_invoice_ids
        
        # Modify one invoice to have an intentional discrepancy for inference testing
        test_inv_id = df_raw.iloc[0]["invoice_no"]
        df_raw.loc[0, "unit_price"] = 250000.0  # Extreme unit price
        df_raw.loc[0, "taxable_value"] = 250000.0 * df_raw.loc[0, "quantity"]
        df_raw.loc[0, "total_tax"] = df_raw.loc[0, "taxable_value"] * (df_raw.loc[0, "gst_rate"] / 100.0)
        df_raw.loc[0, "total_amount"] = df_raw.loc[0, "taxable_value"] + df_raw.loc[0, "total_tax"]

        csv_buffer = io.BytesIO()
        df_raw.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # POST /api/upload
        resp = self.client.post(
            "/api/upload",
            data={"file": (csv_buffer, "batch_test_upload.csv")},
            content_type="multipart/form-data"
        )

        self.check("Upload", "POST /api/upload returns 200 OK", resp.status_code == 200)
        res_json = resp.get_json()
        self.check("Upload", "Upload response status is success", res_json.get("status") == "success")
        self.check("Upload", "Processed count equals 10", res_json.get("processed_count") == 10)
        self.check("Upload", "Summary dict returned with risk counts", "summary" in res_json)

        # Verify database was updated and invoice can be retrieved via API
        resp_check = self.client.get(f"/api/invoices/{test_inv_id}")
        self.check("Upload", f"Uploaded invoice {test_inv_id} queryable in API", resp_check.status_code == 200)
        inv_data = resp_check.get_json().get("data", {})
        self.check("Upload", "Updated invoice unit price reflected in database", inv_data.get("unit_price") == 250000.0)
        self.check("Upload", "ML analysis record created with risk score", inv_data.get("risk_score") is not None)

        connection = sqlite3.connect(DB_PATH)
        try:
            placeholders = ", ".join("?" for _ in test_invoice_ids)
            cursor = connection.cursor()
            for table in ("investigation_history", "investigations", "analysis", "invoices"):
                cursor.execute(
                    f"DELETE FROM {table} WHERE invoice_no IN ({placeholders})",
                    test_invoice_ids
                )
            connection.commit()
        finally:
            connection.close()

    # =========================================================================
    # SUITE 5: UI Template & View Rendering Integration
    # =========================================================================
    def test_ui_views_and_templates(self):
        print("\n" + "=" * 70)
        print("  SUITE 5: UI TEMPLATE & VIEW RENDERING INTEGRATION")
        print("=" * 70)

        # 1. Login flow
        resp = self.client.get("/login")
        self.check("UI", "GET /login returns 200", resp.status_code == 200)
        login_html = resp.data.decode("utf-8")
        self.check("UI", "Login page does not disclose demo credentials", "admin123" not in login_html and "Demo Credentials" not in login_html)

        # Valid login creates session
        resp = self.client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=True)
        self.check("UI", "POST /login with demo credentials authenticates", resp.status_code == 200)

        # 2. Dashboard View
        resp = self.client.get("/dashboard")
        self.check("UI", "GET /dashboard returns 200", resp.status_code == 200)
        dash_html = resp.data.decode("utf-8")
        self.check("UI", "Dashboard contains stat cards", "Total Invoices" in dash_html and "20,000" in dash_html or "20000" in dash_html)
        self.check("UI", "Dashboard contains risk doughnut canvas", "riskDoughnutChart" in dash_html)
        self.check("UI", "Dashboard contains anomaly bar canvas", "anomalyBarChart" in dash_html)
        self.check("UI", "Dashboard loads Chart.js CDN script", "chart.umd.min.js" in dash_html)

        # 3. All Invoices View
        resp = self.client.get("/invoices")
        self.check("UI", "GET /invoices returns 200", resp.status_code == 200)
        inv_html = resp.data.decode("utf-8")
        self.check("UI", "Invoices page displays table", "<table" in inv_html)
        self.check("UI", "Invoices page displays pagination", "pagination" in inv_html)

        # 4. Search Filter
        resp = self.client.get("/invoices?search=INV019074")
        self.check("UI", "Search for INV019074 returns 200", resp.status_code == 200)
        self.check("UI", "Search result contains INV019074", "INV019074" in resp.data.decode("utf-8"))

        # 5. Suspicious View
        resp = self.client.get("/suspicious")
        self.check("UI", "GET /suspicious returns 200", resp.status_code == 200)
        susp_html = resp.data.decode("utf-8")
        self.check("UI", "Suspicious view displays warning banner", "Compliance" in susp_html)
        self.check("UI", "Suspicious view states anomaly findings require verification", "does not establish fraud" in susp_html)

        # 6. Invoice Details Drill-Down View
        resp = self.client.get("/invoice/INV019074")
        self.check("UI", "GET /invoice/INV019074 returns 200", resp.status_code == 200)
        det_html = resp.data.decode("utf-8")
        self.check("UI", "Details page shows Isolation Forest score", "Isolation Forest" in det_html)
        self.check("UI", "Details page shows LOF score", "Local Outlier Factor" in det_html)
        self.check("UI", "Details page shows Z-Score", "Z-Score" in det_html)
        self.check("UI", "Details page shows Risk Score progress bar", "progress-bar" in det_html)
        self.check("UI", "Details page shows GST tax breakdown", "Total Tax" in det_html)

        # 7. Reports View
        resp = self.client.get("/reports")
        self.check("UI", "GET /reports returns 200", resp.status_code == 200)
        rep_html = resp.data.decode("utf-8")
        self.check("UI", "Reports page identifies report export options", "Export Reports" in rep_html)
        self.check("UI", "Reports page has CSV download buttons", "/api/export/csv" in rep_html)

        # 8. Upload View
        resp = self.client.get("/upload")
        self.check("UI", "GET /upload returns 200", resp.status_code == 200)
        up_html = resp.data.decode("utf-8")
        self.check("UI", "Upload page contains drag-and-drop zone", "drop-zone" in up_html)
        self.check("UI", "Upload page references /api/upload endpoint", "/api/upload" in up_html)

        # 9. Removed informational page
        resp = self.client.get("/about")
        self.check("UI", "GET /about returns 404", resp.status_code == 404)

        # 10. 404 Error Page
        resp = self.client.get("/non-existent-page-route")
        self.check("UI", "Invalid route returns 404", resp.status_code == 404)
        self.check("UI", "Custom 404 template rendered", "404" in resp.data.decode("utf-8"))

    # =========================================================================
    # SUITE 6: Edge Cases, Boundary Tests & Error Handling
    # =========================================================================
    def test_edge_cases_and_error_handling(self):
        print("\n" + "=" * 70)
        print("  SUITE 6: EDGE CASES, BOUNDARY TESTS & ERROR HANDLING")
        print("=" * 70)

        # 1. Non-CSV upload
        resp = self.client.post(
            "/api/upload",
            data={"file": (io.BytesIO(b"Hello world"), "test.txt")},
            content_type="multipart/form-data"
        )
        self.check("EdgeCase", "Non-CSV upload rejected with 400", resp.status_code == 400)

        # 2. Empty file upload
        resp = self.client.post(
            "/api/upload",
            data={"file": (io.BytesIO(b""), "empty.csv")},
            content_type="multipart/form-data"
        )
        self.check("EdgeCase", "Empty CSV upload rejected with 400", resp.status_code == 400)

        # 3. CSV with missing required columns
        incomplete_csv = io.BytesIO(
            b"invoice_no,invoice_date,total_amount\n"
            b"INV001,2023-01-01,500.0\n"
            b"INV002,2023-01-02,600.0\n"
        )
        resp = self.client.post(
            "/api/upload",
            data={"file": (incomplete_csv, "incomplete.csv")},
            content_type="multipart/form-data"
        )
        self.check("EdgeCase", "Incomplete schema rejected with 400", resp.status_code == 400)
        self.check("EdgeCase", "Error message lists missing columns", "Missing required columns" in resp.get_json().get("message", ""))

        # 4. Search with non-existent invoice
        resp = self.client.get("/invoices?search=INV999999")
        self.check("EdgeCase", "Search non-existent invoice returns 200 cleanly", resp.status_code == 200)
        self.check("EdgeCase", "Empty state message rendered", "No invoices found" in resp.data.decode("utf-8"))

        # 5. Single invoice 404
        resp = self.client.get("/invoice/INV999999")
        self.check("EdgeCase", "GET /invoice/INV999999 returns 404", resp.status_code == 404)

        # 6. Pagination cap (prevent memory exhaustion)
        resp = self.client.get("/api/invoices?per_page=5000")
        self.check("EdgeCase", "Large per_page capped safely to max 100", resp.get_json().get("pagination", {}).get("per_page") == 100)

        # 7. SQL injection test in search parameter
        resp = self.client.get("/invoices?search=' OR '1'='1")
        self.check("EdgeCase", "SQL injection string handled safely without error", resp.status_code == 200)

    # =========================================================================
    # SUITE 7: Viva Compliance & Unsupervised ML Integrity
    # =========================================================================
    def test_viva_and_ml_compliance(self):
        print("\n" + "=" * 70)
        print("  SUITE 7: VIVA COMPLIANCE & UNSUPERVISED ML INTEGRITY")
        print("=" * 70)

        # 1. Unsupervised model isolation: ground truth is not used in ML training or calculation
        df_ml = pd.read_csv(os.path.join(BASE_DIR, "data", "ml_features.csv"))
        forbidden = ["actual_anomaly", "anomaly_type", "ground_truth", "label", "is_anomaly"]
        forbidden_present = [c for c in forbidden if c in df_ml.columns]
        self.check("Compliance", "ml_features.csv contains no ground truth labels", len(forbidden_present) == 0, f"Found {forbidden_present}")

        # Check training/calculation functions in model files
        with open(os.path.join(BASE_DIR, "models", "isolation_forest.py"), "r", encoding="utf-8") as f:
            if_code = f.read()
            if_func = if_code[if_code.find("def run_isolation_forest"):if_code.find("def save_results")]
            self.check("Compliance", "Isolation Forest training function uses no ground truth", "ground_truth" not in if_func and "actual_anomaly" not in if_func)

        with open(os.path.join(BASE_DIR, "models", "lof.py"), "r", encoding="utf-8") as f:
            lof_code = f.read()
            lof_func = lof_code[lof_code.find("def run_lof"):lof_code.find("def save_results")]
            self.check("Compliance", "LOF training function uses no ground truth", "ground_truth" not in lof_func and "actual_anomaly" not in lof_func)

        with open(os.path.join(BASE_DIR, "models", "zscore.py"), "r", encoding="utf-8") as f:
            z_code = f.read()
            z_func = z_code[z_code.find("def calculate_zscores"):z_code.find("def save_results")]
            self.check("Compliance", "Z-Score calculation function uses no ground truth", "ground_truth" not in z_func and "actual_anomaly" not in z_func)

        # 2. Risk classification thresholds verification
        session = SessionLocal()
        try:
            # Low: 0-39.99
            max_low = session.query(sqlite3.func.max(Analysis.risk_score) if hasattr(sqlite3, 'func') else Analysis.risk_score).filter(Analysis.risk_level == "Low Risk").first()
            scores_low = [r[0] for r in session.query(Analysis.risk_score).filter(Analysis.risk_level == "Low Risk").all()]
            self.check("Compliance", "Low Risk scores are strictly below 40.0", max(scores_low) < 40.0, f"Max Low = {max(scores_low)}")

            # Medium: 40-69.99
            scores_med = [r[0] for r in session.query(Analysis.risk_score).filter(Analysis.risk_level == "Medium Risk").all()]
            self.check("Compliance", "Medium Risk scores between 40.0 and 69.99", min(scores_med) >= 40.0 and max(scores_med) < 70.0, f"Min={min(scores_med)}, Max={max(scores_med)}")

            # High: 70-100
            scores_high = [r[0] for r in session.query(Analysis.risk_score).filter(Analysis.risk_level == "High Risk").all()]
            self.check("Compliance", "High Risk scores are 70.0 or above", min(scores_high) >= 70.0, f"Min High = {min(scores_high)}")

        finally:
            session.close()

    # =========================================================================
    # Runner
    # =========================================================================
    def run_all(self):
        print("\n" + "=" * 70)
        print("  AUTOMATED GST INVOICE FRAUD DETECTION SYSTEM")
        print("  STAGE 11: FULL END-TO-END INTEGRATION TEST SUITE")
        print("=" * 70)

        self.test_pipeline_artifacts()
        self.test_database_integration()
        self.test_rest_api_endpoints()
        self.test_upload_and_inference_lifecycle()
        self.test_ui_views_and_templates()
        self.test_edge_cases_and_error_handling()
        self.test_viva_and_ml_compliance()

        print("\n" + "=" * 70)
        print("  STAGE 11 INTEGRATION TEST SUMMARY")
        print("=" * 70)
        print(f"  TOTAL TESTS EXECUTED : {self.total_tests}")
        print(f"  TESTS PASSED         : {self.passed_tests}")
        print(f"  TESTS FAILED         : {self.failed_tests}")

        if self.failed_tests > 0:
            print("\n  FAILURES:")
            for f in self.failures:
                print(f"    - {f}")
            print("\n  STAGE 11 INTEGRATION VERDICT: FAIL")
            return False
        else:
            print("\n  ALL INTEGRATION TESTS PASSED PERFECTLY!")
            print("  STAGE 11 INTEGRATION VERDICT: PASS")
            return True


if __name__ == "__main__":
    suite = IntegrationTestSuite()
    success = suite.run_all()
    sys.exit(0 if success else 1)
