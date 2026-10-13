"""
data/test_gstguard_ai.py
========================
Comprehensive Verification Suite for GSTGuard AI Transformation:
  1. Blueprint and App Factory Registration
  2. Database Schema & Investigation Model
  3. Investigation Workflow API (GET, PATCH, status updates)
  4. Supplier Risk Intelligence & APIs
  5. Priority Review Queue & APIs
  6. Compliance Copilot Q&A Engine & API
  7. Explanation Service & Empirical Evidence Benchmarks
  8. MSME Compliance Action Center Checklists
  9. Report Generation Service & Views
  10. Centralized Anomaly Pipeline & Demo Analysis
"""

import unittest
import json
import io
from pathlib import Path
from app import create_app
from database.database import (
    SessionLocal, Invoice, Analysis, get_review_queue, get_supplier_metrics,
    get_supplier_detail, get_or_create_investigation, get_active_upload_batch_id
)
from services.explanation_service import get_detailed_explanation
from services.recommendation_service import get_recommendations_for_invoice
from services.copilot_service import query_copilot
from services.report_service import generate_invoice_report_data
from services.anomaly_service import (
    _build_features,
    calculate_weighted_anomaly_index,
    classify_risk_level,
    run_anomaly_pipeline,
)
from services.financial_validation import validate_invoice_arithmetic
import pandas as pd


class TestGSTGuardAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        login_response = cls.client.post(
            "/login",
            data={
                "username": cls.app.config["DEMO_USER"],
                "password": cls.app.config["DEMO_PASSWORD"],
            },
        )
        if login_response.status_code != 302:
            raise RuntimeError("Failed to authenticate the integration test client.")
        session = SessionLocal()
        try:
            session.add_all([
                Invoice(
                    invoice_no="TRAINING-ONLY-001",
                    invoice_date="2023-01-01",
                    supplier_id="TRAINING-ONLY",
                    supplier_gstin="29ABCDE1234F1Z5",
                    buyer_id="TRAINING-BUYER",
                    buyer_gstin="29ABCDE5678F1Z5",
                    product_category="Training",
                    hsn_code="0000",
                    quantity=1,
                    unit_price=10,
                    discount_percentage=0,
                    discount_amount=0,
                    taxable_value=10,
                    gst_rate=18,
                    cgst=0.9,
                    sgst=0.9,
                    igst=0,
                    cess_rate=0,
                    cess_amount=0,
                    total_tax=1.8,
                    total_amount=11.8,
                    payment_method="Test",
                ),
                Analysis(
                    invoice_no="TRAINING-ONLY-001",
                    isolation_score=0,
                    isolation_anomaly=0,
                    lof_score=0,
                    lof_anomaly=0,
                    max_abs_zscore=0,
                    zscore_anomaly=0,
                    risk_score=0,
                    risk_level="Low Risk",
                    anomaly_type="Normal",
                    explanation="Training-only test row",
                ),
            ])
            session.commit()
        finally:
            session.close()

        empty_dashboard = cls.client.get("/api/dashboard").get_json()["data"]
        empty_invoice_list = cls.client.get("/api/invoices").get_json()
        if empty_dashboard["total_invoices"] != 0 or empty_invoice_list["pagination"]["total"] != 0:
            raise RuntimeError("Legacy training rows leaked into application results.")

        source = pd.read_csv(Path(__file__).with_name("gst_invoices.csv"))
        supplier_examples = source[source["supplier_id"] == "SUP0079"].drop_duplicates(
            "supplier_gstin"
        ).head(2)
        outlier = source.iloc[[0]].copy()
        outlier["invoice_no"] = "INV987654321"
        outlier["unit_price"] = 10_000_000
        outlier["quantity"] = 10_000
        outlier["taxable_value"] = 100_000_000_000
        outlier["total_tax"] = 12_000_000_000
        outlier["total_amount"] = 112_000_000_000
        cls.baseline_frame = pd.concat(
            [source.head(100), supplier_examples, outlier], ignore_index=True
        )
        response = cls.upload_frame(cls.baseline_frame, "test-baseline.csv")
        if response.status_code != 200:
            raise RuntimeError("Failed to load the test invoice dataset.")
        with cls.client.session_transaction() as flask_session:
            if not flask_session.get("has_uploaded_csv"):
                raise RuntimeError("Successful CSV upload did not enable dashboard statistics.")

    @classmethod
    def upload_frame(cls, frame, filename):
        buffer = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
        return cls.client.post(
            "/api/upload",
            data={"file": (buffer, filename)},
            content_type="multipart/form-data"
        )

    # ── 1. App Factory & Blueprints ──────────────────────────────────────────
    def test_01_blueprints_registered(self):
        """All 6 core blueprints must be registered."""
        blueprint_names = set(self.app.blueprints.keys())
        expected = {"dashboard", "invoices", "reports", "upload", "suppliers", "copilot"}
        self.assertTrue(expected.issubset(blueprint_names), f"Missing blueprints: {expected - blueprint_names}")

    def test_01a_login_required_for_application_pages_and_apis(self):
        """Unauthenticated visitors must sign in before accessing application data."""
        with self.client.session_transaction() as flask_session:
            flask_session.clear()

        root_response = self.client.get("/")
        dashboard_response = self.client.get("/dashboard")
        queue_response = self.client.get("/review-queue")
        api_response = self.client.get("/api/dashboard")

        self.assertEqual(root_response.status_code, 302)
        self.assertTrue(root_response.headers["Location"].endswith("/login"))
        self.assertEqual(dashboard_response.status_code, 302)
        self.assertTrue(dashboard_response.headers["Location"].endswith("/login"))
        self.assertEqual(queue_response.status_code, 302)
        self.assertTrue(queue_response.headers["Location"].endswith("/login"))
        self.assertEqual(api_response.status_code, 401)
        self.assertEqual(api_response.get_json()["status"], "error")

        with self.client.session_transaction() as flask_session:
            flask_session["logged_in"] = True
            flask_session["user"] = "test-reviewer"
            flask_session["has_uploaded_csv"] = True

    def test_01b_dashboard_waits_for_csv_after_each_login(self):
        """A fresh login hides the previous batch until a CSV is uploaded."""
        with self.client.session_transaction() as flask_session:
            self.assertTrue(flask_session.get("has_uploaded_csv"))

        login_response = self.client.post(
            "/login",
            data={
                "username": self.app.config["DEMO_USER"],
                "password": self.app.config["DEMO_PASSWORD"],
            },
        )
        self.assertEqual(login_response.status_code, 302)

        dashboard_response = self.client.get("/dashboard")
        self.assertEqual(dashboard_response.status_code, 200)
        self.assertIn(b"Upload and analyze a CSV to populate dashboard statistics", dashboard_response.data)
        self.assertIn(b"Total Invoices Evaluated", dashboard_response.data)
        self.assertIn(b"Priority Review Queue", dashboard_response.data)
        self.assertIn(b"Top Risky Suppliers", dashboard_response.data)
        self.assertIn(b'"total_invoices": 0', dashboard_response.data)

        for path, page_heading in (
            ("/review-queue", b"Priority Review Queue"),
            ("/suppliers", b"Supplier Risk"),
            ("/reports", b"Export"),
            ("/invoices", b"All Invoices Ledger"),
            ("/suspicious", b"Invoices Flagged for Review"),
        ):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn(page_heading, response.data, path)

        empty_lists = (
            ("/api/review-queue", "data"),
            ("/api/suppliers", "data"),
            ("/api/invoices", "data"),
            ("/api/suspicious-invoices", "data"),
        )
        for path, key in empty_lists:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertEqual(response.get_json()[key], [], path)

        report_response = self.client.get("/api/export/csv")
        self.assertEqual(report_response.status_code, 200)
        self.assertEqual(
            report_response.data.decode("utf-8").count("\n"),
            1,
            "CSV export should contain its header but no previous-session data.",
        )

        api_response = self.client.get("/api/dashboard")
        self.assertEqual(api_response.status_code, 200)
        self.assertEqual(api_response.get_json()["data"]["total_invoices"], 0)

        upload_response = self.upload_frame(self.baseline_frame, "fresh-login-upload.csv")
        self.assertEqual(upload_response.status_code, 200)
        refreshed_dashboard = self.client.get("/dashboard")
        self.assertIn(b"Total Invoices Evaluated", refreshed_dashboard.data)
        self.assertIn(b'"total_invoices": 103', refreshed_dashboard.data)

        with self.client.session_transaction() as flask_session:
            self.assertTrue(flask_session.get("has_uploaded_csv"))

    # ── 2. Database Schema & Models ──────────────────────────────────────────
    def test_02_database_models(self):
        """Database must have Invoices, Analysis, and Investigations accessible."""
        session = SessionLocal()
        try:
            active_batch_id = get_active_upload_batch_id(session)
            inv_count = session.query(Invoice).filter(
                Invoice.upload_batch_id == active_batch_id
            ).count()
            ans_count = session.query(Analysis).join(
                Invoice, Invoice.invoice_no == Analysis.invoice_no
            ).filter(Invoice.upload_batch_id == active_batch_id).count()
            self.assertGreater(inv_count, 0)
            self.assertGreater(ans_count, 0)

            # Test investigation query
            first_inv = session.query(Invoice).filter(
                Invoice.upload_batch_id == active_batch_id
            ).first()
            self.assertIsNotNone(first_inv)

            inv_rec = get_or_create_investigation(first_inv.invoice_no)
            self.assertIn("status", inv_rec)
            self.assertIn("invoice_no", inv_rec)
        finally:
            session.close()

    # ── 3. Investigation Workflow ────────────────────────────────────────────
    def test_03_investigation_workflow_api(self):
        """Test GET and PATCH /api/investigations/<invoice_no>."""
        session = SessionLocal()
        try:
            active_batch_id = get_active_upload_batch_id(session)
            first_inv = session.query(Invoice).filter(
                Invoice.upload_batch_id == active_batch_id
            ).first()
            inv_no = first_inv.invoice_no
        finally:
            session.close()

        # GET
        res_get = self.client.get(f"/api/investigations/{inv_no}")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.get_json()
        self.assertEqual(data_get["status"], "success")

        with self.client.session_transaction() as flask_session:
            flask_session["logged_in"] = True
            flask_session["user"] = "test-reviewer"
            flask_session["has_uploaded_csv"] = True

        # PATCH update
        payload = {
            "status": "UNDER REVIEW",
            "notes": "Verified against e-Way bill #EW12345.",
            "assigned_to": "Senior Officer Raman"
        }
        res_patch = self.client.patch(
            f"/api/investigations/{inv_no}",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(res_patch.status_code, 200)
        data_patch = res_patch.get_json()
        self.assertEqual(data_patch["data"]["status"], "UNDER REVIEW")
        self.assertEqual(data_patch["data"]["assigned_to"], "Senior Officer Raman")
        self.assertEqual(data_patch["data"]["notes"], payload["notes"])
        self.assertTrue(any(
            event["reviewer"] == "test-reviewer"
            and event["new_status"] == "UNDER REVIEW"
            and event["decision_notes"] == payload["notes"]
            for event in data_patch["history"]
        ))

    def test_03_investigation_rejects_invalid_updates(self):
        session = SessionLocal()
        try:
            active_batch_id = get_active_upload_batch_id(session)
            invoice_no = session.query(Invoice.invoice_no).filter(
                Invoice.upload_batch_id == active_batch_id
            ).first()[0]
        finally:
            session.close()
        with self.client.session_transaction() as flask_session:
            flask_session.clear()
        unauthenticated = self.client.patch(
            f"/api/investigations/{invoice_no}",
            json={"status": "UNDER REVIEW"}
        )
        self.assertEqual(unauthenticated.status_code, 401)

        with self.client.session_transaction() as flask_session:
            flask_session["logged_in"] = True
            flask_session["user"] = "test-reviewer"
            flask_session["has_uploaded_csv"] = True
        invalid_status = self.client.patch(
            f"/api/investigations/{invoice_no}",
            json={"status": "FRAUD"}
        )
        self.assertEqual(invalid_status.status_code, 400)

        missing_invoice = self.client.patch(
            "/api/investigations/DOES-NOT-EXIST",
            json={"status": "UNDER REVIEW"}
        )
        self.assertEqual(missing_invoice.status_code, 404)

        missing_get = self.client.get("/api/investigations/DOES-NOT-EXIST")
        self.assertEqual(missing_get.status_code, 404)

    def test_03_upload_rejects_invalid_source_values(self):
        original_sample = pd.read_csv(Path(__file__).with_name("gst_invoices.csv")).head(2).copy()
        collision_response = self.client.post(
            "/api/upload",
            data={
                "file": (
                    io.BytesIO(original_sample.to_csv(index=False).encode("utf-8")),
                    "duplicate-ledger-ids.csv"
                )
            },
            content_type="multipart/form-data"
        )
        self.assertEqual(collision_response.status_code, 200)
        self.assertEqual(collision_response.get_json()["processed_count"], 2)
        self.assertEqual(
            collision_response.get_json()["dataset_filename"],
            "duplicate-ledger-ids.csv"
        )
        active_invoice_ids = {
            item["invoice_no"]
            for item in self.client.get("/api/invoices?per_page=100").get_json()["data"]
        }
        self.assertEqual(active_invoice_ids, set(original_sample["invoice_no"].astype(str)))
        self.assertNotIn("TRAINING-ONLY-001", active_invoice_ids)
        dashboard = self.client.get("/api/dashboard").get_json()["data"]
        self.assertEqual(dashboard["total_invoices"], 2)
        self.assertEqual(
            self.client.get("/api/invoices?per_page=100").get_json()["pagination"]["total"],
            2
        )
        exported = pd.read_csv(
            io.BytesIO(self.client.get("/api/export/csv").data)
        )
        self.assertEqual(set(exported["Invoice No"]), active_invoice_ids)

        sample = original_sample.copy()
        sample["invoice_no"] = ["VALIDATION-001", "VALIDATION-002"]

        def upload(frame):
            buffer = io.BytesIO()
            frame.to_csv(buffer, index=False)
            buffer.seek(0)
            return self.client.post(
                "/api/upload",
                data={"file": (buffer, "validation.csv")},
                content_type="multipart/form-data"
            )

        negative_amount = sample.copy()
        negative_amount.loc[0, "unit_price"] = -1
        response = upload(negative_amount)
        self.assertEqual(response.status_code, 400)
        self.assertIn("negative value", response.get_json()["message"])

        duplicate_ids = sample.copy()
        duplicate_ids.loc[1, "invoice_no"] = duplicate_ids.loc[0, "invoice_no"]
        response = upload(duplicate_ids)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Duplicate invoice_no", response.get_json()["message"])

        invalid_date = sample.copy()
        invalid_date.loc[0, "invoice_date"] = "not-a-date"
        response = upload(invalid_date)
        self.assertEqual(response.status_code, 400)
        self.assertIn("invoice_date", response.get_json()["message"])

        restored = self.upload_frame(self.baseline_frame, "test-baseline.csv")
        self.assertEqual(restored.status_code, 200)

    # ── 4. Supplier Intelligence ─────────────────────────────────────────────
    def test_04_supplier_service_and_apis(self):
        """Test supplier aggregation queries, views, and APIs."""
        suppliers_res = get_supplier_metrics(page=1, per_page=10)
        self.assertIn("items", suppliers_res)
        self.assertGreater(len(suppliers_res["items"]), 0)

        first_supplier = suppliers_res["items"][0]
        self.assertIn("supplier_risk_score", first_supplier)
        self.assertIn("supplier_risk_level", first_supplier)

        # Directory and scorecard must aggregate all GSTIN registrations under one supplier ID.
        multi_gstin_supplier = "SUP0079"
        directory_supplier = next(
            item for item in get_supplier_metrics(page=1, per_page=1000)["items"]
            if item["supplier_id"] == multi_gstin_supplier
        )
        scorecard_supplier = get_supplier_detail(multi_gstin_supplier)
        self.assertEqual(directory_supplier["supplier_risk_score"], scorecard_supplier["supplier_risk_score"])
        self.assertEqual(directory_supplier["supplier_risk_level"], scorecard_supplier["supplier_risk_level"])

        # Test API
        api_res = self.client.get("/api/suppliers")
        self.assertEqual(api_res.status_code, 200)
        api_data = api_res.get_json()
        self.assertEqual(api_data["status"], "success")

        # Test detail view
        sup_id = first_supplier["supplier_id"]
        detail_res = self.client.get(f"/suppliers/{sup_id}")
        self.assertEqual(detail_res.status_code, 200)
        self.assertIn(b"Supplier Risk Score", detail_res.data)

    # ── 5. Priority Review Queue ─────────────────────────────────────────────
    def test_05_priority_review_queue(self):
        """Test ranked review queue generation and view."""
        queue = get_review_queue(limit=25)
        self.assertIsInstance(queue, list)
        self.assertGreater(len(queue), 0)

        top_item = queue[0]
        self.assertIn("priority_score", top_item)
        self.assertIn("consensus_count", top_item)
        self.assertGreaterEqual(top_item["priority_score"], 0)

        # Review Queue Route
        res = self.client.get("/review-queue")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Priority Review Queue", res.data)
        self.assertIn(b"CONFIRMED IRREGULARITY", res.data)
        self.assertIn(b"FALSE POSITIVE", res.data)

        # Review Queue API
        api_res = self.client.get("/api/review-queue")
        self.assertEqual(api_res.status_code, 200)

    # ── 6. Compliance Copilot ────────────────────────────────────────────────
    def test_06_compliance_copilot(self):
        """Test deterministic Copilot query engine."""
        # General query
        res1 = query_copilot("Explain GSTR-2B rules")
        self.assertIn("answer", res1)
        self.assertIn("GSTR-2B", res1["answer"])
        self.assertIn("does not by itself", res1["answer"])

        scoring_guidance = query_copilot("Explain the ensemble scoring formula")
        self.assertIn("training-set score ranges", scoring_guidance["answer"])
        self.assertNotIn("calibrated to", scoring_guidance["answer"])

        # General query API accepts the null invoice number sent by the web client.
        general_api_res = self.client.post(
            "/api/copilot/query",
            data=json.dumps({"query": "Explain GSTR-2B compliance rules", "invoice_no": None}),
            content_type="application/json"
        )
        self.assertEqual(general_api_res.status_code, 200)
        general_data = general_api_res.get_json()
        self.assertEqual(general_data["status"], "success")
        self.assertIn("GSTR-2B", general_data["data"]["answer"])

        risk_api_res = self.client.post(
            "/api/copilot/query",
            data=json.dumps({"query": "What are the key risk categories?", "invoice_no": None}),
            content_type="application/json"
        )
        self.assertEqual(risk_api_res.status_code, 200)
        self.assertEqual(risk_api_res.get_json()["data"]["category"], "risk_thresholds")
        self.assertIn("40–69", risk_api_res.get_json()["data"]["answer"])

        # Invoice query
        session = SessionLocal()
        try:
            high_risk_inv = session.query(Invoice, Analysis).join(
                Analysis, Invoice.invoice_no == Analysis.invoice_no
            ).filter(
                Invoice.upload_batch_id == get_active_upload_batch_id(session),
                Analysis.risk_level == "High Risk"
            ).first()
            if high_risk_inv:
                inv_dict = high_risk_inv[0].to_dict()
                inv_dict.update(high_risk_inv[1].to_dict())
            else:
                inv_dict = None
        finally:
            session.close()

        if inv_dict:
            res_why = query_copilot("Why was this invoice flagged?", invoice_data=inv_dict)
            self.assertIn("answer", res_why)
            self.assertIn("Training-baseline anomaly index", res_why["answer"])
            itc_guidance = query_copilot("Is it safe to claim ITC?", invoice_data=inv_dict)
            self.assertIn("neither establishes nor determines ITC eligibility", itc_guidance["answer"])
            self.assertNotIn("SAFE TO CLAIM", itc_guidance["answer"])
            self.assertNotIn("DO NOT CLAIM YET", itc_guidance["answer"])

            embedded_invoice_api_res = self.client.post(
                "/api/copilot/query",
                data=json.dumps({
                    "query": f"Why was invoice {inv_dict['invoice_no']} flagged?"
                }),
                content_type="application/json"
            )
            self.assertEqual(embedded_invoice_api_res.status_code, 200)
            embedded_data = embedded_invoice_api_res.get_json()["data"]
            self.assertEqual(embedded_data["category"], "explanation")
            self.assertIn(inv_dict["invoice_no"], embedded_data["answer"])

            supplier_summary_res = self.client.post(
                "/api/copilot/query",
                data=json.dumps({
                    "query": (
                        f"What company provided invoice {inv_dict['invoice_no']}, how many invoices "
                        "has it produced, and what are their risk levels and anomaly patterns?"
                    )
                }),
                content_type="application/json"
            )
            self.assertEqual(supplier_summary_res.status_code, 200)
            supplier_summary = supplier_summary_res.get_json()["data"]
            self.assertEqual(supplier_summary["category"], "supplier_summary")
            self.assertIn("Invoices in this dataset", supplier_summary["answer"])
            self.assertIn("Taxable turnover in this dataset", supplier_summary["answer"])
            self.assertIn("Total invoice value", supplier_summary["answer"])
            self.assertIn("Anomaly patterns", supplier_summary["answer"])
            self.assertIn("Not recorded in this dataset", supplier_summary["answer"])
            self.assertEqual(
                supplier_summary_res.get_json()["supplier_id"],
                inv_dict["supplier_id"],
            )

            supplier_turnover_res = self.client.post(
                "/api/copilot/query",
                json={
                    "query": (
                        f"What is the total invoice turnover for supplier "
                        f"{inv_dict['supplier_id']}?"
                    )
                },
            )
            self.assertEqual(supplier_turnover_res.status_code, 200)
            supplier_turnover = supplier_turnover_res.get_json()["data"]
            self.assertEqual(
                supplier_turnover["category"],
                "supplier_summary",
                supplier_turnover_res.get_json(),
            )
            self.assertIn("Taxable turnover in this dataset", supplier_turnover["answer"])

            invoice_details_res = self.client.post(
                "/api/copilot/query",
                json={"query": f"Show me the full invoice details for {inv_dict['invoice_no']}"},
            )
            self.assertEqual(invoice_details_res.status_code, 200)
            invoice_details = invoice_details_res.get_json()["data"]
            self.assertEqual(invoice_details["category"], "invoice_details")
            self.assertIn("Total amount", invoice_details["answer"])
            self.assertIn(inv_dict["invoice_no"], invoice_details["answer"])

            suggested_methods_res = self.client.post(
                "/api/copilot/query",
                json={"query": f"Suggest methods to verify invoice {inv_dict['invoice_no']}"},
            )
            self.assertEqual(suggested_methods_res.status_code, 200)
            self.assertEqual(
                suggested_methods_res.get_json()["data"]["category"],
                "recommendations",
            )

            risk_reduction_res = self.client.post(
                "/api/copilot/query",
                data=json.dumps({
                    "query": f"How can I reduce the risk score of invoice {inv_dict['invoice_no']}?"
                }),
                content_type="application/json"
            )
            self.assertEqual(risk_reduction_res.status_code, 200)
            risk_reduction = risk_reduction_res.get_json()["data"]
            self.assertEqual(risk_reduction["category"], "risk_reduction")
            self.assertIn("rerun the analysis", risk_reduction["answer"])

            # Copilot API
            api_res = self.client.post(
                "/api/copilot/query",
                data=json.dumps({"query": "What is the tax discrepancy?", "invoice_no": inv_dict["invoice_no"]}),
                content_type="application/json"
            )
            self.assertEqual(api_res.status_code, 200)
            data = api_res.get_json()
            self.assertEqual(data["status"], "success")

            model_api_res = self.client.post(
                "/api/copilot/query",
                data=json.dumps({
                    "query": "What models detected this anomaly?",
                    "invoice_no": inv_dict["invoice_no"]
                }),
                content_type="application/json"
            )
            self.assertEqual(model_api_res.status_code, 200)
            self.assertEqual(model_api_res.get_json()["data"]["category"], "models")

        unknown_invoice_res = self.client.post(
            "/api/copilot/query",
            data=json.dumps({"query": "Why was invoice INV999999 flagged?"}),
            content_type="application/json"
        )
        self.assertEqual(unknown_invoice_res.status_code, 404)

    # ── 7. AI Explanation & Empirical Evidence ───────────────────────────────
    def test_07_explanation_and_evidence(self):
        """Test evidence extraction and actual vs typical benchmarks."""
        dummy_invoice = {
            "invoice_no": "TEST-001",
            "quantity": 1200.0,
            "unit_price": 95000.0,
            "discount_percentage": 65.0,
            "taxable_value": 250000.0,
            "gst_rate": 18.0,
            "total_tax": 45000.0,
            "cgst": 22500.0,
            "sgst": 22500.0,
            "igst": 0.0,
            "total_amount": 295000.0,
            "isolation_anomaly": 1,
            "isolation_score": 0.65,
            "lof_anomaly": 1,
            "lof_score": 2.1,
            "zscore_anomaly": 1,
            "max_abs_zscore": 4.5,
            "risk_score": 88.5,
            "risk_level": "High Risk",
            "anomaly_type": "Extreme Unit Price"
        }
        exp = get_detailed_explanation(dummy_invoice)
        self.assertEqual(exp["risk_level"], "High Risk")
        self.assertEqual(exp["consensus_count"], 3)
        self.assertGreater(exp["flagged_evidence_count"], 0)

        # Check evidence features
        features = [e["feature"] for e in exp["evidence"]]
        self.assertIn("Unit Price", features)
        self.assertIn("Billed Quantity", features)

    def test_07a_financial_arithmetic_and_missing_values(self):
        valid_invoice = {
            "quantity": 2,
            "unit_price": 50,
            "discount_percentage": 10,
            "discount_amount": 10,
            "taxable_value": 90,
            "gst_rate": 18,
            "cgst": 8.1,
            "sgst": 8.1,
            "igst": 0,
            "cess_rate": 1,
            "cess_amount": 0.9,
            "total_tax": 17.1,
            "total_amount": 107.1,
            "supplier_gstin": "07ABC0001Z1AA",
            "buyer_gstin": "07DEF0002Z2BB",
        }
        results = validate_invoice_arithmetic(valid_invoice)
        self.assertTrue(all(item["status"] == "MATCH" for item in results))

        valid_invoice["total_amount"] = 108.0
        results = validate_invoice_arithmetic(valid_invoice)
        self.assertEqual(
            next(item for item in results if item["check"].startswith("Grand total"))["status"],
            "MISMATCH"
        )

        valid_invoice["unit_price"] = "not-a-number"
        results = validate_invoice_arithmetic(valid_invoice)
        self.assertEqual(
            next(item for item in results if item["check"].startswith("Quantity ×"))["status"],
            "UNAVAILABLE"
        )
        self.assertTrue(any(
            item["status"] == "MISMATCH" and "unit_price" in item["check"]
            for item in results
        ))

    def test_07b_weighted_anomaly_index_and_thresholds(self):
        isolation, lof, zscore, scores = calculate_weighted_anomaly_index(
            [0.1, 0.2, 0.3],
            [1.0, 1.5, 2.0],
            [0.0, 4.0, 8.0],
        )
        self.assertEqual(scores.tolist(), [0.0, 50.0, 100.0])
        self.assertAlmostEqual(isolation[2] + lof[2] + zscore[2], scores[2])
        self.assertEqual(classify_risk_level(39.99), "Low Risk")
        self.assertEqual(classify_risk_level(40.0), "Medium Risk")
        self.assertEqual(classify_risk_level(69.99), "Medium Risk")
        self.assertEqual(classify_risk_level(70.0), "High Risk")

    def test_07c_duplicate_detection_excludes_training_reference(self):
        """Only repeated signatures within the upload should be marked duplicate."""
        reference = pd.read_csv(Path(__file__).with_name("gst_invoices.csv")).iloc[[0]]
        first_upload = reference.copy()
        first_upload["invoice_no"] = "UPLOAD-DUPLICATE-001"
        single_upload = pd.concat([reference, first_upload], ignore_index=True)
        _, prepared, _, _ = _build_features(
            single_upload,
            duplicate_context_start=1,
        )
        self.assertEqual(
            int(prepared.iloc[1]["duplicate_transaction_flag"]),
            0,
        )

        second_upload = first_upload.copy()
        second_upload["invoice_no"] = "UPLOAD-DUPLICATE-002"
        repeated_upload = pd.concat(
            [reference, first_upload, second_upload],
            ignore_index=True,
        )
        _, prepared, _, _ = _build_features(
            repeated_upload,
            duplicate_context_start=1,
        )
        self.assertEqual(
            prepared.iloc[1:]["duplicate_transaction_flag"].tolist(),
            [1, 1],
        )

    def test_07d_severity_is_normal_or_anomaly_level(self):
        """Normal rows get Normal severity; every flagged pattern gets a score tier."""
        upload_response = self.upload_frame(
            self.baseline_frame,
            "severity-regression.csv",
        )
        self.assertEqual(upload_response.status_code, 200)
        analysis = run_anomaly_pipeline(self.baseline_frame)
        expected_levels = {"Normal", "Low Risk", "Medium Risk", "High Risk"}
        self.assertLessEqual(set(analysis["risk_level"]), expected_levels)
        self.assertEqual(
            analysis["risk_level"].eq("Normal").tolist(),
            analysis["anomaly_type"].eq("Normal").tolist(),
        )
        self.assertTrue(analysis["risk_level"].eq("Normal").any())
        self.assertTrue(analysis["risk_level"].ne("Normal").any())
        dashboard = self.client.get("/api/dashboard").get_json()["data"]
        self.assertEqual(
            dashboard["normal_invoices"]
            + dashboard["low_risk"]
            + dashboard["medium_risk"]
            + dashboard["high_risk"],
            len(self.baseline_frame),
        )
        normal_filter = self.client.get(
            "/api/invoices?risk_level=Normal&per_page=100"
        ).get_json()
        self.assertEqual(
            normal_filter["pagination"]["total"],
            dashboard["normal_invoices"],
        )

    # ── 8. MSME Recommendation Action Center ─────────────────────────────────
    def test_08_msme_recommendations(self):
        """Test actionable checklists generated per anomaly type."""
        dummy_tax_mismatch = {
            "anomaly_type": "Tax Calculation Mismatch",
            "risk_level": "High Risk"
        }
        rec = get_recommendations_for_invoice(dummy_tax_mismatch)
        self.assertEqual(rec["workflow_action"], "PRIORITISE_VERIFICATION")
        self.assertGreater(len(rec["actions"]), 0)
        self.assertIn("Tax Calculation Mismatch", rec["actions"][0]["why_recommended"])
        self.assertTrue(all(a.get("why_recommended") and a.get("doc_required") for a in rec["actions"]))

    # ── 9. Compliance Report Generation ──────────────────────────────────────
    def test_09_compliance_report(self):
        """Test report data generation and report view."""
        session = SessionLocal()
        try:
            first_inv = session.query(Invoice, Analysis).join(
                Analysis, Invoice.invoice_no == Analysis.invoice_no
            ).filter(
                Invoice.upload_batch_id == get_active_upload_batch_id(session)
            ).first()
            inv_dict = first_inv[0].to_dict()
            inv_dict.update(first_inv[1].to_dict())
            inv_no = inv_dict["invoice_no"]
        finally:
            session.close()

        rep = generate_invoice_report_data(inv_dict)
        self.assertIn("report_id", rep)
        self.assertIn("recommendations", rep)
        self.assertIn("explanation", rep)

        # View route
        res = self.client.get(f"/invoice/{inv_no}/report")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"GSTGuard AI Compliance Report", res.data)

        # API route
        api_res = self.client.get(f"/api/invoices/{inv_no}/report")
        self.assertEqual(api_res.status_code, 200)

    # ── 10. Centralized Anomaly Pipeline & Demo Analysis ─────────────────────
    def test_10_centralized_pipeline_and_demo(self):
        """Test anomaly service execution and demo analysis endpoint."""
        if not Path(__file__).with_name("sample_upload.csv").exists():
            self.skipTest("Packaged demo CSV is not present.")
        demo_res = self.client.post("/api/demo-analysis")
        self.assertEqual(demo_res.status_code, 200)
        demo_data = demo_res.get_json()
        self.assertEqual(demo_data["status"], "success")
        self.assertEqual(demo_data["processed_count"], 100)
        self.assertIn("summary", demo_data)

    # ── 11. Dashboard & Invoice Details Views ────────────────────────────────
    def test_11_dashboard_and_details_views(self):
        """Test dashboard renders with review queue and invoice details with Risk Story."""
        dash_res = self.client.get("/dashboard")
        self.assertEqual(dash_res.status_code, 200)
        self.assertIn(b"Total Invoices Evaluated", dash_res.data)
        self.assertIn(b"Priority Review Queue", dash_res.data)
        self.assertIn(b"Top Risky Suppliers", dash_res.data)

        session = SessionLocal()
        try:
            first_inv = session.query(Invoice).filter(
                Invoice.upload_batch_id == get_active_upload_batch_id(session)
            ).first()
            inv_no = first_inv.invoice_no
        finally:
            session.close()

        det_res = self.client.get(f"/invoice/{inv_no}")
        self.assertEqual(det_res.status_code, 200)
        for removed_narrative in (
            b"AI Risk Assessment Narrative",
            b"Source provenance and any synthetic ground-truth label",
            b"Peer price/quantity benchmarks are unavailable",
            b"Weighted formula:",
            b"Verification context:",
            b"The unit price was flagged as an outlier by the batch analysis",
        ):
            self.assertNotIn(removed_narrative, det_res.data)
        self.assertIn(b"Invoice Arithmetic &amp; Screening Evidence", det_res.data)
        self.assertIn(b"MSME Compliance Action Center", det_res.data)
        self.assertIn(b"Compliance Investigation Tracker", det_res.data)


if __name__ == "__main__":
    unittest.main()
