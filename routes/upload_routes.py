"""
routes/upload_routes.py
=======================
Handles CSV upload, validation, full ML inference pipeline execution,
and SQLite database insertion:
  User uploads CSV
  ↓
  Validate CSV (format, columns, non-empty)
  ↓
  Preprocess & generate 43 ML features
  ↓
  Run Isolation Forest inference
  ↓
  Run Local Outlier Factor inference
  ↓
  Run Statistical Z-Score inference
  ↓
  Compute Ensemble Risk Score & Risk Level
  ↓
  Generate compliance explanations
  ↓
  Insert transactions & analysis into SQLite database
  ↓
  Return status & metrics to UI
"""

import os
import sqlite3
import tempfile
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
from config import Config
from database.database import DB_PATH
from services.anomaly_service import run_anomaly_pipeline

upload_bp = Blueprint("upload", __name__)

REQUIRED_CSV_COLUMNS = [
    "invoice_no", "invoice_date", "supplier_id", "supplier_gstin",
    "buyer_id", "buyer_gstin", "product_category", "hsn_code",
    "quantity", "unit_price", "discount_percentage", "discount_amount",
    "taxable_value", "gst_rate", "cgst", "sgst", "igst",
    "cess_rate", "cess_amount", "total_tax", "total_amount",
    "payment_method"
]


def _find_existing_invoice_ids(invoice_ids):
    existing_ids = set()
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        for start in range(0, len(invoice_ids), 900):
            batch = invoice_ids[start:start + 900]
            placeholders = ", ".join("?" for _ in batch)
            rows = cursor.execute(
                f"SELECT invoice_no FROM invoices WHERE invoice_no IN ({placeholders})",
                batch
            ).fetchall()
            existing_ids.update(row[0] for row in rows)
    finally:
        connection.close()
    return existing_ids


def _store_invoice_analysis(df_raw, analysis_records):
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        invoice_tuples = [tuple(row) for row in df_raw[REQUIRED_CSV_COLUMNS].to_numpy()]
        cursor.executemany(f"""
            INSERT INTO invoices ({', '.join(REQUIRED_CSV_COLUMNS)})
            VALUES ({', '.join(['?'] * len(REQUIRED_CSV_COLUMNS))});
        """, invoice_tuples)

        analysis_columns = list(analysis_records.columns)
        analysis_tuples = [tuple(row) for row in analysis_records.to_numpy()]
        cursor.executemany(f"""
            INSERT INTO analysis ({', '.join(analysis_columns)})
            VALUES ({', '.join(['?'] * len(analysis_columns))});
        """, analysis_tuples)
        connection.commit()
    except sqlite3.IntegrityError:
        connection.rollback()
        raise
    finally:
        connection.close()


@upload_bp.route("/upload")
def upload_view():
    """Render file upload page."""
    return render_template("upload.html")


@upload_bp.route("/api/upload", methods=["POST"])
def api_upload_csv():
    """
    POST /api/upload
    Receives multipart/form-data containing 'file'.
    Executes complete end-to-end anomaly detection pipeline and stores results in SQLite.
    """
    # 1. Validate file presence
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file part in request. Please select a valid CSV file."
        }), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "No file selected. Please choose a CSV file to upload."
        }), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({
            "status": "error",
            "message": "Invalid file type. Only CSV (.csv) files are supported."
        }), 400

    # 2. Save file temporarily
    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({
            "status": "error",
            "message": "The uploaded filename is invalid."
        }), 400
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        prefix="invoice-upload-", suffix=".csv", dir=Config.UPLOAD_FOLDER
    )
    os.close(descriptor)

    try:
        file.save(temp_path)
        # 3. Read and validate structure
        try:
            df_raw = pd.read_csv(temp_path)
        except pd.errors.EmptyDataError:
            return jsonify({
                "status": "error",
                "message": "The uploaded CSV file is empty. Please upload a file containing invoice data."
            }), 400
        except Exception as e:
            current_app.logger.exception("Invoice CSV analysis failed")
            return jsonify({
                "status": "error",
                "message": f"Malformed CSV file could not be parsed: {str(e)}"
            }), 400

        if df_raw.empty:
            return jsonify({
                "status": "error",
                "message": "The uploaded CSV file is empty. Please upload a file containing invoice data."
            }), 400
        if len(df_raw) < 2:
            return jsonify({
                "status": "error",
                "message": "Upload at least two invoices so the anomaly models can form a comparison batch."
            }), 400

        # Validate required columns
        missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in df_raw.columns]
        if missing_cols:
            return jsonify({
                "status": "error",
                "message": f"Missing required columns ({len(missing_cols)}): {', '.join(missing_cols)}"
            }), 400

        invoice_ids = df_raw["invoice_no"]
        if invoice_ids.isna().any() or invoice_ids.astype(str).str.strip().eq("").any():
            return jsonify({
                "status": "error",
                "message": "Every row must contain a non-empty invoice_no."
            }), 400
        if invoice_ids.duplicated().any():
            return jsonify({
                "status": "error",
                "message": "Duplicate invoice_no values were found in the upload; each row must have a unique identifier."
            }), 400
        existing_ids = _find_existing_invoice_ids(invoice_ids.astype(str).tolist())
        if existing_ids:
            conflicts = ", ".join(sorted(existing_ids)[:10])
            return jsonify({
                "status": "error",
                "message": (
                    f"{len(existing_ids)} invoice_no value(s) already exist in the ledger "
                    f"and were not overwritten. Conflicting IDs include: {conflicts}"
                )
            }), 409

        # Handle numeric validation and date parsing
        df_raw["invoice_date"] = pd.to_datetime(
            df_raw["invoice_date"], errors="coerce", format="mixed"
        )
        if df_raw["invoice_date"].isnull().any():
            return jsonify({
                "status": "error",
                "message": "One or more invoice_date values are missing or invalid. Correct the source CSV and upload again."
            }), 400

        num_cols = ["quantity", "unit_price", "discount_percentage", "discount_amount",
                    "taxable_value", "gst_rate", "cgst", "sgst", "igst",
                    "cess_rate", "cess_amount", "total_tax", "total_amount"]
        for c in num_cols:
            converted = pd.to_numeric(df_raw[c], errors="coerce")
            if converted.isnull().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
                return jsonify({
                    "status": "error",
                    "message": f"Column '{c}' contains a missing, malformed, or non-finite numeric value."
                }), 400
            if (converted < 0).any():
                return jsonify({
                    "status": "error",
                    "message": f"Column '{c}' contains a negative value. Source values were not modified."
                }), 400
            df_raw[c] = converted
        if ((df_raw["discount_percentage"] > 100) |
                (df_raw["gst_rate"] > 100) | (df_raw["cess_rate"] > 100)).any():
            return jsonify({
                "status": "error",
                "message": "Discount, GST, and cess rates must be between 0 and 100 percent."
            }), 400

        # 4. End-to-End ML Pipeline Execution via Anomaly Service
        n_samples = len(df_raw)
        analysis_records = run_anomaly_pipeline(df_raw)
        risk_levels = analysis_records["risk_level"].tolist()
        risk_score = analysis_records["risk_score"].values
        df_raw["invoice_date"] = pd.to_datetime(df_raw["invoice_date"]).dt.strftime("%Y-%m-%d")


        _store_invoice_analysis(df_raw, analysis_records)

        # Summary counts
        lvl_series = pd.Series(risk_levels)
        high_cnt = int((lvl_series == "High Risk").sum())
        med_cnt = int((lvl_series == "Medium Risk").sum())
        low_cnt = int((lvl_series == "Low Risk").sum())

        return jsonify({
            "status": "success",
            "message": f"Successfully processed {n_samples:,} invoices and updated SQLite compliance database.",
            "processed_count": n_samples,
            "summary": {
                "high_risk": high_cnt,
                "medium_risk": med_cnt,
                "low_risk": low_cnt,
                "suspicious": high_cnt + med_cnt,
                "avg_risk_score": round(float(np.mean(risk_score)), 2)
            }
        }), 200

    except sqlite3.IntegrityError:
        current_app.logger.warning("Rejected upload after a concurrent invoice identifier conflict.")
        return jsonify({
            "status": "error",
            "message": "One or more invoice IDs were added by another request; no existing invoice data was overwritten. Please review and retry."
        }), 409
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Internal pipeline execution error: {str(e)}"
        }), 500
    finally:
        # Cleanup temporary uploaded file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                current_app.logger.warning("Unable to remove temporary upload file %s", temp_path)


@upload_bp.route("/api/demo-analysis", methods=["POST"])
def api_demo_analysis():
    """
    POST /api/demo-analysis
    Executes live pipeline on packaged sample dataset for hackathon judges
    without requiring manual file upload.
    """
    try:
        sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_upload.csv")
        if not os.path.exists(sample_path):
            return jsonify({"status": "error", "message": "Packaged sample upload file not found."}), 404

        df_raw = pd.read_csv(sample_path)
        n_samples = len(df_raw)
        existing_ids = _find_existing_invoice_ids(df_raw["invoice_no"].astype(str).tolist())
        if existing_ids:
            conflicts = ", ".join(sorted(existing_ids)[:10])
            return jsonify({
                "status": "error",
                "message": (
                    "The sample invoices already exist in the ledger and were not overwritten. "
                    f"Conflicting IDs include: {conflicts}"
                )
            }), 409

        # Run pipeline
        analysis_records = run_anomaly_pipeline(df_raw)
        risk_levels = analysis_records["risk_level"].tolist()
        risk_score = analysis_records["risk_score"].values
        df_raw["invoice_date"] = pd.to_datetime(df_raw["invoice_date"]).dt.strftime("%Y-%m-%d")

        _store_invoice_analysis(df_raw, analysis_records)

        lvl_series = pd.Series(risk_levels)
        high_cnt = int((lvl_series == "High Risk").sum())
        med_cnt = int((lvl_series == "Medium Risk").sum())
        low_cnt = int((lvl_series == "Low Risk").sum())

        return jsonify({
            "status": "success",
            "message": f"Successfully evaluated {n_samples} demo invoices through all 3 unsupervised models.",
            "processed_count": n_samples,
            "summary": {
                "high_risk": high_cnt,
                "medium_risk": med_cnt,
                "low_risk": low_cnt,
                "suspicious": high_cnt + med_cnt,
                "avg_risk_score": round(float(np.mean(risk_score)), 2)
            }
        }), 200
    except sqlite3.IntegrityError:
        current_app.logger.warning("Rejected demo analysis after a concurrent invoice identifier conflict.")
        return jsonify({
            "status": "error",
            "message": "The sample invoice IDs were added by another request; no existing invoice data was overwritten."
        }), 409
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
