"""
routes/report_routes.py
=======================
Compliance reporting, ML model performance metrics, and CSV export functionality.
"""

import io
import pandas as pd
from flask import Blueprint, render_template, jsonify, send_file, request
from database.database import (
    SessionLocal, Invoice, Analysis, get_active_upload_batch_id
)

report_bp = Blueprint("reports", __name__)

@report_bp.route("/reports")
def reports_view():
    """Render the invoice CSV export page."""
    return render_template("export_reports.html")


@report_bp.route("/api/model-performance", methods=["GET"])
def api_model_performance():
    """
    GET /api/model-performance
    Legacy artifacts contain predictions scored on the same data used to fit
    the detectors, so they are not suitable as held-out performance metrics.
    """
    return jsonify({
        "status": "unavailable",
        "metrics_available": False,
        "data": None,
        "message": (
            "Precision, recall, F1, and false-positive rate are withheld until "
            "a reproducible held-out evaluation is available. Existing model "
            "artifacts were evaluated on in-sample predictions."
        )
    }), 200


@report_bp.route("/api/export/csv", methods=["GET"])
def api_export_csv():
    """
    GET /api/export/csv
    Download compliance report as CSV file.
    Supports filtering by risk_level (e.g. ?risk_level=High%20Risk).
    """
    session = SessionLocal()
    try:
        risk_level = request.args.get("risk_level", "").strip()
        active_batch_id = get_active_upload_batch_id(session)
        if active_batch_id is None:
            results = []
        else:
            query = session.query(
                Invoice.invoice_no,
                Invoice.invoice_date,
                Invoice.supplier_id,
                Invoice.supplier_gstin,
                Invoice.buyer_id,
                Invoice.buyer_gstin,
                Invoice.product_category,
                Invoice.quantity,
                Invoice.unit_price,
                Invoice.taxable_value,
                Invoice.gst_rate,
                Invoice.total_tax,
                Invoice.total_amount,
                Analysis.isolation_score,
                Analysis.lof_score,
                Analysis.max_abs_zscore,
                Analysis.risk_score,
                Analysis.risk_level,
                Analysis.anomaly_type,
                Analysis.explanation
            ).join(Analysis, Invoice.invoice_no == Analysis.invoice_no).filter(
                Invoice.upload_batch_id == active_batch_id
            )

            if risk_level:
                query = query.filter(Analysis.risk_level == risk_level)

            results = query.order_by(Analysis.risk_score.desc()).all()

        # Build DataFrame
        columns = [
            "Invoice No", "Date", "Supplier ID", "Supplier GSTIN", "Buyer ID", "Buyer GSTIN",
            "Category", "Quantity", "Unit Price", "Taxable Value", "GST Rate", "Total Tax",
            "Total Amount", "Isolation Forest Score", "LOF Score", "Max Z-Score",
            "Risk Score", "Risk Level", "Anomaly Type", "Explanation"
        ]
        df = pd.DataFrame(results, columns=columns)

        # Write to in-memory buffer
        output = io.BytesIO()
        df.to_csv(output, index=False, encoding="utf-8")
        output.seek(0)

        filename = f"gst_compliance_report_{risk_level.lower().replace(' ', '_') or 'all'}.csv"
        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        session.close()
