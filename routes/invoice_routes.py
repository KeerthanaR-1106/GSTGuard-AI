"""
routes/invoice_routes.py
========================
Invoice listing, search, filtering, detailed audit drill-down views,
and corresponding REST APIs.
"""

from flask import Blueprint, render_template, request, jsonify, abort, session, current_app
from database.database import (
    get_invoices, get_suspicious_invoices, get_invoice_by_no,
    get_or_create_investigation, update_investigation, get_review_queue,
    get_investigation_history
)
from services.explanation_service import get_detailed_explanation
from services.recommendation_service import get_recommendations_for_invoice
from services.report_service import generate_invoice_report_data

invoice_bp = Blueprint("invoices", __name__)


# ── FRONTEND TEMPLATE ROUTES ──────────────────────────────────────────────────

@invoice_bp.route("/invoices")
def invoices_view():
    """All invoices view with pagination, search, and risk filtering."""
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "", type=str)
    risk_level = request.args.get("risk_level", "", type=str)
    anomaly_type = request.args.get("anomaly_type", "", type=str)

    data = get_invoices(page=page, per_page=20, search=search, risk_level=risk_level, anomaly_type=anomaly_type)
    return render_template(
        "invoices.html",
        invoices=data["items"],
        pagination=data,
        search=search,
        risk_level=risk_level,
        anomaly_type=anomaly_type
    )


@invoice_bp.route("/suspicious")
def suspicious_view():
    """Filtered view of suspicious invoices (Medium and High Risk only)."""
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "", type=str)
    risk_level = request.args.get("risk_level", "", type=str)
    anomaly_type = request.args.get("anomaly_type", "", type=str)

    data = get_suspicious_invoices(page=page, per_page=20, search=search, risk_level=risk_level, anomaly_type=anomaly_type)
    return render_template(
        "suspicious.html",
        invoices=data["items"],
        pagination=data,
        search=search,
        risk_level=risk_level,
        anomaly_type=anomaly_type
    )


@invoice_bp.route("/review-queue")
def review_queue_view():
    """Priority review queue for compliance officers."""
    queue = get_review_queue(limit=50)
    return render_template("review_queue.html", queue=queue)


@invoice_bp.route("/invoice/<path:invoice_no>/report")
@invoice_bp.route("/invoice/<invoice_no>/report")
def invoice_report_view(invoice_no):
    """Render printable compliance audit report."""
    invoice = get_invoice_by_no(invoice_no)
    if not invoice:
        abort(404, description=f"Invoice '{invoice_no}' not found for report generation.")

    report_data = generate_invoice_report_data(invoice)
    return render_template("invoice_report.html", report=report_data)


@invoice_bp.route("/invoice/<path:invoice_no>")
@invoice_bp.route("/invoice/<invoice_no>")
def invoice_detail_view(invoice_no):
    """Detailed audit view of an individual invoice showing raw line-items and model outputs."""
    invoice = get_invoice_by_no(invoice_no)
    if not invoice:
        abort(404, description=f"Invoice '{invoice_no}' was not found in compliance database.")

    # Enrich with intelligence services
    explanation_data = get_detailed_explanation(invoice)
    recommendations_data = get_recommendations_for_invoice(invoice)

    return render_template(
        "invoice_details.html",
        invoice=invoice,
        explanation=explanation_data,
        recommendations=recommendations_data,
        back_to_queue=request.args.get("return_to") == "review-queue",
    )


# ── REST API ENDPOINTS ────────────────────────────────────────────────────────

@invoice_bp.route("/api/invoices", methods=["GET"])
def api_get_invoices():
    """
    GET /api/invoices
    Query parameters:
      - page (int, default=1)
      - per_page (int, default=20)
      - search (string)
      - risk_level (Normal, Low Risk, Medium Risk, High Risk)
      - anomaly_type (string)
    """
    try:
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
        search = request.args.get("search", "", type=str)
        risk_level = request.args.get("risk_level", "", type=str)
        anomaly_type = request.args.get("anomaly_type", "", type=str)

        per_page = min(max(per_page, 5), 100)

        result = get_invoices(
            page=page,
            per_page=per_page,
            search=search,
            risk_level=risk_level,
            anomaly_type=anomaly_type
        )
        return jsonify({
            "status": "success",
            "pagination": {
                "page": result["page"],
                "per_page": result["per_page"],
                "total": result["total"],
                "pages": result["pages"]
            },
            "data": result["items"]
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@invoice_bp.route("/api/invoices/<path:invoice_no>/report", methods=["GET"])
@invoice_bp.route("/api/invoices/<invoice_no>/report", methods=["GET"])
def api_get_invoice_report(invoice_no):
    """
    GET /api/invoices/<invoice_no>/report
    Returns structured compliance audit report JSON.
    """
    try:
        invoice = get_invoice_by_no(invoice_no)
        if not invoice:
            return jsonify({
                "status": "error",
                "message": f"Invoice '{invoice_no}' not found."
            }), 404

        report_data = generate_invoice_report_data(invoice)
        return jsonify({
            "status": "success",
            "report": report_data
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@invoice_bp.route("/api/invoices/<path:invoice_no>", methods=["GET"])
@invoice_bp.route("/api/invoices/<invoice_no>", methods=["GET"])
def api_get_invoice_detail(invoice_no):
    """
    GET /api/invoices/<invoice_no>
    Returns complete invoice transaction fields, Isolation Forest, LOF,
    Z-Score outputs, ensemble risk score, risk level, explanation,
    and investigation status.
    """
    try:
        invoice = get_invoice_by_no(invoice_no)
        if not invoice:
            return jsonify({
                "status": "error",
                "message": f"Invoice '{invoice_no}' not found."
            }), 404

        explanation_data = get_detailed_explanation(invoice)
        recommendations_data = get_recommendations_for_invoice(invoice)

        return jsonify({
            "status": "success",
            "data": invoice,
            "explanation": explanation_data,
            "recommendations": recommendations_data
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@invoice_bp.route("/api/suspicious-invoices", methods=["GET"])
def api_get_suspicious_invoices():
    """
    GET /api/suspicious-invoices
    Returns only transactions classified as Medium Risk (40-69) or High Risk (70-100).
    """
    try:
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
        search = request.args.get("search", "", type=str)
        risk_level = request.args.get("risk_level", "", type=str)
        anomaly_type = request.args.get("anomaly_type", "", type=str)

        per_page = min(max(per_page, 5), 100)

        result = get_suspicious_invoices(
            page=page,
            per_page=per_page,
            search=search,
            risk_level=risk_level,
            anomaly_type=anomaly_type
        )
        return jsonify({
            "status": "success",
            "pagination": {
                "page": result["page"],
                "per_page": result["per_page"],
                "total": result["total"],
                "pages": result["pages"]
            },
            "data": result["items"]
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@invoice_bp.route("/api/review-queue", methods=["GET"])
def api_get_review_queue():
    """
    GET /api/review-queue
    Returns ranked priority review queue.
    """
    try:
        limit = request.args.get("limit", 50, type=int)
        limit = min(max(limit, 5), 100)
        queue = get_review_queue(limit=limit)
        return jsonify({
            "status": "success",
            "count": len(queue),
            "data": queue
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@invoice_bp.route("/api/investigations/<path:invoice_no>", methods=["GET", "POST", "PATCH"])
@invoice_bp.route("/api/investigations/<invoice_no>", methods=["GET", "POST", "PATCH"])
def api_investigation(invoice_no):
    """
    GET /api/investigations/<invoice_no> -> Fetch status & notes
    POST/PATCH /api/investigations/<invoice_no> -> Update status, notes, assigned_to
    """
    try:
        if request.method == "GET":
            if not get_invoice_by_no(invoice_no):
                return jsonify({"status": "error", "message": "Invoice not found."}), 404
            inv = get_or_create_investigation(invoice_no)
            return jsonify({
                "status": "success",
                "data": inv,
                "history": get_investigation_history(invoice_no)
            }), 200

        if not session.get("logged_in") or not session.get("user"):
            return jsonify({
                "status": "error",
                "message": "Sign in is required to update an investigation."
            }), 401

        # POST / PATCH update
        data = request.get_json(silent=True) or request.form or {}
        if not hasattr(data, "get"):
            return jsonify({"status": "error", "message": "Request body must be an object."}), 400
        new_status = data.get("status")
        notes = data.get("notes")
        assigned_to = data.get("assigned_to")
        if any(value is not None and not isinstance(value, str)
               for value in (new_status, notes, assigned_to)):
            return jsonify({"status": "error", "message": "Status, notes, and reviewer must be text."}), 400

        updated = update_investigation(
            invoice_no=invoice_no,
            status=new_status,
            notes=notes,
            assigned_to=assigned_to,
            reviewer=str(session["user"])
        )
        return jsonify({
            "status": "success",
            "message": f"Investigation for invoice {invoice_no} updated.",
            "data": updated,
            "history": get_investigation_history(invoice_no)
        }), 200
    except LookupError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        current_app.logger.exception("Failed to process investigation for invoice %s", invoice_no)
        return jsonify({"status": "error", "message": str(e)}), 500
