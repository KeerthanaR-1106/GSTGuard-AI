"""
routes/copilot_routes.py
========================
Compliance Copilot routes and APIs for GSTGuard AI.
Provides deterministic, data-grounded AI-assisted compliance guidance.
"""

import re
from flask import Blueprint, render_template, request, jsonify
from database.database import get_invoice_by_no, get_supplier_detail
from services.copilot_service import query_copilot

copilot_bp = Blueprint("copilot", __name__)


@copilot_bp.route("/copilot")
def copilot_view():
    """Dedicated interactive Compliance Copilot workspace."""
    invoice_no = request.args.get("invoice_no", "").strip()
    invoice_data = None
    if invoice_no:
        invoice_data = get_invoice_by_no(invoice_no)

    return render_template("copilot.html", invoice_no=invoice_no, invoice=invoice_data)


@copilot_bp.route("/api/copilot/query", methods=["POST"])
def api_copilot_query():
    """
    POST /api/copilot/query
    Request Body:
      - query (string, required)
      - invoice_no (string, optional)
      - supplier_id (string, optional)
    """
    try:
        data = request.get_json(silent=True) or request.form or {}
        query_text = str(data.get("query") or "").strip()

        if not query_text:
            return jsonify({
                "status": "error",
                "message": "Query parameter cannot be empty."
            }), 400

        invoice_no = str(data.get("invoice_no") or "").strip()
        if not invoice_no:
            invoice_match = re.search(r"\bINV\d+\b", query_text, re.IGNORECASE)
            if invoice_match:
                invoice_no = invoice_match.group(0).upper()
        supplier_id = str(data.get("supplier_id") or "").strip()
        if not supplier_id:
            supplier_match = re.search(r"\bSUP[A-Z0-9-]+\b", query_text, re.IGNORECASE)
            if supplier_match:
                supplier_id = supplier_match.group(0).upper()

        invoice_data = None
        if invoice_no:
            invoice_data = get_invoice_by_no(invoice_no)
            if not invoice_data:
                return jsonify({
                    "status": "error",
                    "message": f"Invoice {invoice_no} was not found. Check the invoice number and try again."
                }), 404

        if invoice_data and not supplier_id:
            supplier_id = invoice_data.get("supplier_id", "")

        supplier_data = None
        if supplier_id:
            supplier_data = get_supplier_detail(supplier_id)

        result = query_copilot(
            query_text=query_text,
            invoice_data=invoice_data,
            supplier_data=supplier_data
        )

        return jsonify({
            "status": "success",
            "query": query_text,
            "invoice_no": invoice_no or None,
            "data": result
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Copilot query failed: {str(e)}"
        }), 500
