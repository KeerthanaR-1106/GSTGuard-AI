"""
routes/supplier_routes.py
=========================
Supplier Risk Intelligence routes and APIs for GSTGuard AI.
Provides vendor scorecards, risk level aggregations, and supplier profile views.
"""

from flask import Blueprint, render_template, request, jsonify, abort
from services.supplier_service import list_suppliers, get_supplier_profile

supplier_bp = Blueprint("suppliers", __name__)


# ── FRONTEND TEMPLATE ROUTES ──────────────────────────────────────────────────

@supplier_bp.route("/suppliers")
def suppliers_view():
    """Supplier directory view with risk scores and transaction statistics."""
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "", type=str)
    sort_by = request.args.get("sort_by", "risk_score", type=str)
    order = request.args.get("order", "desc", type=str)

    data = list_suppliers(page=page, per_page=20, search=search, sort_by=sort_by, order=order)
    return render_template(
        "suppliers.html",
        suppliers=data["items"],
        pagination=data,
        search=search,
        sort_by=sort_by,
        order=order
    )


@supplier_bp.route("/suppliers/<supplier_id>")
def supplier_detail_view(supplier_id):
    """Detailed vendor scorecard and historical invoice risk audit."""
    profile = get_supplier_profile(supplier_id)
    if not profile:
        abort(404, description=f"Supplier '{supplier_id}' not found in compliance database.")
    return render_template("supplier_details.html", supplier=profile)


# ── REST API ENDPOINTS ────────────────────────────────────────────────────────

@supplier_bp.route("/api/suppliers", methods=["GET"])
def api_get_suppliers():
    """
    GET /api/suppliers
    Query parameters:
      - page (int, default=1)
      - per_page (int, default=20)
      - search (string)
      - sort_by (risk_score, high_risk_count, total_invoices, total_taxable_value)
      - order (desc, asc)
    """
    try:
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
        search = request.args.get("search", "", type=str)
        sort_by = request.args.get("sort_by", "risk_score", type=str)
        order = request.args.get("order", "desc", type=str)

        per_page = min(max(per_page, 5), 100)

        data = list_suppliers(
            page=page,
            per_page=per_page,
            search=search,
            sort_by=sort_by,
            order=order
        )
        return jsonify({
            "status": "success",
            "pagination": {
                "page": data["page"],
                "per_page": data["per_page"],
                "total": data["total"],
                "pages": data["pages"]
            },
            "data": data["items"]
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@supplier_bp.route("/api/suppliers/<supplier_id>", methods=["GET"])
def api_get_supplier_detail(supplier_id):
    """
    GET /api/suppliers/<supplier_id>
    Returns supplier scorecard, metrics, and associated invoices.
    """
    try:
        profile = get_supplier_profile(supplier_id)
        if not profile:
            return jsonify({
                "status": "error",
                "message": f"Supplier '{supplier_id}' not found."
            }), 404

        return jsonify({
            "status": "success",
            "data": profile
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
