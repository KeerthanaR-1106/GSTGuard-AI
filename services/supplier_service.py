"""
services/supplier_service.py
============================
Supplier Intelligence & Vendor Risk Profiling Service.
Computes supplier risk metrics, flags high-risk vendors, and provides
vendor scorecards for MSME risk assessment.
"""

from typing import Dict, Any, List
from database.database import get_supplier_metrics, get_supplier_detail


def list_suppliers(page: int = 1, per_page: int = 20, search: str = "", sort_by: str = "risk_score", order: str = "desc") -> Dict[str, Any]:
    """
    Paginated supplier directory with risk scores and transaction statistics.
    """
    return get_supplier_metrics(
        page=page,
        per_page=per_page,
        search=search,
        sort_by=sort_by,
        order=order
    )


def get_supplier_profile(supplier_id: str) -> Dict[str, Any]:
    """
    Full supplier profile including risk score, total transactions,
    suspicious invoice history, and risk breakdown.
    """
    data = get_supplier_detail(supplier_id)
    if not data:
        return None

    # Derive additional risk insights
    tot = data["total_invoices"]
    high = data["high_risk_count"]
    med = data["medium_risk_count"]
    suspicious_pct = round((high + med) / tot * 100, 1) if tot > 0 else 0.0

    if data["supplier_risk_score"] >= 70.0:
        recommendation = "CRITICAL: Halt future credit purchases. Initiate comprehensive audit of historical invoices."
        recommendation_badge = "danger"
    elif data["supplier_risk_score"] >= 40.0:
        recommendation = "CAUTION: Mandatory documentary verification for all consignments exceeding ₹50,000."
        recommendation_badge = "warning"
    else:
        recommendation = "APPROVED: Standard vendor profile. Routine monthly GSTR-2B matching recommended."
        recommendation_badge = "success"

    data["suspicious_percentage"] = suspicious_pct
    data["recommendation"] = recommendation
    data["recommendation_badge"] = recommendation_badge

    return data


def get_top_risky_suppliers(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Top risky suppliers for dashboard alert widget.
    """
    res = get_supplier_metrics(page=1, per_page=limit, sort_by="risk_score", order="desc")
    return res["items"]
