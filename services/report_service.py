"""
services/report_service.py
==========================
Compliance Report Generation Service for GSTGuard AI.
Produces structured compliance audit reports with executive summary,
evidence verification, model breakdown, and statutory action checklists.
"""

from typing import Dict, Any
from datetime import datetime, timezone
from services.explanation_service import get_detailed_explanation
from services.recommendation_service import get_recommendations_for_invoice


def generate_invoice_report_data(invoice_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct all report sections for an invoice audit report.
    """
    exp = get_detailed_explanation(invoice_data)
    recs = get_recommendations_for_invoice(invoice_data)

    report_id = f"GSTGUARD-REP-{invoice_data.get('invoice_no')}-{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    return {
        "report_id": report_id,
        "generated_at": datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
        "invoice": invoice_data,
        "explanation": exp,
        "recommendations": recs,
        "auditor": invoice_data.get("investigation", {}).get("assigned_to", "Compliance Review Team"),
        "status": invoice_data.get("investigation", {}).get("status", "NEW"),
        "notes": invoice_data.get("investigation", {}).get("notes", "")
    }
