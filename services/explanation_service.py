"""
services/explanation_service.py
===============================
AI-Powered Anomaly Explanation Engine for GSTGuard AI.
Transforms raw model outputs, statistical deviations, and invoice parameters into:
  1. Transparent multi-model consensus breakdown
  2. Empirical evidence table (Actual vs Typical MSME Benchmark)
  3. Actionable plain-language compliance narrative
"""

import math
from typing import Dict, Any, List
from services.financial_validation import validate_invoice_arithmetic
from config import Config


def _display_number(value: Any, pattern: str, prefix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if not math.isfinite(number):
        return "Unavailable"
    return f"{prefix}{number:{pattern}}"


def build_evidence_items(invoice_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return arithmetic checks and explicitly unbenchmarked transaction indicators.
    """
    evidence = []
    for field, label, actual, detail in (
        ("unit_price", "Unit Price", _display_number(invoice_data.get("unit_price"), ",.2f", "₹"),
         "No category-specific historical comparison group is available."),
        ("quantity", "Billed Quantity", _display_number(invoice_data.get("quantity"), ",.2f") + " units",
         "No product-specific historical comparison group is available."),
        ("discount_percentage", "Discount Percentage",
         _display_number(invoice_data.get("discount_percentage"), ",.2f") + "%",
         "No contract, scheme, or historical peer benchmark is available."),
    ):
        evidence.append({
            "feature": label,
            "actual": actual,
            "benchmark": "Insufficient comparison data",
            "status": "UNAVAILABLE",
            "status_label": "Requires document comparison",
            "badge_class": "secondary",
            "detail": detail,
        })

    for check in validate_invoice_arithmetic(invoice_data):
        status = check["status"]
        evidence.append({
            "feature": check["check"],
            "actual": check["actual"],
            "benchmark": check["expected"],
            "status": status,
            "status_label": {
                "MATCH": "Arithmetic matches",
                "MISMATCH": "Arithmetic discrepancy",
                "UNAVAILABLE": "Cannot evaluate",
            }[status],
            "badge_class": {
                "MATCH": "success",
                "MISMATCH": "danger",
                "UNAVAILABLE": "secondary",
            }[status],
            "detail": check["detail"],
        })

    return evidence


def build_model_breakdown(invoice_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Construct multi-model consensus and anomaly contribution breakdown.
    """
    if_anom = int(invoice_data.get("isolation_anomaly", 0))
    lof_anom = int(invoice_data.get("lof_anomaly", 0))
    z_anom = int(invoice_data.get("zscore_anomaly", 0))

    if_score = float(invoice_data.get("isolation_score", 0.0))
    lof_score = float(invoice_data.get("lof_score", 0.0))
    max_z = float(invoice_data.get("max_abs_zscore", 0.0))

    models = [
        {
            "name": "Isolation Forest",
            "type": "Tree-Partitioning (Global Structure)",
            "flagged": bool(if_anom),
            "raw_score": f"{if_score:.4f}",
            "weight": f"{Config.WEIGHT_ISOLATION_FOREST:.0%}",
            "contribution": invoice_data.get("isolation_contribution"),
            "assessment": "Identified anomalous tree-depth path" if if_anom else "Within standard feature partition",
            "badge_class": "danger" if if_anom else "success"
        },
        {
            "name": "Local Outlier Factor (LOF)",
            "type": "Density-Based (Local Neighborhood)",
            "flagged": bool(lof_anom),
            "raw_score": f"{lof_score:.4f}",
            "weight": f"{Config.WEIGHT_LOF:.0%}",
            "contribution": invoice_data.get("lof_contribution"),
            "assessment": "Significantly lower density than k-nearest neighbors" if lof_anom else "Consistent with local cluster density",
            "badge_class": "danger" if lof_anom else "success"
        },
        {
            "name": "Statistical Z-Score",
            "type": "Parametric Standard Deviation",
            "flagged": bool(z_anom),
            "raw_score": f"{max_z:.2f} σ",
            "weight": f"{Config.WEIGHT_ZSCORE:.0%}",
            "contribution": invoice_data.get("zscore_contribution"),
            "assessment": f"Exceeds 3.0 standard deviations ({max_z:.1f}σ)" if z_anom else "Within 3.0σ normal distribution",
            "badge_class": "danger" if z_anom else "success"
        }
    ]
    return models


def get_detailed_explanation(invoice_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full explanatory payload for invoice details view, Copilot, and compliance reports.
    """
    evidence = build_evidence_items(invoice_data)
    models = build_model_breakdown(invoice_data)

    flagged_evidence = [e for e in evidence if e["status"] == "MISMATCH"]
    consensus_count = sum(1 for m in models if m["flagged"])

    risk_level = invoice_data.get("risk_level", "Low Risk")
    risk_score = float(invoice_data.get("risk_score", 0.0))
    anomaly_type = invoice_data.get("anomaly_type", "Normal")

    # Generate multi-paragraph synthesized assessment
    if risk_level == "High Risk":
        summary = (
            f"This invoice has a high batch-relative anomaly index of {risk_score:.1f}/100. "
            f"{consensus_count} of 3 detectors flagged it; detector agreement is not proof of fraud. "
            f"Observed pattern: {anomaly_type}. Verify the source documents and tax treatment before "
            f"making a compliance decision."
        )
    elif risk_level == "Medium Risk":
        summary = (
            f"This invoice has a medium batch-relative anomaly index of {risk_score:.1f}/100. "
            f"{consensus_count} of 3 detectors flagged it; this is a screening result, not a fraud finding. "
            f"Observed pattern: {anomaly_type}. Review relevant supporting documents."
        )
    else:
        summary = (
            f"This invoice has a low batch-relative anomaly index of {risk_score:.1f}/100 in this analysis "
            f"batch. This does not establish legal compliance; check the arithmetic findings and documents."
        )

    return {
        "summary": summary,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "anomaly_type": anomaly_type,
        "consensus_count": consensus_count,
        "evidence": evidence,
        "flagged_evidence_count": len(flagged_evidence),
        "models": models
    }
