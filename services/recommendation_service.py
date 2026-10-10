"""
services/recommendation_service.py
==================================
MSME Compliance Action Center recommendations.
Generates context-aware, practical compliance action checklists based on
the detected anomaly type, model consensus, and statutory GST regulations.
"""

from typing import Dict, Any


# Recommendation patterns mapped by anomaly type
ANOMALY_ACTION_MAP = {
    "Extreme Unit Price": {
        "title": "Unit Price Outlier Verification",
        "description": "The unit price was flagged as an outlier by the batch analysis; no independent category-price benchmark is available.",
        "risk_impact": "A price outlier is a screening signal only. Section 15 concerns value of taxable supply; valuation rules are separately prescribed under the CGST Rules. No compliance conclusion is made here.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Verify Original Purchase Order & Rate Contract",
                "detail": "Cross-check billed rates against the approved vendor purchase agreement or master price list.",
                "doc_required": "Purchase Order (PO), Rate Contract"
            },
            {
                "priority": "HIGH",
                "action": "Inspect Physical Delivery Challan & Goods Receipt Note (GRN)",
                "detail": "Confirm that high-value specialized items were physically received, inspected, and logged in warehouse records.",
                "doc_required": "Signed Goods Receipt Note (GRN), Inspection Sheet"
            },
            {
                "priority": "MEDIUM",
                "action": "Check HSN Code Classification",
                "detail": "Check that the described goods and recorded HSN classification correspond; classification does not by itself set a price limit.",
                "doc_required": "Product specification and applicable HSN classification reference"
            }
        ]
    },
    "Extreme Invoice Amount": {
        "title": "Substantial Invoice Total Reconciliation",
        "description": "The invoice total was flagged for review because of a statistical outlier pattern or a recorded arithmetic discrepancy.",
        "risk_impact": "An unusually large amount or arithmetic discrepancy should be reconciled to source records. Amount alone does not establish an ITC outcome.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Perform Line-Item Summation Check",
                "detail": "Verify that (Taxable Value + CGST + SGST + IGST + Cess) exactly equals Grand Total without unexplained surcharges.",
                "doc_required": "Detailed Itemized Invoice"
            },
            {
                "priority": "HIGH",
                "action": "Reconcile applicable e-Way Bill records",
                "detail": "Where an e-Way bill is applicable, compare its invoice reference, validity, and movement details with the source records.",
                "doc_required": "Applicable e-Way Bill and transport records"
            },
            {
                "priority": "MEDIUM",
                "action": "Validate Bank Payment Trails",
                "detail": "Reconcile the payment trail and review any applicable statutory payment-period treatment against the facts.",
                "doc_required": "Bank Statement / Payment Advice"
            }
        ]
    },
    "Tax Calculation Mismatch": {
        "title": "Statutory Tax Formula Reconciliation",
        "description": "Recorded tax amounts do not reconcile arithmetically with the entered taxable value and rate; confirm the source invoice and applicable treatment.",
        "risk_impact": "A calculated difference is an arithmetic concern. Verify the invoice and applicable tax treatment; this tool does not determine statutory liability or ITC eligibility.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Recalculate and reconcile the tax amounts",
                "detail": "Compare the recorded taxable value, GST tax heads, cess, and grand total with the source invoice.",
                "doc_required": "Source tax invoice and reconciliation worksheet"
            },
            {
                "priority": "HIGH",
                "action": "Compare supplier-reported invoice details",
                "detail": "Check whether the supplier's filed invoice data matches the source document and the recipient's records.",
                "doc_required": "Supplier filing details and recipient reconciliation"
            },
            {
                "priority": "MEDIUM",
                "action": "Request clarification or correction if warranted",
                "detail": "Ask the supplier to explain the observed difference; obtain an amendment only if source review confirms it is needed.",
                "doc_required": "Supplier response and any amended invoice"
            }
        ]
    },
    "Incorrect CGST SGST Split": {
        "title": "Intra-State CGST / SGST Balancing",
        "description": "The recorded CGST/SGST split was flagged for review; first confirm the transaction is intra-state and verify the tax amounts.",
        "risk_impact": "Check the tax-head split against supplier registration, recipient details, place of supply, and source records; the screen is not a legal determination.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Verify Place of Supply vs Supplier Registration",
                "detail": "Confirm that both supplier and recipient states match and that the transaction is indeed intra-state.",
                "doc_required": "GSTIN State Code Verification (First 2 digits)"
            },
            {
                "priority": "HIGH",
                "action": "Reconcile the recorded tax-head split",
                "detail": "Compare the tax components to the invoice and confirmed place-of-supply treatment; request correction only if the source review supports it.",
                "doc_required": "Source invoice, registration details, and any supported correction"
            }
        ]
    },
    "Incorrect IGST": {
        "title": "Inter-State IGST Apportionment Verification",
        "description": "Inter-state IGST calculation mismatch or unexpected IGST levy on what appears to be local supply.",
        "risk_impact": "The apparent tax-head allocation should be checked against registration and place-of-supply records; this screen is not a legal determination.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Validate Place of Supply (Section 10/12 of IGST Act)",
                "detail": "Ascertain whether the goods or services were delivered across state boundaries.",
                "doc_required": "Delivery Address / Lorry Receipt (LR)"
            },
            {
                "priority": "HIGH",
                "action": "Verify GSTR-2B Tax Head Mapping",
                "detail": "Ensure that the invoice appears under Table 4(A)(5) for IGST in GSTR-2B.",
                "doc_required": "GSTR-2B Portal Download"
            }
        ]
    },
    "Duplicate Transaction": {
        "title": "Potential Duplicate Invoice Verification",
        "description": "A supplier, buyer, date, and amount signature resembles another record; compare complete source details before treating the invoices as duplicates.",
        "risk_impact": "A duplicate signature is a screening candidate, not proof of duplicate supply or credit. Compare the underlying source documents.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Compare potential duplicate ledger entries",
                "detail": "Check supplier, invoice number, date, amount, buyer, and line-item details before deciding whether the records duplicate one supply.",
                "doc_required": "Both source invoices and purchase ledger entries"
            },
            {
                "priority": "HIGH",
                "action": "Cross-Verify GSTR-2B Auto-Populated List",
                "detail": "Compare the supplier-reported details with both source invoices and recipient records; a repeated portal entry alone does not establish a duplicate supply.",
                "doc_required": "GSTR-2B Supplier Summary"
            },
            {
                "priority": "MEDIUM",
                "action": "Obtain Cancellation Confirmation from Vendor",
                "detail": "If source review confirms an invoice was issued in error, request documented supplier clarification and the appropriate supporting record.",
                "doc_required": "Vendor Cancellation Letter"
            }
        ]
    },
    "Future Invoice Date": {
        "title": "Post-Dated Invoice Compliance Review",
        "description": "The invoice date falls beyond the explicit analysis cutoff used for this batch; verify the intended reporting period and source record.",
        "risk_impact": "The date is flagged only when an explicit analysis cutoff is supplied. Verify the reporting period and source record before drawing a conclusion.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Verify the applicable reporting period",
                "detail": "Confirm the source invoice date and determine the appropriate recording period under the facts and current rules.",
                "doc_required": "Source invoice and period reconciliation"
            },
            {
                "priority": "MEDIUM",
                "action": "Confirm Actual Date of Physical Dispatch",
                "detail": "Check whether the invoice date was a clerical typo versus actual future supply date.",
                "doc_required": "Transporter Consignment Note"
            }
        ]
    },
    "Unusual Quantity": {
        "title": "Bulk Volume & Logistics Audit",
        "description": "The recorded quantity was flagged as unusual within the analyzed batch; no independent category-volume benchmark is available.",
        "risk_impact": "A volume outlier should be checked against order, receipt, and logistics records; it does not establish fictitious billing.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Inspect Weighbridge & Toll Receipts",
                "detail": "Verify that commercial vehicles had appropriate payload capacity for the recorded quantity.",
                "doc_required": "Weighbridge Slips, Fastag / Toll Receipts"
            },
            {
                "priority": "MEDIUM",
                "action": "Cross-Check Warehouse Inward Capacity",
                "detail": "Confirm warehouse space was available and that goods were logged in inventory records.",
                "doc_required": "Stock Ledger / Bin Card"
            }
        ]
    },
    "Unusual Discount": {
        "title": "Post-Sale / Trade Discount Documentation",
        "description": "The discount was flagged as unusual by the batch analysis; the commercial basis and supporting terms are not available to this screen.",
        "risk_impact": "A large discount is not automatically non-compliant. Review its terms and supporting commercial documents; Section 15 treatment depends on the facts and applicable rules.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Verify Pre-Supply Discount Agreement",
                "detail": "Ensure terms of the discount scheme were documented in the commercial contract prior to supply.",
                "doc_required": "Vendor Commercial Scheme Circular"
            },
            {
                "priority": "MEDIUM",
                "action": "Review any post-supply discount adjustment",
                "detail": "If the source records indicate a post-supply credit note, obtain professional review of its tax and ITC treatment under the applicable rules.",
                "doc_required": "Credit note, underlying agreement, and relevant return records"
            }
        ]
    }
}


DEFAULT_ACTIONS = {
    "title": "Standard Compliance Verification",
    "description": "Routine compliance checklist for MSME GST audit readiness.",
    "risk_impact": "No anomaly category-specific checklist is available. Routine document checks are suggestions, not a compliance opinion or eligibility determination.",
    "actions": [
        {
            "priority": "LOW",
            "action": "Match with Monthly GSTR-2B Statement",
            "detail": "Compare supplier-reported details with the source invoice and recipient records; portal matching alone does not determine eligibility.",
            "doc_required": "Source invoice and applicable supplier-filing reconciliation"
        },
        {
            "priority": "LOW",
            "action": "Confirm Payment within 180 Days",
            "detail": "Reconcile payment dates and obtain advice on any applicable statutory payment-period treatment.",
            "doc_required": "Accounts payable ledger and payment evidence"
        }
    ]
}


def get_recommendations_for_invoice(invoice_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate customized compliance action checklist for an invoice.
    """
    anomaly_type = invoice_data.get("anomaly_type", "Normal")
    risk_level = invoice_data.get("risk_level", "Low Risk")

    source_recommendation = ANOMALY_ACTION_MAP.get(anomaly_type, DEFAULT_ACTIONS)
    rec = {
        **source_recommendation,
        "actions": [dict(action) for action in source_recommendation["actions"]],
    }
    if anomaly_type == "Normal":
        rec["actions"] = rec["actions"][:1]
    for action in rec["actions"]:
        action["why_recommended"] = (
            f"Recommended for the '{anomaly_type}' screening pattern: {action['detail']} "
            "Confirm against primary documents before deciding."
        )

    # Determine recommended workflow action
    if risk_level == "High Risk":
        workflow_action = "PRIORITISE_VERIFICATION"
        workflow_label = "Prioritise document verification"
        workflow_badge = "danger"
    elif risk_level == "Medium Risk":
        workflow_action = "VERIFY_DOCUMENTS"
        workflow_label = "Request Supporting Documents"
        workflow_badge = "warning"
    else:
        workflow_action = "ROUTINE_VERIFICATION"
        workflow_label = "Routine verification"
        workflow_badge = "success"

    return {
        "title": rec["title"],
        "description": rec["description"],
        "risk_impact": rec["risk_impact"],
        "workflow_action": workflow_action,
        "workflow_label": workflow_label,
        "workflow_badge": workflow_badge,
        "actions": rec["actions"]
    }
