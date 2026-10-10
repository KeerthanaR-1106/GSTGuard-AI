"""
services/copilot_service.py
===========================
Compliance Copilot: Deterministic, data-grounded assistant for MSME GST compliance.
Executes locally without external API dependencies. Answers questions by synthesizing
actual transaction line items, ML model outputs, empirical benchmarks, and GST rules.
"""

import re
from collections import Counter
from typing import Dict, Any, Optional
from services.explanation_service import get_detailed_explanation
from services.recommendation_service import get_recommendations_for_invoice


def _supplier_summary(supplier_data: Dict[str, Any], invoice_no: Optional[str] = None) -> Dict[str, str]:
    supplier_id = supplier_data.get("supplier_id", "Unknown")
    supplier_gstin = supplier_data.get("supplier_gstin", "Unknown")
    invoice_records = supplier_data.get("invoices", [])
    anomaly_counts = Counter()

    for invoice in invoice_records:
        anomaly_type = invoice.get("anomaly_type") or "Normal"
        if anomaly_type == "Normal" and invoice.get("risk_level") in ("Medium Risk", "High Risk"):
            anomaly_type = "Statistical Model Outlier"
        anomaly_counts[anomaly_type] += 1

    anomaly_summary = "\n".join(
        f"- **{name}:** {count} invoice(s)"
        for name, count in anomaly_counts.most_common()
    ) or "- No invoice pattern data is available."

    context = f" for invoice `{invoice_no}`" if invoice_no else ""
    company_name = supplier_data.get("company_name")
    company_line = (
        f"- **Company name:** {company_name}"
        if company_name else
        "- **Company name:** Not recorded in this dataset; the supplier is identified by its supplier ID and GSTIN."
    )
    answer = (
        f"### Supplier Summary{context}\n\n"
        f"{company_line}\n"
        f"- **Supplier ID:** `{supplier_id}`\n"
        f"- **GSTIN:** `{supplier_gstin}`\n"
        f"- **Invoices in this dataset:** {supplier_data.get('total_invoices', len(invoice_records))}\n"
        f"- **Risk levels:** High {supplier_data.get('high_risk_count', 0)}, "
        f"Medium {supplier_data.get('medium_risk_count', 0)}, "
        f"Low {supplier_data.get('low_risk_count', 0)}\n\n"
        f"**Anomaly patterns across its invoices:**\n{anomaly_summary}"
    )
    return {"answer": answer, "confidence": "High (Data-grounded)", "category": "supplier_summary"}


def query_copilot(query_text: str, invoice_data: Optional[Dict[str, Any]] = None, supplier_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a user query in natural language and return a structured, data-grounded answer.
    """
    q = query_text.lower().strip()
    
    # ── CASE 1: Invoice-Specific Context ─────────────────────────────────────
    if invoice_data:
        inv_no = invoice_data.get("invoice_no", "Unknown")
        risk_level = invoice_data.get("risk_level", "Low Risk")
        risk_score = float(invoice_data.get("risk_score", 0.0))
        anomaly_type = invoice_data.get("anomaly_type", "Normal")
        supplier_id = invoice_data.get("supplier_id", "Unknown")
        supplier_gstin = invoice_data.get("supplier_gstin", "Unknown")
        taxable_val = float(invoice_data.get("taxable_value", 0.0))
        total_tax = float(invoice_data.get("total_tax", 0.0))
        gst_rate = float(invoice_data.get("gst_rate", 0.0))

        if any(term in q for term in ["company", "supplier", "vendor", "how many invoices", "produced", "issued"]):
            if supplier_data:
                return _supplier_summary(supplier_data, inv_no)
            return {
                "answer": (
                    f"The invoice lists supplier `{supplier_id}` with GSTIN `{supplier_gstin}`, "
                    "but no supplier invoice history is available for a count or anomaly breakdown."
                ),
                "confidence": "Limited (Supplier data unavailable)",
                "category": "supplier_summary"
            }

        if any(term in q for term in [
            "reduce risk", "lower risk", "reduce the risk score", "lower the risk score",
            "reduce score", "lower score", "improve risk score", "bring the risk score down"
        ]):
            recommendations = get_recommendations_for_invoice(invoice_data)
            specific_actions = recommendations.get("actions", [])
            if recommendations.get("title") == "Standard Compliance Verification":
                action_text = (
                    "- Reconcile invoice line items, taxable value, GST rate, and tax components against the purchase order.\n"
                    "- Verify the supplier filing and invoice match in GSTR-2B.\n"
                    "- Confirm delivery evidence, check for duplicate booking, and retain payment records."
                )
            else:
                action_text = "\n".join(
                    f"- **{item['action']}:** {item['detail']}"
                    for item in specific_actions[:3]
                )
            answer = (
                f"### Risk Reduction Plan for Invoice `{inv_no}`\n\n"
                f"Current score: **{risk_score:.1f}/100 ({risk_level})**; detected pattern: **{anomaly_type}**.\n\n"
                "The score should not be changed manually. Resolve and document the underlying issue, "
                "then rerun the analysis to get an updated score. Suggested checks:\n\n"
                f"{action_text}\n"
                "- Confirm the corrected source documents and supporting records are retained."
            )
            return {"answer": answer, "confidence": "High (Data-grounded)", "category": "risk_reduction"}

        # Handle explicit model questions before anomaly/explanation keywords.
        if any(w in q for w in ["model", "algorithm", "isolation", "lof", "zscore", "consensus"]):
            exp = get_detailed_explanation(invoice_data)
            model_cards = "\n".join([
                f"- **{m['name']}** ({m['type']}): Flagged = **{'YES' if m['flagged'] else 'NO'}** "
                f"(raw model output: `{m['raw_score']}`, weight: {m['weight']}, "
                f"score contribution: {m['contribution'] if m['contribution'] is not None else 'unavailable'} points). "
                f"*{m['assessment']}*"
                for m in exp["models"]
            ])
            answer = (
                f"### Model Evaluation Breakdown for `{inv_no}`\n\n"
                f"{model_cards}\n\n"
                f"**Score:** Batch-relative 0–100 anomaly index. Each detector's displayed score contribution "
                f"is added directly using its configured weight; detector agreement does not prove fraud. "
                f"Category thresholds: Low <40, Medium 40–69.99, High ≥70. Current index: **{risk_score:.1f}/100**."
            )
            return {
                "answer": answer,
                "confidence": "High (Data-grounded)",
                "category": "models"
            }

        # 1. Why was it flagged? / Why is it risky?
        if any(w in q for w in ["why", "reason", "flag", "flagged", "anomaly", "suspicious", "detect"]):
            exp = get_detailed_explanation(invoice_data)
            flagged_models = [m["name"] for m in exp["models"] if m["flagged"]]
            m_text = ", ".join(flagged_models) if flagged_models else "None (Low Risk transaction)"

            evidence_bullets = "\n".join([
                f"- **{e['feature']}**: Actual `{e['actual']}`; reference `{e['benchmark']}` — *{e['detail']}*"
                for e in exp["evidence"] if e["status"] == "MISMATCH"
            ])
            if not evidence_bullets:
                evidence_bullets = "- No arithmetic mismatch was found in the evaluated fields; comparison benchmarks may be unavailable."

            answer = (
                f"### Analysis for Invoice `{inv_no}`\n\n"
                f"**Batch-relative anomaly index:** `{risk_score:.1f}/100` ({risk_level}); not a fraud probability.\n"
                f"**Classification:** `{anomaly_type}`\n"
                f"**Flagged by Models:** {m_text} ({exp['consensus_count']}/3 consensus)\n\n"
                f"**Key Evidence Traced:**\n{evidence_bullets}\n\n"
                f"**Summary:** {exp['summary']}"
            )
            return {
                "answer": answer,
                "confidence": "High (Data-grounded)",
                "category": "explanation"
            }

        # 3. Tax discrepancy / Tax calculation
        if any(w in q for w in ["tax", "gst", "calculation", "cgst", "sgst", "igst", "diff"]):
            cgst = float(invoice_data.get("cgst", 0.0))
            sgst = float(invoice_data.get("sgst", 0.0))
            igst = float(invoice_data.get("igst", 0.0))
            cess = float(invoice_data.get("cess_amount", 0.0))
            expected_tax = round(taxable_val * (gst_rate / 100.0), 2)
            actual_gst = round(cgst + sgst + igst, 2)
            diff = abs(actual_gst - expected_tax)

            answer = (
                f"### Tax Computation Audit for `{inv_no}`\n\n"
                f"- **Taxable Value:** ₹{taxable_val:,.2f}\n"
                f"- **Statutory GST Slab:** {gst_rate:.0f}%\n"
                f"- **Expected Tax Amount:** ₹{expected_tax:,.2f}\n"
                f"- **GST tax heads:** ₹{actual_gst:,.2f}\n"
                f"- **Cess:** ₹{cess:,.2f}\n"
                f"- **Billed Total Tax (including cess):** ₹{total_tax:,.2f}\n"
                f"- **Variance:** ₹{diff:,.2f}\n\n"
                f"**Tax Components:**\n"
                f"- CGST: ₹{cgst:,.2f}\n"
                f"- SGST: ₹{sgst:,.2f}\n"
                f"- IGST: ₹{igst:,.2f}\n\n"
                f"**Status:** " + (
                    f"⚠️ GST arithmetic variance of ₹{diff:,.2f} detected. Verify the source invoice and applicable treatment."
                    if diff > 5.0 else
                    "GST arithmetic matches the recorded rate; this is not a determination of legal compliance."
                )
            )
            return {
                "answer": answer,
                "confidence": "High (Data-grounded)",
                "category": "tax"
            }

        # 4. Actions to take / What should I do? / Next steps
        if any(w in q for w in ["action", "recommend", "step", "fix", "resolve", "checklist", "what should i do", "what do i do"]):
            rec = get_recommendations_for_invoice(invoice_data)
            action_items = "\n".join([
                f"{i+1}. **[{a['priority']}] {a['action']}**\n   - *Details:* {a['detail']}\n   - *Required Doc:* `{a['doc_required']}`"
                for i, a in enumerate(rec["actions"])
            ])
            answer = (
                f"### Recommended Compliance Actions for `{inv_no}`\n\n"
                f"**Recommended Workflow Action:** `{rec['workflow_label']}`\n\n"
                f"**Action Checklist:**\n{action_items}\n\n"
                f"**Risk Context:** {rec['risk_impact']}"
            )
            return {
                "answer": answer,
                "confidence": "High (Rule-based)",
                "category": "recommendations"
            }

        # 5. Can I claim ITC? / Is it safe?
        if any(w in q for w in ["itc", "claim", "input tax", "safe", "credit"]):
            answer = (
                f"### Input Tax Credit (ITC) Guidance for `{inv_no}`\n\n"
                f"The batch-relative anomaly index is **{risk_score:.1f}/100 ({risk_level})** and the recorded pattern is "
                f"**{anomaly_type}**. This statistical screen neither establishes nor determines ITC eligibility, "
                "and it cannot by itself support a decision to claim, defer, or reverse credit.\n\n"
                "Verify the source tax invoice, receipt of the supply, supplier-reported details and applicable "
                "return-period records, payment evidence, and the current statutory requirements against the facts. "
                "Escalate unresolved discrepancies to an appropriately qualified GST professional."
            )
            return {
                "answer": answer,
                "confidence": "General verification guidance",
                "category": "itc"
            }

        # 6. Supplier details
        if any(w in q for w in ["supplier", "vendor", "gstin"]):
            if supplier_data:
                return _supplier_summary(supplier_data, inv_no)
            answer = (
                f"### Supplier Context for `{inv_no}`\n\n"
                f"- **Supplier ID:** `{supplier_id}`\n"
                f"- **Supplier GSTIN:** `{supplier_gstin}`\n"
                f"- **State Code:** `{supplier_gstin[:2] if len(supplier_gstin) >= 2 else 'N/A'}`\n"
                f"- View full profile and compliance track record under the **Suppliers** tab."
            )
            return {
                "answer": answer,
                "confidence": "High (Data-grounded)",
                "category": "supplier"
            }

        if any(w in q for w in ["risk level", "risk category", "risk categories", "risk threshold", "risk thresholds"]):
            answer = (
                f"### Risk Classification for Invoice `{inv_no}`\n\n"
                f"This invoice has a batch-relative anomaly index of **{risk_score:.1f}/100**, classified as **{risk_level}**. "
                "The index is not a fraud probability or legal conclusion.\n\n"
                "- **Low screening priority:** <40\n"
                "- **Medium screening priority:** 40–<70\n"
                "- **High screening priority:** ≥70"
            )
            return {"answer": answer, "confidence": "High (Data-grounded)", "category": "risk_thresholds"}

    # ── CASE 2: General Knowledge & FAQ Queries ──────────────────────────────
    if any(w in q for w in ["gstr-2b", "gstr 2b", "2b"]):
        answer = (
            "### GSTR-2B Reconciliation\n\n"
            "**GSTR-2B** is an auto-drafted statement used to help reconcile supplier-reported details with purchase "
            "records. A statistical anomaly in this application, or a single portal comparison, does not by itself "
            "determine Input Tax Credit eligibility. Review the current statutory conditions, source invoice, supply "
            "receipt, supplier filing information, and applicable return-period records."
        )
        return {"answer": answer, "confidence": "General verification guidance", "category": "general_gst"}

    if any(w in q for w in ["180 day", "180-day", "section 16"]):
        answer = (
            "### Section 16(2) Payment Verification\n\n"
            "Section 16(2) includes a payment-related condition and a 180-day period for the value of supply plus tax, "
            "subject to the statutory text and applicable exceptions. If payment is delayed, confirm the facts and "
            "current treatment, including any applicable interest or re-availment rules, with a qualified GST adviser. "
            "This general note is not a determination for a particular invoice."
        )
        return {"answer": answer, "confidence": "General verification guidance", "category": "general_gst"}

    if any(w in q for w in ["weights", "formula", "ensemble", "scoring"]):
        answer = (
            "### GSTGuard AI Ensemble Scoring Model\n\n"
            "The platform synthesizes 3 complementary anomaly detection models:\n\n"
            "1. **Isolation Forest (40% Weight):** Detects global multidimensional outliers via random decision trees.\n"
            "2. **Local Outlier Factor (40% Weight):** Evaluates local neighborhood density around each invoice.\n"
            "3. **Statistical Z-Score (20% Weight):** Scales the maximum absolute standardized feature deviation.\n\n"
            "Isolation Forest and LOF outputs are min-max scaled within the analyzed batch; the Z-score term is "
            "`min(max_abs_zscore / 8 × 100, 100)`. The terms are weighted 40% / 40% / 20% and added. "
            "The result is a batch-relative 0–100 screening index, not a fraud probability. Detector agreement "
            "does not alter the index or prove fraud, and scores may change when the batch changes."
        )
        return {"answer": answer, "confidence": "Data-grounded explanation", "category": "general_ml"}

    if any(w in q for w in ["risk level", "risk category", "risk categories", "risk threshold", "risk thresholds"]):
        answer = (
            "### GSTGuard AI Risk Categories\n\n"
            "Invoice risk scores are classified using these thresholds:\n\n"
            "- **Low Risk:** 0–39\n"
            "- **Medium Risk:** 40–69\n"
            "- **High Risk:** 70–100\n\n"
            "Higher scores indicate a greater need for supporting-document and compliance review."
        )
        return {"answer": answer, "confidence": "High", "category": "risk_thresholds"}

    # Default fallback
    if invoice_data:
        exp = get_detailed_explanation(invoice_data)
        answer = (
            f"### Invoice `{invoice_data.get('invoice_no')}` Overview\n\n"
            f"- **Screening priority:** {invoice_data.get('risk_level')} ({invoice_data.get('risk_score', 0):.1f}/100, batch-relative)\n"
            f"- **Anomaly Type:** {invoice_data.get('anomaly_type')}\n"
            f"- **Summary:** {exp['summary']}\n\n"
            f"You can ask me:\n"
            f"- *'Why was this invoice flagged?'*\n"
            f"- *'What models flagged this transaction?'*\n"
            f"- *'Is it safe to claim ITC?'*\n"
            f"- *'What are the recommended actions?'*\n"
            f"- *'What is the tax discrepancy?'*"
        )
    else:
        answer = (
            "### GSTGuard AI Compliance Copilot\n\n"
            "I can assist you with:\n"
            "- Explaining why an invoice was flagged as anomalous\n"
            "- Model consensus and evidence verification\n"
            "- Input Tax Credit (ITC) eligibility and risk checks\n"
            "- Supplier profile intelligence\n"
            "- GST statutory rules (GSTR-2B, Section 16(2), e-Way Bill guidelines)\n\n"
            "Please ask a specific compliance question or select an invoice to begin."
        )

    return {
        "answer": answer,
        "confidence": "Medium (General Guidance)",
        "category": "general"
    }
