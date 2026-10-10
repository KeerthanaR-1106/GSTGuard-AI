"""Financial arithmetic checks for invoice investigation evidence."""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional


CENT = Decimal("0.01")
TOLERANCE = CENT


def _decimal(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return result if result.is_finite() else None


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _check(name: str, actual: Optional[Decimal], expected: Optional[Decimal],
           detail: str) -> Dict[str, Any]:
    if actual is None or expected is None:
        status = "UNAVAILABLE"
        variance = None
    else:
        variance = _money(abs(actual - expected))
        status = "MATCH" if variance <= TOLERANCE else "MISMATCH"
    return {
        "check": name,
        "actual": f"₹{_money(actual):,.2f}" if actual is not None else "Unavailable",
        "expected": f"₹{_money(expected):,.2f}" if expected is not None else "Unavailable",
        "variance": f"₹{variance:,.2f}" if variance is not None else "Unavailable",
        "status": status,
        "detail": detail,
    }


def validate_invoice_arithmetic(invoice: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare stored invoice amounts without mutating or normalizing source values."""
    quantity = _decimal(invoice.get("quantity"))
    unit_price = _decimal(invoice.get("unit_price"))
    discount_pct = _decimal(invoice.get("discount_percentage"))
    discount_amount = _decimal(invoice.get("discount_amount"))
    taxable = _decimal(invoice.get("taxable_value"))
    rate = _decimal(invoice.get("gst_rate"))
    cgst = _decimal(invoice.get("cgst"))
    sgst = _decimal(invoice.get("sgst"))
    igst = _decimal(invoice.get("igst"))
    cess_rate = _decimal(invoice.get("cess_rate"))
    cess = _decimal(invoice.get("cess_amount"))
    total_tax = _decimal(invoice.get("total_tax"))
    total_amount = _decimal(invoice.get("total_amount"))

    gross = _money(quantity * unit_price) if quantity is not None and unit_price is not None else None
    after_discount = (
        _money(gross - discount_amount)
        if gross is not None and discount_amount is not None else None
    )
    discount_by_rate = (
        _money(gross * discount_pct / Decimal("100"))
        if gross is not None and discount_pct is not None else None
    )
    expected_gst = (
        _money(taxable * rate / Decimal("100"))
        if taxable is not None and rate is not None else None
    )
    expected_cess = (
        _money(taxable * cess_rate / Decimal("100"))
        if taxable is not None and cess_rate is not None else None
    )

    checks = [
        _check(
            "Quantity × unit price − discount = taxable value",
            taxable, after_discount,
            "Arithmetic reconciliation only; it does not establish that the commercial price or supply is genuine.",
        ),
        _check(
            "Discount amount agrees with discount percentage",
            discount_amount, discount_by_rate,
            "Compare the recorded discount with the gross line amount and stated percentage.",
        ),
        _check(
            "GST tax heads agree with taxable value × GST rate",
            _money(cgst + sgst + igst) if None not in (cgst, sgst, igst) else None,
            expected_gst,
            "Tax amount arithmetic is not a determination of the legally applicable rate.",
        ),
        _check(
            "Cess agrees with taxable value × cess rate",
            cess, expected_cess,
            "Verify cess applicability and rate against the applicable product and supporting documents.",
        ),
        _check(
            "Total tax equals GST tax heads plus cess",
            total_tax,
            _money(cgst + sgst + igst + cess)
            if None not in (cgst, sgst, igst, cess) else None,
            "Cess is included once in total tax.",
        ),
        _check(
            "Grand total equals taxable value plus total tax",
            total_amount,
            _money(taxable + total_tax)
            if taxable is not None and total_tax is not None else None,
            "Discount is already reflected in taxable value; do not subtract it again.",
        ),
    ]

    supplier_gstin = str(invoice.get("supplier_gstin") or "")
    buyer_gstin = str(invoice.get("buyer_gstin") or "")
    supplier_state = supplier_gstin[:2] if len(supplier_gstin) >= 2 else ""
    buyer_state = buyer_gstin[:2] if len(buyer_gstin) >= 2 else ""
    if not supplier_state.isdigit() or not buyer_state.isdigit():
        checks.append({
            "check": "GST tax-head split matches supplier/buyer GSTIN state codes",
            "actual": "Unavailable",
            "expected": "Unavailable",
            "variance": "Unavailable",
            "status": "UNAVAILABLE",
            "detail": "State-code comparison requires usable supplier and buyer GSTIN prefixes.",
        })
    elif None in (cgst, sgst, igst, expected_gst):
        checks.append({
            "check": "GST tax-head split matches supplier/buyer GSTIN state codes",
            "actual": "Unavailable",
            "expected": "Unavailable",
            "variance": "Unavailable",
            "status": "UNAVAILABLE",
            "detail": "One or more tax-head amounts are missing or invalid.",
        })
    else:
        if supplier_state == buyer_state:
            expected_cgst = _money(expected_gst / Decimal("2"))
            expected_sgst = _money(expected_gst / Decimal("2"))
            valid_split = (
                abs(cgst - expected_cgst) <= TOLERANCE
                and abs(sgst - expected_sgst) <= TOLERANCE
                and abs(igst) <= TOLERANCE
            )
            expected_heads = (
                f"CGST ₹{expected_cgst:,.2f}; SGST ₹{expected_sgst:,.2f}; IGST ₹0.00"
            )
        else:
            valid_split = (
                abs(cgst) <= TOLERANCE
                and abs(sgst) <= TOLERANCE
                and abs(igst - expected_gst) <= TOLERANCE
            )
            expected_heads = f"CGST ₹0.00; SGST ₹0.00; IGST ₹{expected_gst:,.2f}"
        checks.append({
            "check": "GST tax-head split matches supplier/buyer GSTIN state codes",
            "actual": f"CGST ₹{cgst:,.2f}; SGST ₹{sgst:,.2f}; IGST ₹{igst:,.2f}",
            "expected": expected_heads,
            "variance": "₹0.00" if valid_split else "See tax-head amounts",
            "status": "MATCH" if valid_split else "MISMATCH",
            "detail": (
                "Intra-state/inter-state treatment is inferred from the first two GSTIN characters; "
                "verify place of supply and registration details independently."
            ),
        })

    for field, value in (
        ("quantity", quantity), ("unit_price", unit_price),
        ("discount_percentage", discount_pct), ("discount_amount", discount_amount),
        ("taxable_value", taxable), ("gst_rate", rate), ("cgst", cgst),
        ("sgst", sgst), ("igst", igst), ("cess_rate", cess_rate),
        ("cess_amount", cess), ("total_tax", total_tax),
        ("total_amount", total_amount),
    ):
        if value is None or value < 0:
            checks.append({
                "check": f"Valid non-negative {field}",
                "actual": "Missing or invalid" if value is None else str(value),
                "expected": "A finite value greater than or equal to zero",
                "variance": "Unavailable",
                "status": "MISMATCH",
                "detail": f"The stored {field} value requires source-document verification.",
            })
    if discount_pct is not None and discount_pct > 100:
        checks.append({
            "check": "Discount percentage is within 0–100%",
            "actual": f"{discount_pct}%",
            "expected": "0–100%",
            "variance": "Unavailable",
            "status": "MISMATCH",
            "detail": "The recorded discount percentage is outside the arithmetic range.",
        })

    return checks
