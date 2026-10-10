"""
database/database.py
====================
SQLite Database connection, session management, table initialization,
and database query helper functions.
"""

import os
import sys
import sqlite3
from datetime import timezone
import pandas as pd
from sqlalchemy import create_engine, func, case, and_, or_, inspect
from sqlalchemy.orm import sessionmaker, scoped_session

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import ORM models
from database.models import Base, Invoice, Analysis, Investigation, InvestigationHistory

# ── Database Path & Engine Configuration ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = os.path.abspath(os.environ.get(
    "GSTGUARD_DATABASE_PATH",
    os.path.join(BASE_DIR, "gst_fraud.db")
))
DATABASE_URI = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URI, echo=False, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


def get_db():
    """Contextual database session generator for web routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all SQLite database tables according to SQLAlchemy models."""
    print(f"[DB] Initializing database schema at: {DB_PATH}")
    Base.metadata.create_all(bind=engine)
    analysis_columns = {column["name"] for column in inspect(engine).get_columns("analysis")}
    for column_name in (
        "isolation_contribution", "lof_contribution", "zscore_contribution"
    ):
        if column_name not in analysis_columns:
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    f"ALTER TABLE analysis ADD COLUMN {column_name} FLOAT"
                )
    print("     Tables 'invoices' and 'analysis' created successfully.")


def populate_database(invoices_path: str = None, ensemble_path: str = None):
    """
    Populate SQLite database from raw CSVs (gst_invoices.csv and ensemble_results.csv).
    Uses high-speed pandas to_sql with index handling.
    """
    if invoices_path is None:
        invoices_path = os.path.join(ROOT_DIR, "data", "gst_invoices.csv")
    if ensemble_path is None:
        ensemble_path = os.path.join(ROOT_DIR, "results", "ensemble_results.csv")

    print("[DB] Populating database from CSV data ...")
    if not os.path.exists(invoices_path):
        raise FileNotFoundError(f"Invoices CSV not found: {invoices_path}")
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Ensemble results CSV not found: {ensemble_path}")

    # Ensure tables exist
    init_db()

    # Load data
    print("     Reading input CSV files ...")
    df_inv = pd.read_csv(invoices_path)
    df_ens = pd.read_csv(ensemble_path)

    # Use SQLite native connection for maximum bulk insert performance
    conn = sqlite3.connect(DB_PATH)
    try:
        # Clear existing data to avoid primary key collisions if re-running
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis;")
        cursor.execute("DELETE FROM invoices;")
        conn.commit()

        print(f"     Inserting {len(df_inv):,} raw invoices into 'invoices' table ...")
        df_inv.to_sql("invoices", conn, if_exists="append", index=False)

        print(f"     Inserting {len(df_ens):,} ML analysis records into 'analysis' table ...")
        df_ens.to_sql("analysis", conn, if_exists="append", index=False)

        conn.commit()
        print("     Bulk insertion completed successfully.")
    finally:
        conn.close()

    # Verify counts
    session = SessionLocal()
    try:
        inv_count = session.query(Invoice).count()
        ans_count = session.query(Analysis).count()
        print(f"[DB] Verification: Invoices table = {inv_count:,} records | Analysis table = {ans_count:,} records.")
    finally:
        session.close()


# =============================================================================
#  Query Helpers for Dashboard, API & Reports
# =============================================================================

def get_dashboard_summary():
    """
    Compute live dynamic metrics for dashboard cards and summary widgets.
    Retrieves real information directly from SQLite without hardcoded numbers.
    """
    session = SessionLocal()
    try:
        total_invoices = session.query(Invoice).count()

        # Risk level counts
        risk_counts = dict(
            session.query(Analysis.risk_level, func.count(Analysis.invoice_no))
            .group_by(Analysis.risk_level)
            .all()
        )

        low_risk    = risk_counts.get("Low Risk", 0)
        medium_risk = risk_counts.get("Medium Risk", 0)
        high_risk   = risk_counts.get("High Risk", 0)
        suspicious  = medium_risk + high_risk
        normal      = low_risk

        # Average risk score
        avg_score = session.query(func.avg(Analysis.risk_score)).scalar() or 0.0

        # Anomaly type breakdown
        anomaly_pattern = case(
            (
                and_(
                    Analysis.anomaly_type == "Normal",
                    Analysis.risk_level.in_(["Medium Risk", "High Risk"]),
                ),
                "Statistical Model Outlier",
            ),
            else_=Analysis.anomaly_type,
        )
        type_counts = dict(
            session.query(anomaly_pattern, func.count(Analysis.invoice_no))
            .group_by(anomaly_pattern)
            .order_by(func.count(Analysis.invoice_no).desc())
            .all()
        )

        return {
            "total_invoices": total_invoices,
            "normal_invoices": normal,
            "suspicious_invoices": suspicious,
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "low_risk": low_risk,
            "avg_risk_score": round(float(avg_score), 2),
            "risk_distribution": {
                "Low Risk": low_risk,
                "Medium Risk": medium_risk,
                "High Risk": high_risk
            },
            "anomaly_types": type_counts
        }
    finally:
        session.close()


def get_invoices(page: int = 1, per_page: int = 25, search: str = "", risk_level: str = "", anomaly_type: str = ""):
    """Paginated list of invoices joined with analysis risk scores."""
    session = SessionLocal()
    try:
        query = session.query(Invoice, Analysis).join(Analysis, Invoice.invoice_no == Analysis.invoice_no)

        if search:
            search_term = f"%{search.strip()}%"
            query = query.filter(
                (Invoice.invoice_no.like(search_term)) |
                (Invoice.supplier_gstin.like(search_term)) |
                (Invoice.buyer_gstin.like(search_term)) |
                (Invoice.product_category.like(search_term))
            )

        if risk_level:
            query = query.filter(Analysis.risk_level == risk_level)

        if anomaly_type == "Statistical Model Outlier":
            query = query.filter(or_(
                Analysis.anomaly_type == anomaly_type,
                and_(
                    Analysis.anomaly_type == "Normal",
                    Analysis.risk_level.in_(["Medium Risk", "High Risk"]),
                ),
            ))
        elif anomaly_type == "Normal":
            query = query.filter(
                Analysis.anomaly_type == "Normal",
                Analysis.risk_level == "Low Risk",
            )
        elif anomaly_type:
            query = query.filter(Analysis.anomaly_type == anomaly_type)

        total_count = query.count()
        offset = (page - 1) * per_page
        items = query.order_by(Analysis.risk_score.desc()).offset(offset).limit(per_page).all()

        records = []
        for inv, ans in items:
            rec = inv.to_dict()
            rec.update(ans.to_dict())
            _normalize_anomaly_pattern(rec)
            records.append(rec)

        total_pages = (total_count + per_page - 1) // per_page

        return {
            "items": records,
            "total": total_count,
            "page": page,
            "per_page": per_page,
            "pages": total_pages
        }
    finally:
        session.close()


def get_suspicious_invoices(page: int = 1, per_page: int = 25, search: str = "",
                            risk_level: str = "", anomaly_type: str = ""):
    """Retrieve only suspicious (Medium Risk & High Risk) transactions."""
    session = SessionLocal()
    try:
        query = session.query(Invoice, Analysis).join(
            Analysis, Invoice.invoice_no == Analysis.invoice_no
        ).filter(Analysis.risk_level.in_(["Medium Risk", "High Risk"]))

        if search:
            search_term = f"%{search.strip()}%"
            query = query.filter(
                (Invoice.invoice_no.like(search_term)) |
                (Invoice.supplier_gstin.like(search_term)) |
                (Invoice.buyer_gstin.like(search_term)) |
                (Invoice.product_category.like(search_term))
            )

        if risk_level in ["Medium Risk", "High Risk"]:
            query = query.filter(Analysis.risk_level == risk_level)

        if anomaly_type == "Statistical Model Outlier":
            query = query.filter(or_(
                Analysis.anomaly_type == anomaly_type,
                and_(
                    Analysis.anomaly_type == "Normal",
                    Analysis.risk_level.in_(["Medium Risk", "High Risk"]),
                ),
            ))
        elif anomaly_type == "Normal":
            query = query.filter(
                Analysis.anomaly_type == "Normal",
                Analysis.risk_level == "Low Risk",
            )
        elif anomaly_type:
            query = query.filter(Analysis.anomaly_type == anomaly_type)

        total_count = query.count()
        offset = (page - 1) * per_page
        items = query.order_by(Analysis.risk_score.desc()).offset(offset).limit(per_page).all()

        records = []
        for inv, ans in items:
            rec = inv.to_dict()
            rec.update(ans.to_dict())
            _normalize_anomaly_pattern(rec)
            records.append(rec)

        total_pages = (total_count + per_page - 1) // per_page

        return {
            "items": records,
            "total": total_count,
            "page": page,
            "per_page": per_page,
            "pages": total_pages
        }
    finally:
        session.close()


def get_invoice_by_no(invoice_no: str):
    """Retrieve complete transaction details, model scores, explanation, and investigation status for an invoice."""
    session = SessionLocal()
    try:
        item = session.query(Invoice, Analysis).join(
            Analysis, Invoice.invoice_no == Analysis.invoice_no
        ).filter(Invoice.invoice_no == invoice_no).first()

        if not item:
            return None

        inv, ans = item
        data = inv.to_dict()
        data.update(ans.to_dict())
        _normalize_anomaly_pattern(data)

        # Fetch investigation record if exists
        inv_record = session.query(Investigation).filter(Investigation.invoice_no == invoice_no).first()
        if inv_record:
            data["investigation"] = inv_record.to_dict()
        else:
            data["investigation"] = {
                "id": None,
                "invoice_no": invoice_no,
                "status": "NEW",
                "assigned_to": "Unassigned",
                "notes": "",
                "updated_at": None
            }
        data["investigation_history"] = get_investigation_history(invoice_no)
        return data
    finally:
        session.close()


def _normalize_anomaly_pattern(invoice_data):
    """Label model-elevated invoices when no specific rule-based pattern matched."""
    if (
        invoice_data.get("anomaly_type") == "Normal"
        and invoice_data.get("risk_level") in ("Medium Risk", "High Risk", "Medium", "High")
    ):
        invoice_data["anomaly_type"] = "Statistical Model Outlier"
    return invoice_data


INVESTIGATION_STATUSES = {
    "NEW", "UNDER REVIEW", "CONFIRMED IRREGULARITY", "FALSE POSITIVE", "RESOLVED",
    "VERIFIED", "CLEARED", "ESCALATED",
}
INVESTIGATION_TRANSITIONS = {
    "NEW": {"UNDER REVIEW", "FALSE POSITIVE", "RESOLVED", "VERIFIED", "CLEARED", "ESCALATED"},
    "UNDER REVIEW": {"NEW", "CONFIRMED IRREGULARITY", "FALSE POSITIVE", "RESOLVED",
                     "VERIFIED", "CLEARED", "ESCALATED"},
    "CONFIRMED IRREGULARITY": {"UNDER REVIEW", "RESOLVED", "ESCALATED"},
    "FALSE POSITIVE": {"NEW", "UNDER REVIEW", "RESOLVED"},
    "RESOLVED": {"UNDER REVIEW"},
    "VERIFIED": {"UNDER REVIEW", "RESOLVED", "ESCALATED", "CONFIRMED IRREGULARITY"},
    "CLEARED": {"UNDER REVIEW", "RESOLVED", "FALSE POSITIVE"},
    "ESCALATED": {"UNDER REVIEW", "CONFIRMED IRREGULARITY", "RESOLVED"},
}


def get_or_create_investigation(invoice_no: str):
    """Fetch existing investigation or create a new one with default 'NEW' status."""
    session = SessionLocal()
    try:
        rec = session.query(Investigation).filter(Investigation.invoice_no == invoice_no).first()
        if not rec:
            rec = Investigation(invoice_no=invoice_no, status="NEW", assigned_to="Unassigned", notes="")
            session.add(rec)
            session.commit()
            session.refresh(rec)
        return rec.to_dict()
    finally:
        session.close()


def update_investigation(
    invoice_no: str,
    status: str = None,
    notes: str = None,
    assigned_to: str = None,
    reviewer: str = "System",
):
    """Update an existing investigation and append its decision to history."""
    from datetime import datetime
    session = SessionLocal()
    try:
        invoice = session.query(Invoice.invoice_no).filter(
            Invoice.invoice_no == invoice_no
        ).first()
        if not invoice:
            raise LookupError(f"Invoice '{invoice_no}' does not exist.")

        rec = session.query(Investigation).filter(Investigation.invoice_no == invoice_no).first()
        if not rec:
            rec = Investigation(
                invoice_no=invoice_no,
                status="NEW",
                assigned_to="Unassigned",
                notes="",
            )
            session.add(rec)

        previous_status = rec.status or "NEW"
        new_status = status.strip().upper() if status is not None else previous_status
        if new_status not in INVESTIGATION_STATUSES:
            raise ValueError("Invalid investigation status.")
        if new_status != previous_status and new_status not in INVESTIGATION_TRANSITIONS.get(
            previous_status, set()
        ):
            raise ValueError(
                f"Invalid investigation status transition: {previous_status} to {new_status}."
            )
        if notes is not None and len(notes) > 10000:
            raise ValueError("Finding notes must be 10,000 characters or fewer.")
        if assigned_to is not None and len(assigned_to.strip()) > 100:
            raise ValueError("Assigned reviewer must be 100 characters or fewer.")
        if not reviewer.strip() or len(reviewer.strip()) > 100:
            raise ValueError("Reviewer must be between 1 and 100 characters.")

        rec.status = new_status
        if notes is not None:
            rec.notes = notes
        if assigned_to is not None:
            rec.assigned_to = assigned_to.strip() or "Unassigned"

        updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        rec.updated_at = updated_at
        history = InvestigationHistory(
            invoice_no=invoice_no,
            previous_status=previous_status,
            new_status=new_status,
            reviewer=reviewer.strip(),
            updated_at=updated_at,
            decision_notes=notes if notes is not None else (rec.notes or ""),
        )
        session.add(history)
        session.commit()
        session.refresh(rec)
        return rec.to_dict()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_investigation_history(invoice_no: str, limit: int = 50):
    """Return recent append-only investigation events in chronological order."""
    session = SessionLocal()
    try:
        rows = (
            session.query(InvestigationHistory)
            .filter(InvestigationHistory.invoice_no == invoice_no)
            .order_by(InvestigationHistory.id.desc())
            .limit(min(max(limit, 1), 200))
            .all()
        )
        return [
            {
                "previous_status": row.previous_status,
                "new_status": row.new_status,
                "reviewer": row.reviewer,
                "updated_at": row.updated_at,
                "decision_notes": row.decision_notes,
            }
            for row in reversed(rows)
        ]
    finally:
        session.close()


def get_supplier_metrics(page: int = 1, per_page: int = 20, search: str = "", sort_by: str = "risk_score", order: str = "desc"):
    """
    Aggregate supplier-level risk intelligence directly from invoices and analysis tables.
    Returns:
      - supplier_id, supplier_gstin
      - total_invoices, total_taxable_value, total_tax
      - suspicious_count, high_risk_count, medium_risk_count, low_risk_count
      - avg_risk_score, max_risk_score, supplier_risk_score (composite 0-100)
    """
    session = SessionLocal()
    try:
        # Build base aggregation query
        q = session.query(
            Invoice.supplier_id,
            func.min(Invoice.supplier_gstin).label("supplier_gstin"),
            func.count(Invoice.invoice_no).label("total_invoices"),
            func.sum(Invoice.taxable_value).label("total_taxable_value"),
            func.sum(Invoice.total_tax).label("total_tax"),
            func.avg(Analysis.risk_score).label("avg_risk_score"),
            func.max(Analysis.risk_score).label("max_risk_score"),
            func.sum(case((Analysis.risk_level == "High Risk", 1), else_=0)).label("high_risk_count"),
            func.sum(case((Analysis.risk_level == "Medium Risk", 1), else_=0)).label("medium_risk_count"),
            func.sum(case((Analysis.risk_level == "Low Risk", 1), else_=0)).label("low_risk_count")
        ).join(
            Analysis, Invoice.invoice_no == Analysis.invoice_no
        ).group_by(
            Invoice.supplier_id
        )

        if search:
            s = f"%{search.strip()}%"
            q = q.filter((Invoice.supplier_id.like(s)) | (Invoice.supplier_gstin.like(s)))

        all_rows = q.all()

        suppliers = []
        for r in all_rows:
            tot = r.total_invoices or 1
            high = r.high_risk_count or 0
            med = r.medium_risk_count or 0
            low = r.low_risk_count or 0
            suspicious = high + med
            avg_risk = round(float(r.avg_risk_score or 0.0), 2)
            max_risk = round(float(r.max_risk_score or 0.0), 2)

            # Supplier Composite Risk Index (0-100):
            # 50% average risk + 30% max risk + 20% proportion of high risk invoices
            suspicious_ratio = (high * 2.0 + med * 1.0) / (tot * 2.0)
            supplier_risk_score = round(min(100.0, (avg_risk * 0.4) + (max_risk * 0.3) + (suspicious_ratio * 100.0 * 0.3)), 2)

            if supplier_risk_score >= 70.0:
                supplier_risk_level = "High Risk"
            elif supplier_risk_score >= 40.0:
                supplier_risk_level = "Medium Risk"
            else:
                supplier_risk_level = "Low Risk"

            suppliers.append({
                "supplier_id": r.supplier_id,
                "supplier_gstin": r.supplier_gstin,
                "total_invoices": tot,
                "total_taxable_value": round(float(r.total_taxable_value or 0.0), 2),
                "total_tax": round(float(r.total_tax or 0.0), 2),
                "suspicious_count": suspicious,
                "high_risk_count": high,
                "medium_risk_count": med,
                "low_risk_count": low,
                "avg_risk_score": avg_risk,
                "max_risk_score": max_risk,
                "supplier_risk_score": supplier_risk_score,
                "supplier_risk_level": supplier_risk_level
            })

        # Sorting
        reverse = (order.lower() == "desc")
        if sort_by == "risk_score" or sort_by == "supplier_risk_score":
            suppliers.sort(key=lambda x: x["supplier_risk_score"], reverse=reverse)
        elif sort_by == "high_risk_count":
            suppliers.sort(key=lambda x: x["high_risk_count"], reverse=reverse)
        elif sort_by == "total_invoices":
            suppliers.sort(key=lambda x: x["total_invoices"], reverse=reverse)
        elif sort_by == "total_taxable_value":
            suppliers.sort(key=lambda x: x["total_taxable_value"], reverse=reverse)
        else:
            suppliers.sort(key=lambda x: x["supplier_risk_score"], reverse=reverse)

        total_count = len(suppliers)
        total_pages = (total_count + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        paginated_items = suppliers[start_idx:start_idx + per_page]

        return {
            "items": paginated_items,
            "total": total_count,
            "page": page,
            "per_page": per_page,
            "pages": total_pages
        }
    finally:
        session.close()


def get_supplier_detail(supplier_id: str):
    """Retrieve full details, metrics, and invoices for a single supplier."""
    session = SessionLocal()
    try:
        invoices_q = session.query(Invoice, Analysis, Investigation).join(
            Analysis, Invoice.invoice_no == Analysis.invoice_no
        ).outerjoin(
            Investigation, Invoice.invoice_no == Investigation.invoice_no
        ).filter(Invoice.supplier_id == supplier_id).order_by(Analysis.risk_score.desc()).all()

        if not invoices_q:
            return None

        supplier_gstin = min(inv.supplier_gstin for inv, _, _ in invoices_q)

        total_invoices = len(invoices_q)
        total_taxable = sum(inv.taxable_value for inv, _, _ in invoices_q)
        total_tax = sum(inv.total_tax for inv, _, _ in invoices_q)
        risk_scores = [ans.risk_score for _, ans, _ in invoices_q]
        avg_risk = round(float(sum(risk_scores) / total_invoices), 2)
        max_risk = round(float(max(risk_scores)), 2)

        high_cnt = sum(1 for _, ans, _ in invoices_q if ans.risk_level == "High Risk")
        med_cnt = sum(1 for _, ans, _ in invoices_q if ans.risk_level == "Medium Risk")
        low_cnt = sum(1 for _, ans, _ in invoices_q if ans.risk_level == "Low Risk")

        suspicious_ratio = (high_cnt * 2.0 + med_cnt * 1.0) / (total_invoices * 2.0)
        supplier_risk_score = round(min(100.0, (avg_risk * 0.4) + (max_risk * 0.3) + (suspicious_ratio * 100.0 * 0.3)), 2)

        if supplier_risk_score >= 70.0:
            supplier_risk_level = "High Risk"
        elif supplier_risk_score >= 40.0:
            supplier_risk_level = "Medium Risk"
        else:
            supplier_risk_level = "Low Risk"

        invoice_list = []
        for inv, ans, invg in invoices_q:
            rec = inv.to_dict()
            rec.update(ans.to_dict())
            rec["investigation_status"] = invg.status if invg else "NEW"
            invoice_list.append(rec)

        return {
            "supplier_id": supplier_id,
            "supplier_gstin": supplier_gstin,
            "total_invoices": total_invoices,
            "total_taxable_value": round(float(total_taxable), 2),
            "total_tax": round(float(total_tax), 2),
            "avg_risk_score": avg_risk,
            "max_risk_score": max_risk,
            "high_risk_count": high_cnt,
            "medium_risk_count": med_cnt,
            "low_risk_count": low_cnt,
            "suspicious_count": high_cnt + med_cnt,
            "supplier_risk_score": supplier_risk_score,
            "supplier_risk_level": supplier_risk_level,
            "invoices": invoice_list
        }
    finally:
        session.close()


def get_review_queue(limit: int = 50):
    """
    Retrieve ranked priority review queue for compliance officers:
    Ranked by the screening index, detector-flag count, and taxable value.
    Excludes cleared, verified, false-positive, and resolved investigations.
    """
    session = SessionLocal()
    try:
        items = session.query(Invoice, Analysis, Investigation).join(
            Analysis, Invoice.invoice_no == Analysis.invoice_no
        ).outerjoin(
            Investigation, Invoice.invoice_no == Investigation.invoice_no
        ).filter(
            Analysis.risk_level.in_(["High Risk", "Medium Risk"])
        ).all()

        queue = []
        for inv, ans, invg in items:
            status = invg.status if invg else "NEW"
            if status in ["CLEARED", "VERIFIED", "FALSE POSITIVE", "RESOLVED"]:
                continue

            consensus = ans.isolation_anomaly + ans.lof_anomaly + ans.zscore_anomaly
            # Priority formula:
            priority_score = (ans.risk_score * 0.5) + (consensus * 12.0)
            if inv.taxable_value > 500000:
                priority_score += 15.0
            elif inv.taxable_value > 100000:
                priority_score += 8.0

            priority_score = min(100.0, round(priority_score, 1))

            queue.append({
                "invoice_no": inv.invoice_no,
                "invoice_date": inv.invoice_date,
                "supplier_id": inv.supplier_id,
                "supplier_gstin": inv.supplier_gstin,
                "product_category": inv.product_category,
                "taxable_value": inv.taxable_value,
                "total_amount": inv.total_amount,
                "risk_score": ans.risk_score,
                "risk_level": ans.risk_level,
                "anomaly_type": (
                    "Statistical Model Outlier"
                    if ans.anomaly_type == "Normal" and ans.risk_level in ("Medium Risk", "High Risk")
                    else ans.anomaly_type
                ),
                "explanation": ans.explanation,
                "consensus_count": consensus,
                "priority_score": priority_score,
                "investigation_status": status,
                "assigned_to": invg.assigned_to if invg else "Unassigned",
                "notes": invg.notes if invg else "",
                "updated_at": invg.updated_at if invg else None
            })

        queue.sort(key=lambda x: x["priority_score"], reverse=True)
        return queue[:limit]
    finally:
        session.close()



def main():
    """Standalone runner to initialize and populate the database."""
    print("=" * 70)
    print("  STAGE 8: SQLITE DATABASE INITIALIZATION & POPULATION")
    print("=" * 70)
    populate_database()
    summary = get_dashboard_summary()
    print("\n[DB] Summary verification from live SQLite database:")
    print(f"     Total Invoices      : {summary['total_invoices']:,}")
    print(f"     Normal Invoices     : {summary['normal_invoices']:,}")
    print(f"     Suspicious Invoices : {summary['suspicious_invoices']:,} (High: {summary['high_risk']}, Medium: {summary['medium_risk']})")
    print(f"     Average Risk Score  : {summary['avg_risk_score']}")
    print("\n[DONE] Stage 8 complete.")


if __name__ == "__main__":
    main()
