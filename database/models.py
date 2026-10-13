"""
database/models.py
==================
SQLAlchemy ORM models for SQLite database integration:
  1. Invoice   - Stores raw GST invoice transaction attributes
  2. Analysis  - Stores ML anomaly detection outputs, ensemble risk score, and explanation

The two tables are related via the unique invoice_no primary/foreign key.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Float, Integer, Text, DateTime, ForeignKey, create_engine,
    Boolean
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class UploadBatch(Base):
    """Tracks the CSV dataset currently used by application-facing analytics."""

    __tablename__ = "upload_batches"

    id = Column(String(36), primary_key=True)
    filename = Column(String(255), nullable=False)
    uploaded_at = Column(String(30), nullable=False)
    invoice_count = Column(Integer, nullable=False)
    is_active = Column(Boolean, nullable=False, default=False, index=True)


class Invoice(Base):
    """
    Represents raw GST invoice transactions.
    Matches the schema of data/gst_invoices.csv.
    """
    __tablename__ = "invoices"

    invoice_no          = Column(String(50), primary_key=True, index=True)
    upload_batch_id     = Column(String(36), nullable=True, index=True)
    invoice_date        = Column(String(20), index=True)
    supplier_id         = Column(String(50), index=True)
    supplier_gstin      = Column(String(20), index=True)
    buyer_id            = Column(String(50), index=True)
    buyer_gstin         = Column(String(20), index=True)
    product_category    = Column(String(100), index=True)
    hsn_code            = Column(String(20))
    quantity            = Column(Float, nullable=False)
    unit_price          = Column(Float, nullable=False)
    discount_percentage = Column(Float, default=0.0)
    discount_amount     = Column(Float, default=0.0)
    taxable_value       = Column(Float, nullable=False)
    gst_rate            = Column(Float, nullable=False)
    cgst                = Column(Float, default=0.0)
    sgst                = Column(Float, default=0.0)
    igst                = Column(Float, default=0.0)
    cess_rate           = Column(Float, default=0.0)
    cess_amount         = Column(Float, default=0.0)
    total_tax           = Column(Float, nullable=False)
    total_amount        = Column(Float, nullable=False)
    payment_method      = Column(String(50))

    # Relationship to Analysis result
    analysis = relationship("Analysis", back_populates="invoice", uselist=False, cascade="all, delete-orphan")
    # Relationship to Investigation result
    investigation = relationship("Investigation", back_populates="invoice", uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        """Convert invoice record to serializable dictionary."""
        return {
            "invoice_no": self.invoice_no,
            "invoice_date": self.invoice_date,
            "supplier_id": self.supplier_id,
            "supplier_gstin": self.supplier_gstin,
            "buyer_id": self.buyer_id,
            "buyer_gstin": self.buyer_gstin,
            "product_category": self.product_category,
            "hsn_code": self.hsn_code,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "discount_percentage": self.discount_percentage,
            "discount_amount": self.discount_amount,
            "taxable_value": self.taxable_value,
            "gst_rate": self.gst_rate,
            "cgst": self.cgst,
            "sgst": self.sgst,
            "igst": self.igst,
            "cess_rate": self.cess_rate,
            "cess_amount": self.cess_amount,
            "total_tax": self.total_tax,
            "total_amount": self.total_amount,
            "payment_method": self.payment_method
        }


class Analysis(Base):
    """
    Represents anomaly detection results, ML model scores, ensemble risk level,
    and audit explanations for an invoice.
    Matches the schema of results/ensemble_results.csv.
    """
    __tablename__ = "analysis"

    invoice_no        = Column(String(50), ForeignKey("invoices.invoice_no"), primary_key=True, index=True)
    isolation_score   = Column(Float, nullable=False)
    isolation_anomaly = Column(Integer, nullable=False)   # 1 if anomalous, 0 if normal
    lof_score         = Column(Float, nullable=False)
    lof_anomaly       = Column(Integer, nullable=False)   # 1 if anomalous, 0 if normal
    max_abs_zscore    = Column(Float, nullable=False)
    zscore_anomaly    = Column(Integer, nullable=False)   # 1 if anomalous, 0 if normal
    risk_score        = Column(Float, nullable=False, index=True) # 0.0 to 100.0
    risk_level        = Column(String(20), nullable=False, index=True)  # Normal or Low/Medium/High Risk
    anomaly_type      = Column(String(100), default="Normal", index=True)
    explanation       = Column(Text)
    isolation_contribution = Column(Float, nullable=True)
    lof_contribution = Column(Float, nullable=True)
    zscore_contribution = Column(Float, nullable=True)

    # Reverse relationship to Invoice
    invoice = relationship("Invoice", back_populates="analysis")

    def to_dict(self):
        """Convert analysis record to serializable dictionary."""
        return {
            "invoice_no": self.invoice_no,
            "isolation_score": self.isolation_score,
            "isolation_anomaly": self.isolation_anomaly,
            "lof_score": self.lof_score,
            "lof_anomaly": self.lof_anomaly,
            "max_abs_zscore": self.max_abs_zscore,
            "zscore_anomaly": self.zscore_anomaly,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "anomaly_type": self.anomaly_type,
            "explanation": self.explanation,
            "isolation_contribution": self.isolation_contribution,
            "lof_contribution": self.lof_contribution,
            "zscore_contribution": self.zscore_contribution
        }


class Investigation(Base):
    """
    Represents compliance audit workflow status and investigator notes.
    Statuses: NEW, UNDER REVIEW, VERIFIED, CLEARED, ESCALATED
    """
    __tablename__ = "investigations"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    invoice_no   = Column(String(50), ForeignKey("invoices.invoice_no"), unique=True, index=True, nullable=False)
    status       = Column(String(30), default="NEW", nullable=False, index=True)
    assigned_to  = Column(String(100), default="Unassigned")
    notes        = Column(Text, default="")
    updated_at   = Column(String(30), default=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))

    # Reverse relationship to Invoice
    invoice = relationship("Invoice", back_populates="investigation")

    def to_dict(self):
        """Convert investigation record to serializable dictionary."""
        return {
            "id": self.id,
            "invoice_no": self.invoice_no,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "notes": self.notes,
            "updated_at": self.updated_at
        }


class InvestigationHistory(Base):
    """Append-only application history for investigation tracker updates."""

    __tablename__ = "investigation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_no = Column(String(50), ForeignKey("invoices.invoice_no"), nullable=False, index=True)
    previous_status = Column(String(30), nullable=False)
    new_status = Column(String(30), nullable=False)
    reviewer = Column(String(100), nullable=False)
    updated_at = Column(String(30), nullable=False)
    decision_notes = Column(Text, nullable=False, default="")
