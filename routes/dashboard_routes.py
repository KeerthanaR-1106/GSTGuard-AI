"""
routes/dashboard_routes.py
==========================
Dashboard overview, summary metrics, demonstration authentication,
and informational pages.
"""

from flask import (
    Blueprint, render_template, request, redirect, url_for, session, jsonify, flash
)
from config import Config
from database.database import get_dashboard_summary, get_review_queue
from services.supplier_service import get_top_risky_suppliers

dashboard_bp = Blueprint("dashboard", __name__)

EMPTY_DASHBOARD_SUMMARY = {
    "total_invoices": 0,
    "normal_invoices": 0,
    "suspicious_invoices": 0,
    "high_risk": 0,
    "medium_risk": 0,
    "low_risk": 0,
    "avg_risk_score": 0.0,
    "risk_distribution": {
        "Normal": 0,
        "Low Risk": 0,
        "Medium Risk": 0,
        "High Risk": 0,
    },
    "anomaly_types": {},
}


@dashboard_bp.route("/")
def index():
    """Redirect to dashboard or login page based on session state."""
    destination = (
        "dashboard.dashboard_view"
        if session.get("logged_in") and session.get("user")
        else "dashboard.login_view"
    )
    return redirect(url_for(destination))


@dashboard_bp.route("/login", methods=["GET", "POST"])
def login_view():
    """Simple demonstration login for viva evaluation."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == Config.DEMO_USER and password == Config.DEMO_PASSWORD:
            session["logged_in"] = True
            session["user"] = username
            session["has_uploaded_csv"] = False
            flash("Successfully logged in as Compliance Officer.", "success")
            return redirect(url_for("dashboard.dashboard_view"))
        else:
            flash("Invalid credentials. Use demo account: admin / admin123", "danger")

    return render_template("login.html")


@dashboard_bp.route("/logout")
def logout_view():
    """End demo session."""
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("dashboard.login_view"))


@dashboard_bp.route("/dashboard")
def dashboard_view():
    """Render executive dashboard with real data queried from SQLite."""
    has_uploaded_csv = session.get("has_uploaded_csv", False)
    summary = get_dashboard_summary()
    top_suppliers = get_top_risky_suppliers(limit=5)
    review_queue = get_review_queue(limit=5)
    return render_template(
        "dashboard.html",
        summary=summary,
        top_suppliers=top_suppliers,
        review_queue=review_queue,
        has_uploaded_csv=has_uploaded_csv,
    )


@dashboard_bp.route("/profile")
def profile_view():
    """Render the signed-in user's profile summary."""
    if not session.get("logged_in"):
        return redirect(url_for("dashboard.login_view"))

    username = session.get("user") or "Compliance Officer"
    return render_template(
        "profile.html",
        username=username,
        user_role="Compliance Officer",
        last_login="Today",
        active_page="profile",
    )


# ── REST API ENDPOINTS ────────────────────────────────────────────────────────

@dashboard_bp.route("/api/dashboard", methods=["GET"])
def api_dashboard_summary():
    """
    GET /api/dashboard
    Returns live summary statistics from SQLite:
      - total_invoices
      - normal_invoices
      - suspicious_invoices (medium + high)
      - high_risk, medium_risk, low_risk
      - avg_risk_score
      - risk_distribution
      - anomaly_types
    """
    try:
        data = (
            get_dashboard_summary()
            if session.get("has_uploaded_csv", False)
            else EMPTY_DASHBOARD_SUMMARY
        )
        return jsonify({
            "status": "success",
            "data": data
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to retrieve dashboard metrics: {str(e)}"
        }), 500
