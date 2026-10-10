"""
app.py
======
Main Flask Application for the Automated GST Invoice Fraud Detection System.
Registers blueprints for Dashboard, Invoices, Upload, and Reports.
Provides both web dashboard views and REST APIs.
"""

import os
import sys

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, jsonify, request
from config import Config
from database.database import init_db

# Import route blueprints
from routes.dashboard_routes import dashboard_bp
from routes.invoice_routes import invoice_bp
from routes.report_routes import report_bp
from routes.upload_routes import upload_bp
from routes.supplier_routes import supplier_bp
from routes.copilot_routes import copilot_bp


def create_app(config_class=Config):
    """Flask application factory."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize database tables if not already present
    with app.app_context():
        init_db()

    # Register blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(invoice_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(supplier_bp)
    app.register_blueprint(copilot_bp)

    # ── Context Processors ───────────────────────────────────────────────────
    @app.context_processor
    def inject_globals():
        """Make common variables available across all templates."""
        return {
            "app_title": "GSTGuard AI — Intelligent GST Invoice Risk & MSME Compliance Platform",
            "active_endpoint": request.endpoint or ""
        }


    # ── Error Handlers ───────────────────────────────────────────────────────
    @app.errorhandler(404)
    def page_not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({
                "status": "error",
                "code": 404,
                "message": getattr(e, "description", "Resource not found.")
            }), 404
        return render_template("404.html", error=e), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        if request.path.startswith("/api/"):
            return jsonify({
                "status": "error",
                "code": 500,
                "message": "An internal server error occurred while processing the compliance query."
            }), 500
        return render_template("500.html", error=e), 500

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return jsonify({
            "status": "error",
            "code": 413,
            "message": "File exceeds the 32MB upload size limit."
        }), 413

    return app


app = create_app()

if __name__ == "__main__":
    print("[APP] Starting GST Fraud Detection System on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True)
