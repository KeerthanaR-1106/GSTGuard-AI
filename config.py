"""
config.py
=========
Global configuration settings for the GST Fraud Detection Flask Application.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # Flask Security
    SECRET_KEY = os.environ.get("SECRET_KEY", "gst-fraud-detection-secret-key-2026")
    
    # SQLite Database Configuration
    DB_NAME = "gst_fraud.db"
    DATABASE_PATH = os.path.abspath(os.environ.get(
        "GSTGUARD_DATABASE_PATH",
        os.path.join(BASE_DIR, "database", DB_NAME)
    ))
    DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Uploads Configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    ALLOWED_EXTENSIONS = {"csv"}
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB max upload limit

    # Demo Authentication Credentials (for viva review)
    DEMO_USER = os.environ.get("DEMO_USER", "admin")
    DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "admin123")

    # ML Model Weight Settings
    WEIGHT_ISOLATION_FOREST = 0.40
    WEIGHT_LOF = 0.40
    WEIGHT_ZSCORE = 0.20
