# GSTGuard AI
### Intelligent GST Invoice Fraud Detection and Compliance Copilot

**Track:** AI for Productivity & Automation

GSTGuard AI is an AI-powered invoice anomaly detection and compliance assistance system designed to help Micro, Small and Medium Enterprises (MSMEs) identify suspicious GST invoice patterns, prioritize risk-based reviews, and streamline invoice verification.

## The Problem

MSMEs often process large volumes of invoices, making manual verification time-consuming and difficult to scale. Duplicate invoice records, unusual invoice values, abnormal supplier activity, and potential tax calculation inconsistencies can go unnoticed without systematic checks.

Traditional manual workflows also make it difficult to prioritize invoices that need immediate attention.

## Our Solution

GSTGuard AI combines machine learning and statistical anomaly detection with an interactive web application to automate invoice screening and support evidence-based investigation.

The system analyzes available invoice data, generates anomaly risk scores, highlights unusual patterns, and presents findings through a centralized dashboard. Users can investigate flagged invoices and review supplier-level risk indicators before making compliance decisions.

**Our goal is to turn manual invoice screening into a more efficient, explainable, and auditable review workflow.**

## Key Features

### 1. Executive Dashboard
Provides an overview of invoice activity, anomaly indicators, and priority review tasks.

### 2. AI-Powered Anomaly Detection
Uses three complementary techniques:
- **Isolation Forest:** Detects observations that differ from typical data patterns.
- **Local Outlier Factor (LOF):** Identifies observations that are unusual compared with their local neighborhood.
- **Z-Score Analysis:** Flags values that deviate significantly from statistical averages.

An ensemble scoring approach combines detection signals to help prioritize invoices for review.

### 3. Priority Review Queue
Organizes suspicious invoices by risk level so users can focus on transactions requiring further verification.

### 4. Explainable Invoice Investigation
Presents available model results, invoice-level indicators, benchmark comparisons, and recommended verification steps to help users understand why a record may require attention.

### 5. Compliance Copilot
Provides an interface for invoice-related questions and compliance assistance using the application's implemented functionality and available data.

### 6. Supplier Risk Profiles
Helps users examine supplier-related invoice patterns and identify unusual activity that may warrant further investigation.

### 7. Upload & Analyze
Supports analysis of invoice datasets through CSV upload and data-processing workflows.

### 8. Reports and Audit Support
Provides access to analysis results and available reports to support documentation and review.

## How It Works

1. **Upload:** Provide invoice data in the supported CSV format.
2. **Validate:** Process the data and prepare the required features.
3. **Analyze:** Apply Isolation Forest, LOF, and statistical anomaly detection.
4. **Score:** Combine model signals using the existing ensemble scoring logic.
5. **Investigate:** Review flagged invoices, available explanations, and supplier risk indicators.
6. **Verify:** Use supporting documents and recommended checks to investigate potential irregularities.
7. **Report:** Access available analysis and reporting functionality.

## Technology Stack

| Layer | Technologies |
| Frontend | HTML, CSS, JavaScript, Bootstrap |
| Backend | Python, Flask |
| Machine Learning | Scikit-learn, Isolation Forest, LOF |
| Statistical Analysis | Z-score |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| Data Input | CSV files |

## Architecture

```text
Invoice Dataset
      |
      v
Data Validation and Preprocessing
      |
      v
Feature Engineering
      |
      v
+-------------------------------+
| Isolation Forest              |
| Local Outlier Factor          |
| Z-Score Analysis              |
+-------------------------------+
      |
      v
Ensemble Risk Scoring
      |
      v
Flask Application + SQLite
      |
      v
Dashboard and Investigation
      |
      +--> Priority Review Queue
      +--> Supplier Risk Profiles
      +--> Compliance Copilot
      +--> Reports and Audit Support
```

## Getting Started

### Prerequisites
- Python installed
- pip package manager
- Git (optional)

### Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

Open the local URL printed by Flask, typically:

`http://127.0.0.1:5000`

Follow the project's existing configuration and data requirements if they differ from these general instructions.

## Intended Impact

GSTGuard AI aims to help MSMEs:

- Reduce repetitive manual invoice screening.
- Prioritize invoices that warrant closer inspection.
- Improve visibility into unusual invoice and supplier patterns.
- Support more consistent verification workflows.
- Organize findings for follow-up and audit preparation.

The system supports human decision-making rather than replacing professional compliance review.

## Limitations

- An anomaly is not proof of fraud.
- Detection quality depends on the available data and model configuration.
- Synthetic-data evaluation may not represent real-world performance.
- Compliance findings require verification against source documents and applicable regulations.

## Future Scope

- Stronger invoice-level explanations and evidence traceability.
- More comprehensive GST calculation and duplicate checks.
- Persistent investigation history and reviewer audit trails.
- Improved model evaluation using appropriately labelled datasets.
- Secure deployment and role-based access controls.
