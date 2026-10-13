# FinGuard AI: Real-Time Financial Transaction Anomaly Detection System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Tests Passed](https://img.shields.io/badge/Tests-73%20Passed-10B981.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-00D4FF.svg)](LICENSE)

**FinGuard AI** is a production-grade financial transaction surveillance, anomaly detection, real-time risk intelligence, and forensic case management platform. Designed with an institutional dark navy financial terminal interface, leak-free feature pipelines, multi-model unsupervised anomaly detection engines (Isolation Forest, Local Outlier Factor, Neural Autoencoder), persistent SQLite case dossiers, and an evidence-grounded AI Forensic Investigation Assistant.

> **Decision-Support Notice:** Machine-learning risk signals are decision-support indicators. Alerts require appropriate review and are not proof of fraud.

---

## 1. Executive Summary & Problem Formulation

Global payment clearing networks process hundreds of millions of transactions every day. Manually reviewing every transaction is impossible, and genuine consumer behavior exhibits high variance across ticket amounts, time-of-day velocity, and international spending corridors. Furthermore, confirmed fraudulent activity typically represents less than 0.2% of total transaction volume.

FinGuard AI provides continuous, automated machine learning surveillance to detect multivariate statistical anomalies, prioritize alerts for compliance officers, and explain the mathematical drivers behind model flags without exposing sensitive cardholder identities.

### Core Governance Principles
- **Statistical Deviation $\neq$ Confirmed Fraud:** Anomaly scores denote statistical distance from learned population distributions; they are advisory signals for triage, not legal proof of fraud.
- **Human-in-the-Loop Review:** Alerts route into a persistent SQLite case management desk for investigator evaluation and chronological note logging.
- **Strict Leakage Prevention:** The ground-truth `Class` column and transaction ID tokens are completely isolated prior to unsupervised model fitting.
- **Zero Fabrication:** The system never invents cardholder names, historical bank accounts, or unverified claims.

---

## 2. Visual Direction & Design System

FinGuard AI implements an institutional dark navy and electric blue aesthetic tailored for security operations centers (SOC) and risk departments:

### Design Tokens
| Token | Hex Value | Purpose & Semantic Application |
| :--- | :--- | :--- |
| **Application Background** | `#0B0F19` | Dark navy charcoal canvas |
| **Sidebar Canvas** | `#0D1322` | Deep navy persistent navigation container |
| **Surface Card** | `#131B2E` | Primary KPI metric cards, tables, and tab containers |
| **Elevated Surface** | `#1A253D` | Elevated interaction cards and expanded inspection panels |
| **Electric Blue Accent** | `#00D4FF` | Primary active telemetry and focus indicator |
| **Teal Accent** | `#14B8A6` | Balanced metric highlights and distribution charts |
| **Critical Red** | `#F43F5E` | Immediate hold threshold (Score $\ge 0.85$) |
| **Primary Rose** | `#E11D48` | High risk tier alerts (Score $\ge 0.65$) |
| **Warning Amber** | `#F59E0B` | Medium risk triage badges and cautionary indicators |
| **Success Emerald** | `#10B981` | Cleared transactions and verified normal audits |
| **Primary Text** | `#F3F4F6` | Crisp typography styled with Google Fonts `Inter` |
| **Secondary Text** | `#94A3B8` | Soft slate metadata timestamps, subtitles, and captions |
| **Border Color** | `#1E2C4A` | Subtle geometric card boundaries and gridlines |

---

## 3. Production Architecture & 10-Page Suite

FinGuard AI is structured as a modular enterprise application with 10 dedicated functional pages:

```
FinGuard-AI/
├── app.py                             # Streamlit multi-page entry point & persistent telemetry
├── requirements.txt                   # Pinned compatible cloud dependencies
├── README.md                          # Comprehensive documentation & runbook
├── LICENSE                            # Official MIT License
├── .gitignore                         # Strict exclusion of secrets, DBs, and large datasets
├── .streamlit/
│   └── config.toml                    # Dark navy theme tokens and server settings
├── pages/
│   ├── 1_Executive_Overview.py        # Executive KPIs, trend charts & recent alerts
│   ├── 2_Transaction_Explorer.py      # Multi-parameter filter, sort, paginate & CSV export
│   ├── 3_Anomaly_Detection.py         # CSV upload, threshold control & single/batch scorer
│   ├── 4_Model_Lab.py                 # Multi-model benchmarking, confusion matrices & config export
│   ├── 5_Alert_Center.py              # Urgent triage queue, review status & AI Assistant
│   ├── 6_Investigation_Cases.py       # SQLite case dossiers, chronological notes & export
│   ├── 7_Analytics.py                 # Amount distributions, time-series & PCA variance
│   ├── 8_Data_Management.py           # 7-step schema auditing, missing/duplicate reports
│   ├── 9_Live_Demo.py                 # Simulated stream monitor with non-blocking controls
│   └── 10_Settings.py                 # Health checks, dependency diagnostics & threshold config
├── src/
│   ├── config/                        # Immutable settings & design tokens
│   ├── data/                          # CSV loader, 7-step validator & demo generator
│   ├── features/                      # RobustScaler preprocessor & leak-free splitter
│   ├── models/                        # Isolation Forest, LOF & Neural Autoencoder
│   ├── investigation/                 # Alert feed, CaseManager (SQLite) & AI Assistant
│   ├── monitoring/                    # Real-time metrics & simulated stream engine
│   ├── reports/                       # Markdown dossier & quality certificate generators
│   └── utils/                         # Thread-safe logging & helpers
├── components/
│   ├── theme.py                       # CSS injector & layout manager
│   ├── cards.py                       # Reusable KPI metric boxes, badges & banners
│   ├── charts.py                      # Bespoke Plotly dark financial visualizations
│   └── setup_card.py                  # Zero-data guided configuration component
└── tests/                             # 17 automated Pytest suites (73 unit tests)
```

---

## 4. Machine Learning Pipeline & Explainability

### Models Implemented
1. **Isolation Forest (`src/models/isolation_forest.py`):**
   - Recursively partitions random feature splits. Outliers isolate in few partitions.
   - Outputs continuous anomaly scores bounded in $[0.0, 1.0]$.
   - Fully serializable via Joblib.

2. **Local Outlier Factor Novelty Detector (`src/models/lof_detector.py`):**
   - Configured with `novelty=True` to compute local density deviations on unseen streaming transactions.
   - Features bounded memory-safe reference sampling ($\le 5,000$ records) to protect memory and ensure sub-second inference.

3. **Neural Bottleneck Autoencoder (`src/models/autoencoder.py`):**
   - Multi-layer bottleneck network ($30 	o 16 	o 8 	o 16 	o 30$) trained on normal patterns.
   - Anomaly score derived from normalized Mean Squared Error ($	ext{MSE}$).

### Explainability & Forensic Attribution
For every flagged transaction, the **AI Investigation Assistant** (`src/investigation/assistant.py`) calculates:
- Standardized $Z$-scores along latent principal components (V1-V28).
- Ticket amount extremity (percentile comparisons).
- Score delta relative to the active decision cutoff.
- Actionable regulatory investigation checklists (OOB challenge, provisional hold, cardholder contact).

---

## 5. Automated Testing & Verification

The test suite contains 17 test modules verifying all components across 73 unit tests:

```bash
pytest
```

**Test Execution Results:**
```
tests/test_alert_feed.py ....                                            [ 5%]
tests/test_autoencoder.py ........                                       [ 16%]
tests/test_case_manager.py .......                                       [ 26%]
tests/test_config.py ...                                                 [ 30%]
tests/test_demo_generator.py ..                                          [ 32%]
tests/test_evaluation.py ......                                          [ 41%]
tests/test_investigation_assistant.py .......                            [ 50%]
tests/test_investigation_report.py ..                                    [ 53%]
tests/test_isolation_forest.py ........                                  [ 64%]
tests/test_loader.py ....                                                [ 69%]
tests/test_lof_detector.py .....                                         [ 76%]
tests/test_metrics.py ...                                                [ 80%]
tests/test_preprocessor.py ...                                           [ 84%]
tests/test_quality_report.py ...                                         [ 89%]
tests/test_registry.py .                                                 [ 90%]
tests/test_stream_monitor.py ...                                         [ 94%]
tests/test_validator.py ....                                             [100%]

============================= 73 passed in 6.22s ==============================
```

---

## 6. Installation & Local Execution

### Prerequisites
- Python 3.11, 3.12, or 3.14
- Git

### Quickstart
```bash
# Clone the repository
git clone https://github.com/Ritikprasad123/FinGuard-AI.git
cd FinGuard-AI

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt

# Launch FinGuard AI
streamlit run app.py
```

The application automatically initializes with high-fidelity demonstration data upon initial launch with zero manual setup required.

---

## 7. Streamlit Community Cloud Deployment

1. Fork or push this repository to GitHub: `https://github.com/Ritikprasad123/FinGuard-AI`.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and click **New app**.
4. Configure the deployment settings:
   - **Repository:** `Ritikprasad123/FinGuard-AI`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy!**
6. FinGuard AI starts immediately with synthetic demo transactions and pre-trained model artifacts.

---

## 8. Author & License

- **Author:** Ritik Prasad ([@Ritikprasad123](https://github.com/Ritikprasad123))
- **Repository:** [https://github.com/Ritikprasad123/FinGuard-AI](https://github.com/Ritikprasad123/FinGuard-AI)
- **License:** [MIT License](LICENSE)
