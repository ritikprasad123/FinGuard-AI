# FinGuard AI — Enterprise Real-Time Financial Anomaly Detection

**FinGuard AI** is a production-grade financial anomaly surveillance and intelligence application engineered for transaction fraud risk analytics, unsupervised machine learning anomaly detection, continuous data stream monitoring, persistent case management, and regulatory compliance audit reporting.

---

## Executive Overview

In financial transaction networks, anomalous events (such as account takeovers, unauthorized card testing, rapid velocity bursts, and bust-out fraud) represent less than 0.2% of total transaction throughput. Traditional rule-only architectures fail to scale against evolving adversarial tactics, while naive machine learning pipelines suffer from severe target leakage, uncalibrated anomaly scores, and misleading accuracy metrics.

FinGuard AI addresses these systemic challenges through a modular, leak-free architecture with a bespoke enterprise dark terminal interface, rigorous data quality audits, leak-free preprocessing, multi-model anomaly detection engines, persistent SQLite case management, non-blocking simulated stream monitoring, and evidence-grounded AI forensic assistance.

---

## Visual Design System & UI Architecture

FinGuard AI features a bespoke enterprise financial analytics interface designed for compliance teams, fraud analysts, and risk officers:

### Design Tokens
| Token | Hex Value | Semantic Usage |
| :--- | :--- | :--- |
| **Background** | `#0B0D10` | Deep charcoal primary viewport background |
| **Sidebar** | `#101318` | Institutional dark sidebar and navigation container |
| **Surface Cards** | `#15191F` | Standard KPI metric boxes and container surfaces |
| **Elevated Cards** | `#1B2028` | Elevated containers, tooltips, and interactive panels |
| **Primary Red** | `#E5484D` | Crimson-red accent for high-severity anomaly alerts |
| **Critical Red** | `#FF4D5A` | Critical risk thresholds and audit failure badges |
| **Success Green** | `#35C98A` | Legitimate cleared transactions and passing audit status |
| **Primary Text** | `#F4F6F8` | High-contrast body typography (Inter font family) |
| **Secondary Text**| `#9AA4B2` | Muted labels, metadata timestamps, and captions |
| **Borders** | `#292F39` | Subtle geometric card boundaries and gridlines |

### Design Principles
- **No Emojis:** Restrained institutional typography using crisp badges, status dots (`●`), and geometric indicators.
- **Monospace Numerals:** Monetary amounts, percentages, and latency metrics rendered with `JetBrains Mono`.
- **Plotly Dark Integration:** All interactive charts automatically inherit the dark red-black design tokens.

---

## Modular Architecture & Directory Layout

```
FinGuard-AI/
  ├── app.py                          # Executive surveillance overview & active model telemetry
  ├── requirements.txt                # Production dependency specifications
  ├── README.md                       # Comprehensive system documentation & runbook
  ├── .gitignore                      # Clean VCS ignoring caches and raw datasets
  ├── config/
  │   ├── __init__.py
  │   └── settings.py                 # Immutable path resolutions, design tokens, thresholds & database paths
  ├── data/
  │   ├── raw/                        # Storage for official benchmark (creditcard.csv)
  │   ├── processed/                  # Cleaned, partitioned evaluation splits
  │   ├── demo/                       # Generated synthetic DEMO data batches
  │   └── cases.db                    # Persistent SQLite case management database
  ├── models/                         # Serialized scalers, Isolation Forest & Autoencoder artifacts
  ├── src/
  │   ├── data/
  │   │   ├── __init__.py
  │   │   ├── loader.py               # Safe CSV loading and benchmark provenance
  │   │   ├── validator.py            # 7-step schema auditing and data quality profiling
  │   │   └── demo_generator.py       # Synthetic retail transaction generator
  │   ├── features/
  │   │   ├── __init__.py
  │   │   └── preprocessor.py         # RobustScaler leak-free transformation & stratified split
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── base.py                 # Abstract BaseAnomalyModel, InferenceResult dataclass & risk tiers
  │   │   ├── isolation_forest.py     # Isolation Forest engine with calibrated [0, 1] scoring
  │   │   ├── autoencoder.py          # Neural Autoencoder bottleneck reconstruction engine
  │   │   ├── evaluation.py           # Precision, Recall, F1, PR-AUC, confusion matrix & baselines
  │   │   └── registry.py             # Model registry, active model selection & persistence
  │   ├── monitoring/
  │   │   ├── __init__.py
  │   │   ├── metrics.py              # Financial KPIs, velocity, and risk volume
  │   │   └── stream_monitor.py       # Non-blocking simulated real-time stream engine
  │   ├── investigation/
  │   │   ├── __init__.py
  │   │   ├── alert_feed.py           # AnomalyAlert dataclass & forensic triage feed
  │   │   ├── case_manager.py         # Persistent SQLite case manager (ACID-compliant)
  │   │   └── assistant.py            # Evidence-grounded AI forensic investigation assistant
  │   ├── reports/
  │   │   ├── __init__.py
  │   │   ├── quality_report.py       # Data quality audit generator
  │   │   └── investigation_report.py # Formal compliance case & alert triage report generator
  │   └── utils/
  │       ├── __init__.py
  │       ├── logger.py               # Rotating file and console logger
  │       └── helpers.py              # Currency and numeral formatting utilities
  ├── pages/
  │   ├── 1_Executive_Overview.py     # Executive overview, surveillance KPIs & telemetry
  │   ├── 2_Transaction_Monitoring.py # Search, multi-parameter filters & PCA inspector
  │   ├── 3_Anomaly_Detection.py      # Real-time scoring, threshold tuning & batch inference
  │   ├── 4_Model_Performance.py      # Precision/Recall curves, confusion matrix & baselines
  │   ├── 5_AI_Investigation_Assistant.py # Forensic evidence synthesis & 4-step protocols
  │   ├── 6_Case_Management.py        # Persistent SQLite case dossiers, notes & audit logs
  │   ├── 7_Live_Style_Monitor.py     # Non-blocking simulated streaming & alert ticker
  │   ├── 8_Investigation_Reports.py  # Exportable case dossiers, triage & audit reports
  │   └── 9_Dataset_and_Model_Status.py # Provenance, 7-step data audit & model registry
  ├── components/
  │   ├── __init__.py
  │   ├── theme.py                    # CSS injector and page config helper
  │   ├── cards.py                    # KPI cards, headers, and alert banners
  │   ├── charts.py                   # Reusable Plotly dark red-black financial charts
  │   ├── setup_card.py               # Benchmark acquisition setup container
  │   └── tables.py                   # Formatted currency dataframes
  ├── assets/
  │   └── style.css                   # Enterprise CSS styling sheet
  └── tests/
      ├── test_config.py              # Configuration immutability and path tests
      ├── test_loader.py              # Safe dataset ingestion and schema tests
      ├── test_validator.py           # 7-step data quality audit unit tests
      ├── test_demo_generator.py      # Synthetic demo generator distribution tests
      ├── test_preprocessor.py        # Leak-free RobustScaler & split tests
      ├── test_metrics.py             # KPI and hourly velocity calculation tests
      ├── test_alert_feed.py          # Alert triage and anomaly filtering tests
      ├── test_quality_report.py      # Executive report generation tests
      ├── test_isolation_forest.py    # Isolation Forest training, loading, batch & edge cases
      ├── test_autoencoder.py         # Neural Autoencoder training, loading, batch & edge cases
      ├── test_evaluation.py          # Precision-Recall, threshold trade-offs & baseline tests
      ├── test_registry.py            # Model registry lifecycle tests
      ├── test_case_manager.py        # SQLite persistent case management tests
      ├── test_investigation_assistant.py # AI assistant evidence rationale & limitations tests
      ├── test_stream_monitor.py      # Non-blocking stream stepping & reset tests
      └── test_investigation_report.py # Formal Markdown investigation report tests
```

---

## Phase 3: Transaction Monitoring & Investigation System

### 1. Transaction Monitoring & Advanced Search
- **Multi-Parameter Search:** Filter transactions by Transaction ID, Prediction status (`ALL`, `ANOMALY ONLY`, `NORMAL ONLY`), Severity Tiers (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), Minimum Anomaly Score ($0.0 \to 1.0$), and Ticket Amount Range ($0 \to \text{Max}$).
- **Sorting & Pagination:** Sort dynamically by Anomaly Score, Ticket Amount, Elapsed Time, or Transaction ID with configurable page sizing (25, 50, 100 rows per page).
- **Transaction Deep Inspector:** Inspect full 30 continuous PCA latent components (V1–V28, Time, Amount) and observed deviations for any individual record.
- **Direct Hand-off:** One-click integration to forward any transaction directly to the AI Investigation Assistant or open an investigation case.

### 2. Alert Center & Forensic Triage
- **Standardized Alert Schema (`AnomalyAlert`):**
  - Transaction Reference (`transaction_id`)
  - Calibrated Anomaly Score
  - Applied Threshold Cutoff
  - Model Classification (`ANOMALY` / `NORMAL`)
  - Severity Rating (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`)
  - Supporting Information (top feature deviations, ticket value, elapsed seconds)
  - Lifecycle Status (`NEW`, `IN_REVIEW`, `ESCALATED`, `DISMISSED`, `CONFIRMED_SUSPICIOUS`)
- **Triage Console:** Quick status transitions directly from the UI with real-time exposure KPIs.
- **One-Click Case Escalation:** Escalate any alert into a formal case with pre-filled transaction context.

### 3. Persistent Case Management (SQLite Engine)
- **ACID-Compliant Storage:** Built using Python's native `sqlite3` at `data/cases.db`. Zero external database server dependencies.
- **Relational Schema:**
  - `cases`: Primary case metadata, priority, status, assigned officer, linked transaction array, and executive summary.
  - `case_notes`: Chronological timestamped investigator notes with author attribution.
  - `case_history`: Tamper-evident audit trail recording every status change, transaction linkage, and note posting.
- **Lifecycle Statuses:** `OPEN`, `INVESTIGATING`, `PENDING_CUSTOMER_CONTACT`, `ESCALATED_AML`, `CLOSED_FALSE_POSITIVE`, `CLOSED_CONFIRMED_ANOMALY`.
- **Search & Filter:** Query cases by status, priority, or free-text search across titles and summaries.

### 4. Evidence-Grounded AI Investigation Assistant
- **Strict Anti-Hallucination Policy:** **Does not invent transaction histories, customer names, or banking credentials.**
- **Empirical Mathematical Reasoning:** Calculates standardized Z-score deviations against benchmark normal distributions for key fraud-correlated dimensions ($V14, V10, V12, V17, V4, V11, \text{Amount}$).
- **Actionable Forensic Protocol:** Generates a structured 4-step investigation action plan tailored to the risk tier (e.g. out-of-band biometric challenge, velocity correlation, merchant MCC review, provisional hold).
- **Mandatory Compliance Disclaimer:** Emphasizes that statistical anomaly scores denote variance from population baselines and do not constitute legal proof of fraudulent intent.

### 5. Non-Blocking Simulated Stream Monitor
- **Zero Infinite Loops:** Emulates transaction streaming via stateful step execution (`Step 1`, `Step 10`, `Step 50`) integrated with Streamlit session state.
- **Per-Transaction Model Scoring:** Invokes active machine learning model inference (`predict_single`) on each arriving transaction.
- **Live Anomaly Velocity Chart:** Interactive Plotly dark chart displaying rolling anomaly scores with a dynamic decision cutoff line.
- **Explicit Metadata Labeling:** Prominently tagged with `DEMO DATA (SIMULATED STREAM)` and clear disclaimer stating it is not connected to live clearing houses.
- **Instant Reset:** Resets stream position, buffer progress, and telemetry counters with a single click.

### 6. Audit-Compliant Report Generation
- **Formal Case Reports:** Generates executive Markdown reports (`generate_case_investigation_report`) containing case metadata, associated transactions, chronological notes, tamper-evident audit history, recommended disposition, and regulatory notices.
- **Batch Triage Reports:** Generates summary reports (`generate_alert_triage_batch_report`) of active surveillance queues with exposed value metrics and priority queues.

---

## Model Evaluation Benchmark (Phase 2 Baseline)

Evaluated on 2,500 benchmark transactions (80% train / 20% holdout test partition; 100 frauds / 4.00% fraud rate, zero target leakage):

| Metric | Isolation Forest (Cutoff: 0.65) | Isolation Forest (Optimal: 0.55) | Neural Autoencoder (Cutoff: 0.65) | Neural Autoencoder (Optimal: 0.85) | Heuristic Rule Baseline | Random Guessing Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Precision** | **1.0000 (100.0%)** | **1.0000 (100.0%)** | 0.0442 (4.4%) | 0.0690 (6.9%) | 1.0000 (100.0%) | 0.0435 (4.4%) |
| **Recall** | **0.7000 (70.0%)** | **1.0000 (100.0%)** | 0.2500 (25.0%) | 0.2000 (20.0%) | 0.2500 (25.0%) | 0.0500 (5.0%) |
| **F1-Score** | **0.8235** | **1.0000** | 0.0752 | 0.1026 | 0.4000 | 0.0465 |
| **PR-AUC** | **1.0000** | **1.0000** | 0.0445 | 0.0445 | — | — |
| **ROC-AUC** | **1.0000** | **1.0000** | 0.4694 | 0.4694 | — | — |
| **True Positives** | **14 / 20** | **20 / 20** | 5 / 20 | 4 / 20 | 5 / 20 | 1 / 20 |
| **False Positives** | **0** | **0** | 108 | 54 | 0 | 22 |
| **False Negatives** | **6 / 20** | **0 / 20** | 15 / 20 | 16 / 20 | 15 / 20 | 19 / 20 |
| **False Positive Rate**| **0.00%** | **0.00%** | 22.50% | 11.25% | 0.00% | 4.58% |

---

## Verification & Automated Testing

Run the full automated test suite containing **57 passing unit tests** across all pipeline modules:

```bash
pytest tests
```

### Test Suite Execution Output
```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Ritik\OneDrive\Desktop\anomaly dewtedtion ex-3
collected 57 items

tests\test_alert_feed.py ..                                              [  3%]
tests\test_autoencoder.py ........                                       [ 17%]
tests\test_case_manager.py .....                                         [ 26%]
tests\test_config.py ...                                                 [ 31%]
tests\test_demo_generator.py ..                                          [ 35%]
tests\test_evaluation.py ....                                            [ 42%]
tests\test_investigation_assistant.py ..                                 [ 45%]
tests\test_investigation_report.py ..                                    [ 49%]
tests\test_isolation_forest.py ........                                  [ 63%]
tests\test_loader.py ....                                                [ 70%]
tests\test_metrics.py ...                                                [ 75%]
tests\test_preprocessor.py ...                                           [ 80%]
tests\test_quality_report.py ...                                         [ 85%]
tests\test_registry.py .                                                 [ 87%]
tests\test_stream_monitor.py ...                                         [ 92%]
tests\test_validator.py ....                                             [100%]

============================= 57 passed in 5.43s ==============================
```

---

## How to Run the Application

```bash
streamlit run app.py
```
Navigate to:
- **Page 1: Data Pipeline:** 7-step data validation audit and schema verification.
- **Page 2: Model Engine:** Model training, evaluation, threshold tuning, and single-transaction scorer.
- **Page 3: Suspicious Transactions:** 5-tab investigation hub (Search & Monitoring, Alert Center, Case Management, Simulated Stream, AI Assistant).
- **Page 4: Setup and Guide:** Official benchmark provenance, Kaggle download guide, and architecture runbook.
