# AI-NIDS: AI-Powered Network Intrusion Detection and Threat Analysis System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-208BFE.svg)](https://xgboost.readthedocs.io/)
[![Hardware](https://img.shields.io/badge/Optimized-Apple%20Silicon%20(M1--M4)-black.svg)](#)

An academic research prototype and threat analysis platform developed for the term paper:

> **"AI in Cybersecurity and Defense: Risks, Benefits, and Frameworks"**

AI-NIDS ingests network traffic captures, executes data-leakage-free preprocessing, trains baseline and ensemble machine learning algorithms, computes security-critical evaluation metrics (emphasizing False Negative Rate over raw accuracy), explains model decisions via SHAP/feature importance, and delivers an interactive Streamlit cybersecurity dashboard.

---

## Table of Contents

- [AI-NIDS: AI-Powered Network Intrusion Detection and Threat Analysis System](#ai-nids-ai-powered-network-intrusion-detection-and-threat-analysis-system)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. Research Motivation \& Theoretical Framework](#2-research-motivation--theoretical-framework)
  - [3. Core Objectives](#3-core-objectives)
  - [4. Dataset Specifications (CICIDS2017)](#4-dataset-specifications-cicids2017)
    - [How to Download Raw CSVs:](#how-to-download-raw-csvs)
  - [5. System Architecture \& Project Structure](#5-system-architecture--project-structure)
  - [6. Installation \& Setup (macOS / Apple Silicon)](#6-installation--setup-macos--apple-silicon)
    - [Step 1: Clone or Navigate to the Project Root](#step-1-clone-or-navigate-to-the-project-root)
    - [Step 2: Initialize Virtual Environment (Python 3.11)](#step-2-initialize-virtual-environment-python-311)
    - [Step 3: Upgrade pip and Install Dependencies](#step-3-upgrade-pip-and-install-dependencies)
  - [7. Running the Interactive Dashboard](#7-running-the-interactive-dashboard)
    - [Dashboard Navigation:](#dashboard-navigation)
  - [8. Executing Automated Research Experiments](#8-executing-automated-research-experiments)
    - [Interactive Research Notebook:](#interactive-research-notebook)
  - [9. Running Unit Tests](#9-running-unit-tests)
  - [10. Interpreting Cybersecurity Evaluation Metrics](#10-interpreting-cybersecurity-evaluation-metrics)
  - [11. Explainable AI \& Causality Caveats](#11-explainable-ai--causality-caveats)
  - [12. Research Limitations \& Future Directions](#12-research-limitations--future-directions)
    - [Current Limitations:](#current-limitations)
    - [Future Directions:](#future-directions)
  - [Citation](#citation)

---

## 1. Project Overview

AI-NIDS classifies bidirectional network traffic flows into:
- **`BENIGN`**: Normal, legitimate operational network communications.
- **`MALICIOUS`**: Hostile network activity (DDoS, DoS, PortScan, BruteForce, Botnet, Infiltration).

The system compares three foundational machine learning algorithms:
1. **Model A: Logistic Regression** — Interpretable linear baseline with L2 regularization.
2. **Model B: Random Forest Classifier** — Non-linear bagging ensemble of decision trees with Gini split optimization.
3. **Model C: XGBoost Classifier** — Scalable gradient-boosted decision trees with second-order gradient approximation.

---

## 2. Research Motivation & Theoretical Framework

Intrusion Detection Systems (IDS) operate under extreme asymmetric risk:
- **Class Imbalance:** Benign enterprise network traffic typically accounts for 90%–99% of total packet captures. A naive classifier predicting all traffic as `BENIGN` achieves 99% accuracy while having a **100% False Negative Rate**.
- **High Cost of Missed Attacks (False Negatives):** In cyber defense, an undetected intrusion (False Negative) allows adversaries to achieve initial access, establish persistence, elevate privileges, and exfiltrate confidential data.
- **Alert Fatigue (False Positives):** A detector that generates too many false alarms overwhelms Security Operations Center (SOC) analysts, increasing dwell time for real attacks.
- **Black-Box Opacity:** Autonomous AI systems cannot be trusted in defense operations without explainable feature attributions that verify the model is learning genuine threat characteristics rather than statistical noise.

---

## 3. Core Objectives

- Develop a modular, reproducible Python machine learning pipeline without data leakage.
- Provide automatic multi-class to binary label conversion preserving original labels.
- Implement cost-sensitive class weighting to address extreme traffic imbalance.
- Implement comprehensive evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC, FPR, FNR, and wall-clock latency.
- Provide game-theoretic SHAP and tree-based feature importance for XAI transparency.
- Provide a full-featured Streamlit cyber defense dashboard supporting novel CSV threat inspection.

---

## 4. Dataset Specifications (CICIDS2017)

The system targets the **CICIDS2017** benchmark dataset published by the Canadian Institute for Cybersecurity (UNB):
- **Official Portal:** [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Features:** 78 statistical network flow descriptors (Flow Duration, Packet Lengths, Inter-Arrival Times, Flags).
- **Attack Types:** Brute Force (FTP/SSH), DoS (Slowloris, Slowhttptest, Hulk, GoldenEye), Heartbleed, Web Attacks (XSS, SQLi), Infiltration, Botnet, PortScan, and DDoS (LOIC).

### How to Download Raw CSVs:
1. Visit the UNB portal and download the `MachineLearningCVE` archive.
2. Unzip the CSV files and place one or more files in:
   ```bash
   data/raw/
   ```
3. Alternatively, use the built-in **Benchmark Sample Generator** (`data/sample/`) to immediately test the system with full 78-feature fidelity without downloading multi-gigabyte files.

---

## 5. System Architecture & Project Structure

```
ai-nids/
├── app.py                          # Streamlit application entry point
├── config/
│   ├── __init__.py
│   └── settings.py                 # Central configurations, paths, hyperparameter presets
├── data/
│   ├── raw/                        # Storage for downloaded CICIDS2017 CSVs
│   ├── processed/                  # Processed and split matrices
│   ├── sample/                     # Pre-packaged 78-column benchmark sample
│   └── README.md                   # Detailed dataset guide
├── models/
│   └── trained/                    # Joblib serialized model bundles
├── notebooks/
│   └── research_experiments.ipynb  # End-to-end Jupyter research notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # CSV ingestion, chunking, label detection & binarization
│   ├── preprocessing.py            # Leakage-free split, imputation, scaling, deduplication
│   ├── feature_engineering.py      # Variance filtering, correlation matrix analysis
│   ├── model_trainer.py            # Model A (LR), Model B (RF), Model C (XGBoost) lifecycle
│   ├── evaluation.py               # Comprehensive metrics (FNR, FPR, F1, ROC-AUC, latency)
│   ├── explainability.py           # Feature importance, SHAP values, domain taxonomy
│   ├── prediction.py               # Threat classifier for novel flow captures
│   ├── report_generator.py         # Empirical report exporter (Markdown, JSON, CSV)
│   ├── sample_data.py              # Realistic benchmark sample generator
│   └── experiments_runner.py       # Automated runner for Experiments 1-5
├── dashboard/
│   ├── __init__.py
│   ├── overview.py                 # Telemetry metrics, best model, executive cards
│   ├── dataset_analysis.py         # Class distributions, missing/dupe diagnostics, histograms
│   ├── model_training.py           # Split configuration, interactive training execution
│   ├── model_comparison.py         # Comparative charts, ROC curves, confusion matrices
│   ├── threat_analysis.py          # Novel CSV traffic inspection & threat triage
│   ├── explainability.py           # Feature importance & SHAP explainability
│   └── research_report.py          # Paper summary preview & artifact downloads
├── results/
│   ├── metrics/                    # Saved experimental CSV tables
│   ├── charts/                     # Generated comparison charts
│   └── reports/                    # Exported research summaries (MD/JSON)
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py       # Data leakage & imputation unit tests
│   ├── test_evaluation.py          # Metric calculation & zero-division unit tests
│   └── test_prediction.py          # Model training & inference pipeline unit tests
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive documentation
└── .gitignore                      # Git exclusion rules
```

---

## 6. Installation & Setup (macOS / Apple Silicon)

The project is fully optimized for macOS, including Apple Silicon (M1/M2/M3/M4) and standard x86 systems.

### Step 1: Clone or Navigate to the Project Root
```bash
cd "untitled folder"
```

### Step 2: Initialize Virtual Environment (Python 3.11)
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3: Upgrade pip and Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Running the Interactive Dashboard

Launch the Streamlit web dashboard locally:

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`.

### Dashboard Navigation:
- **Dashboard Overview:** High-level threat exposure rates, best-performing model badge, system status.
- **Dataset Analysis:** Preview flows, check duplicate/infinite values, inspect attack distributions.
- **Model Training:** Configure test split (10%–40%), choose random seed, train LR, RF, and XGBoost.
- **Model Comparison:** Interactive multi-metric bar charts, ROC-AUC curves, confusion matrices.
- **Threat Analysis:** Upload new network flow captures, score malicious probability, download SOC triage CSV.
- **Explainability (XAI):** Top 15 influential features, game-theoretic SHAP scores, domain interpretations.
- **Research Report:** Export empirical findings directly into Markdown, JSON, and CSV for publication.

---

## 8. Executing Automated Research Experiments

To execute all 5 mandatory research experiments and generate verified experimental outputs:

```bash
python3 -m src.experiments_runner
```

This generates:
- `results/metrics/experiment_1_model_comparison.csv`
- `results/metrics/experiment_2_class_imbalance.csv`
- `results/metrics/experiment_3_fnr_vs_accuracy.csv`
- `results/metrics/experiment_4_top_features.csv`
- `results/metrics/experiment_5_efficiency_tradeoff.csv`
- `results/reports/final_research_experiments_summary.md`
- `results/reports/final_research_experiments_summary.json`

### Interactive Research Notebook:
You can also run or present the research notebook in VS Code or JupyterLab:
```bash
jupyter notebook notebooks/research_experiments.ipynb
```

---

## 9. Running Unit Tests

The test suite validates data leakage prevention, stratified splitting, metric calculation, zero-division edge cases, and inference engines:

```bash
python3 -m pytest tests/ -v
```

Expected output:
```
tests/test_evaluation.py::test_metrics_calculation_standard PASSED
tests/test_evaluation.py::test_zero_division_graceful_handling PASSED
tests/test_evaluation.py::test_compile_comparison_table PASSED
tests/test_prediction.py::test_model_trainer_train_and_save PASSED
tests/test_prediction.py::test_threat_classifier_inference PASSED
tests/test_preprocessing.py::test_data_loader_load_and_detect_label PASSED
tests/test_preprocessing.py::test_label_binarization PASSED
tests/test_preprocessing.py::test_preprocessing_pipeline_prevents_leakage PASSED
tests/test_preprocessing.py::test_transform_unseen_data PASSED
============================== 9 passed in 1.7s ===============================
```

---

## 10. Interpreting Cybersecurity Evaluation Metrics

| Metric                        | Formula                           | Cybersecurity Operational Significance                                                  |
| ----------------------------- | --------------------------------- | --------------------------------------------------------------------------------------- |
| **False Negative Rate (FNR)** | $FN / (FN + TP)$                  | **Highest Risk.** Missed intrusions that breached the perimeter undetected.             |
| **False Positive Rate (FPR)** | $FP / (FP + TN)$                  | Legitimate business sessions flagged as hostile; drives SOC alert fatigue.              |
| **Recall (Sensitivity)**      | $TP / (TP + FN)$                  | Proportion of total attack campaigns successfully identified.                           |
| **Precision**                 | $TP / (TP + FP)$                  | Confidence that an alerted flow is genuinely hostile.                                   |
| **F1-Score**                  | $2 \cdot \frac{P \cdot R}{P + R}$ | Harmonic mean balancing precision and recall under severe class imbalance.              |
| **ROC-AUC**                   | Area under ROC                    | Global threshold-independent discriminative power across all operational cutoffs.       |
| **Accuracy**                  | $(TP + TN) / Total$               | **Misleading in IDS.** Can be 99% while missing 100% of attacks in imbalanced captures. |

---

## 11. Explainable AI & Causality Caveats

In the Explainability module:
- **Tree Feature Importance (Gini / MDI):** Measures total variance reduction brought by a feature across all tree splits.
- **SHAP (SHapley Additive exPlanations):** Game-theoretic attribution allocating fair credit to each feature for pushing predictions toward `MALICIOUS` or `BENIGN`.

> **Critical Methodological Distinction (Correlation vs. Causality):**
> High feature importance demonstrates strong statistical association within the dataset, but **does NOT constitute proof of root-cause causality**. For example, `Destination Port 80` may correlate with DoS attacks because automated tools happened to target port 80 during data capture, not because port 80 itself causes malicious behavior. Attackers can alter packet timing or pad payloads to evade detectors that overfit to non-causal statistical artifacts.

---

## 12. Research Limitations & Future Directions

### Current Limitations:
1. **Flow-Level vs Packet-Level:** Analysis operates on pre-extracted statistical flow summaries rather than raw Layer 2/3 packet payloads.
2. **Stationary Assumption:** Models assume static feature distributions; in reality, threat actor tactics undergo adversarial drift.
3. **Binary Abstraction:** Distinguishing `BENIGN` vs `MALICIOUS` obscures nuanced multi-stage attack playbooks (e.g. pivoting vs exfiltration).

### Future Directions:
1. **Streaming Online Learning:** Integrating continual learning algorithms to adapt to concept drift without catastrophic forgetting.
2. **Adversarial Robustness Certification:** Evaluating model susceptibility to evasion attacks via gradient-based perturbation crafting.
3. **Multi-Class Threat Playbooks:** Granular automated triage mapping detected flows to MITRE ATT&CK tactics and techniques.

---

## Citation

```bibtex
@article{nids_academic2026,
  title={AI in Cybersecurity and Defense: Risks, Benefits, and Frameworks},
  author={Kumar, Akshit, Mishra, Adhiraj, Khantwal, Kavya},
  journal={Academic Research Repository},
  year={2026}
}
```