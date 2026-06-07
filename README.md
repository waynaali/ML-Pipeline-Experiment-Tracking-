# 🛠️ End-to-End Predictive Maintenance Pipeline with MLflow

This repository contains a production-ready Machine Learning pipeline designed for **Predictive Maintenance**. The project simulates real-world industrial IoT sensor data, systematically trains multiple classification models, tracks experiments, and leverages the **MLflow Model Registry** for advanced version control, model documentation, staging, production deployment, and automated rollbacks.

---

## 🚀 Project Overview

In industrial environments, unexpected equipment failure leads to high operational costs and downtime. This project applies an MLOps-driven approach to:
1. **Generate Telemetry Data:** Synthesizes 10,000 equipment samples tracking Temperature (°C), Vibration (mm/s), Pressure (PSI), RPM, and Equipment Age (Days).
2. **Experiment Tracking:** Compares three different architectures (**Logistic Regression**, **Random Forest**, and **XGBoost**) using MLflow tracking.
3. **Model Registry & Governance:** Manages model lifecycles, moving the champion model through `Staging` and `Production` environments while establishing automated rollback procedures.

---

## 📊 Pipeline Architecture & Lifecycle

The workflow is split into two distinct lifecycle phases across individual notebooks:


```

[ Sensor Data Generation ] ──> [ EDA & Scaling ]
│
▼
[ MLflow Experiment Tracking UI ]
├── Run 1: Logistic Regression
├── Run 2: Random Forest
└── Run 3: XGBoost (Champion: ~0.94 ROC AUC)
│
▼
[ MLflow Model Registry ]
├── Version 1 (XGBoost)  ──> [ Staging Test ] ──> [ Live Production ]
└── Version 2 (RF)       ──> [ Simulated Rollback Target ]

```

---

## 📈 Performance Summary

Based on experimental evaluations tracked via the MLflow dashboard, the models performed as follows (consistent with expected non-linear patterns in multi-variable failure conditions):

| Model Architecture | Accuracy | F1-Score | ROC AUC | Status |
| :--- | :---: | :---: | :---: | :--- |
| **XGBoost Classifier** | **~94.5%** | **~0.91** | **~0.94** | **Promoted to Production (v1)** |
| Random Forest | ~92.1% | ~0.88 | ~0.91 | Registered / Archived (v2) |
| Logistic Regression | ~86.4% | ~0.79 | ~0.86 | Basclined / Unregistered |

*XGBoost achieved the highest ROC AUC and F1-score due to its robust ability to parse complex non-linear combinations of high-temperature and high-vibration boundaries.*

---

## 🛠️ Repository Structure

```text
├── week13_predictive_maintenance.ipynb  # Phase 1: Synthetic generation, EDA, and Model Training
├── week14_model_registry.ipynb         # Phase 2: MLflow Client setup, Staging, Production, and Rollback
├── README.md                            # Documentation
└── .gitignore                           # Prevents committing local tracking binaries (mlruns/)

```

---

## ⚡ Quick Start Guide

### 1. Clone the Workspace

```bash
git clone [https://github.com/your-username/predictive-maintenance-mlflow.git](https://github.com/your-username/predictive-maintenance-mlflow.git)
cd predictive-maintenance-mlflow

```

### 2. Install Dependencies

Make sure you have your virtual environment active, then install the necessary toolchain libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost mlflow

```

### 3. Initialize the MLflow Local Tracking Server

Before running any code notebooks, open a dedicated terminal window and spin up the central tracking UI server:

```bash
mlflow ui

```

> **Note:** Keep this terminal running! Your interactive MLflow steering dashboard will now be accessible live at `http://localhost:5000`.

### 4. Running the Pipeline

* Run `week13_predictive_maintenance.ipynb` to generate data and populate your experiment leaderboard charts.
* Run `week14_model_registry.ipynb` to execute governance transitions, test simulated high-risk inference payloads, and verify the model deployment strategy.

---

## 🔍 Features Implemented

* **Comprehensive EDA:** Includes class distribution tracking, seaborn feature correlation heatmaps, and target-stratified property histograms.
* **Production Inference API Wrapper:** Implements `predict_equipment_failure()`, which dynamically pulls the latest approved `Production` model from the registry to serve real-time hardware telemetry queries.
* **Risk & Integrity Testing:** Verifies staging assets using custom, edge-case hardware payloads simulating extreme parameters to evaluate predictive fidelity before pushing to live nodes.
* **Safe Disaster Recovery (Rollbacks):** Programmatically demonstrates automated infrastructure rollbacks from a newly released `v2` candidate back to a verified baseline version `v1` seamlessly using the `MlflowClient` API state toggles.

```
