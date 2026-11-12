# SkillWallet Internship: Credit Card Approval Prediction Engine

[![Vercel Deployment](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=flat&logo=vercel)](https://github.com/Bhuwinrag/Credit-Card-Skillwallet-Internship)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.1-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-emerald?style=flat&logo=flask)](https://flask.palletsprojects.com)
[![Status](https://img.shields.io/badge/Status-Live%20Production-success)](#)

---

## 🚀 Overview

The **SkillWallet Credit Decision Engine** is an end-to-end Machine Learning web platform designed to automate credit card risk assessment and underwriting decisions in real time.

By analyzing financial stability, demographic attributes, and multi-year credit bureau vintage records, the platform evaluates default probability, assigns risk tiers, and outputs explainable decision factors within milliseconds—significantly reducing manual loan review turnaround while enforcing strict risk controls.

---

## ✨ Key Features

1. **Modern Fintech Landing Page (`/`)**:
   - Live system status indicators and dynamic KPI metrics counter (150K+ records, 85.2% ROC-AUC, <40ms latency).
   - Interactive glassmorphic credit card preview with real-time approval status.
   - Comprehensive machine learning pipeline breakdown.
   - Algorithm benchmarking matrix comparing Decision Trees, Random Forests, and Logistic Regression.
   - 1-Click Interactive Applicant Presets (Prime, Moderate, and Risky profiles).

2. **Interactive Analytics & Model Interpretability Dashboard (`/analytics`)**:
   - **Feature Importance Ranking Chart**: Dynamic horizontal bar chart illustrating the relative weights of the 20 financial and bureau indicators.
   - **Multi-Model Benchmark Visualizer**: Grouped bar chart comparing Accuracy, Precision, F1-Score, and ROC-AUC across algorithms.
   - **Target Class Imbalance & Resolution**: Doughnut chart illustrating the raw 87:13 non-default skew and the impact of cost-sensitive class balancing.
   - **Credit Bureau Window vs. Approval Rate**: Line trend analyzing risk variations as credit observation depth increases.
   - **Seaborn & Matplotlib EDA Gallery**: High-resolution bivariate correlation heatmaps and occupation distribution graphics with financial interpretations.

3. **Streamlined Live Evaluator (`/evaluate`)**:
   - 1-Click Scenario Loaders (Prime Approved, Moderate, High Risk Declined) with URL query parameter support (`?preset=prime`).
   - Categorized 4-section architecture: Demographics, Financial & Assets, Household & Contacts, and Bureau Vintage History.
   - Dynamic real-time calculation of family dependency indices and history windows.

4. **Explainable AI (XAI) Assessment Result (`/predict`)**:
   - Circular SVG probability gauge displaying exact approval / rejection confidence percentages.
   - Risk tier classification (Tier 1 Prime, Tier 2 Moderate, Tier 3 Elevated Risk).
   - Positive credit drivers and risk consideration flags personalized to the submitted profile.
   - Formatted applicant summary card and automated underwriting policy recommendations.
   - One-click print / PDF export option for risk audit logs.

5. **REST API Endpoints**:
   - `POST /api/predict`: Programmatic JSON inference endpoint for third-party loan origination systems.
   - `GET /api/analytics`: JSON endpoint returning model performance metrics and dataset distribution statistics.

---

## 📊 Model Benchmarking & Performance

Evaluated on a 20% holdout test partition of 150,000 merged bureau and applicant records:

| Algorithm | Balancing Strategy | Accuracy | Precision | F1-Score | ROC-AUC | Deployment Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Balanced Decision Tree** | `class_weight='balanced'` | **74.63%** | **87.97%** | **78.55%** | **0.8524** | **Production Deployed** |
| **Random Forest Ensemble** | `balanced_subsample` | 74.68% | 87.98% | 78.59% | 0.8518 | Benchmark Candidate |
| **Logistic Regression** | Unweighted Baseline | 87.08% | 75.83% | 81.07% | 0.6239 | Baseline |

> **Why Balanced Decision Trees?**  
> In credit risk assessment, false approvals (approving an applicant who defaults) carry high financial loss. Standard models predict the majority non-default class. Applying cost-sensitive balanced weights elevates precision to **87.97%** and achieves a **0.8524 ROC-AUC**, ensuring conservative and regulatory-compliant underwriting decisions.

---

## 🛠 Project Structure

```text
├── app.py                      # Flask backend microservice with web routes & REST API
├── model.pkl                   # Persisted high-precision Decision Tree model
├── data_collection.py          # Structures application_record.csv and credit_record.csv
├── epic2_eda.py                # Univariate & Multivariate Exploratory Data Analysis
├── data_preprocessing.py       # Data cleaning, deduplication, encoding & merging
├── epic4_model_building.py     # Multi-model training and evaluation harness
├── save_model.py               # Persists balanced model for deployment
├── models.py                   # SQLAlchemy schema mapping for credit scoring entities
├── vercel.json                 # Vercel Serverless Edge deployment configuration
├── requirements.txt            # Production dependencies (Flask, scikit-learn, numpy)
├── plots/                      # Generated Seaborn and Matplotlib visualization charts
│   ├── multivariate_heatmap.png
│   ├── univariate_occupation.png
│   ├── multivariate_income_edu.png
│   └── univariate_income.png
└── templates/                  # Modern responsive frontend templates
    ├── landing.html            # High-converting Fintech landing page
    ├── evaluate.html           # 1-Click sandbox evaluation form
    ├── analytics.html          # Interactive Chart.js EDA & metrics dashboard
    ├── result.html             # Explainable AI (XAI) risk assessment result
    └── index.html              # Mirror of landing page for root compatibility
```

---

## 💻 Running the Application Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Bhuwinrag/Credit-Card-Skillwallet-Internship.git
   cd Credit-Card-Skillwallet-Internship
   ```

2. **Activate your Python environment and install dependencies:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Launch the Flask application:**
   ```bash
   python app.py
   ```

4. **Open your browser and navigate to:**
   - **Landing Page**: `http://127.0.0.1:5000/`
   - **Evaluator**: `http://127.0.0.1:5000/evaluate`
   - **Analytics & EDA**: `http://127.0.0.1:5000/analytics`

---

## ☁️ Deployment on Vercel

This repository is configured for serverless deployment on Vercel using `@vercel/python`:

1. Import this repository into your [Vercel Dashboard](https://vercel.com).
2. Vercel automatically detects `vercel.json` and `requirements.txt`.
3. The serverless function compiles and routes all requests directly to `app.py`.
4. Continuous Deployment is triggered on every `git push` to `main`.

---

## 📄 License & Credits

Developed as part of the **SkillWallet AI & ML Internship Program**.  
Designed and engineered by [Bhuwinrag](https://github.com/Bhuwinrag).
