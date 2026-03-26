# Nigerian Bank Customer Churn Prediction
### Group 18 — Master's Capstone Project

A complete end-to-end machine learning pipeline for predicting customer churn at a Nigerian commercial bank. The project covers exploratory data analysis, feature engineering, model training, hyperparameter tuning, model explainability (SHAP), and a full business ROI analysis.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [Dataset](#3-dataset)
4. [Project Structure](#4-project-structure)
5. [Dependencies](#5-dependencies)
6. [Pipeline Walkthrough](#6-pipeline-walkthrough)
7. [Feature Engineering](#7-feature-engineering)
8. [Models & Results](#8-models--results)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [Model Explainability (SHAP)](#10-model-explainability-shap)
11. [Key Business Insights](#11-key-business-insights)
12. [ROI Analysis](#12-roi-analysis)
13. [Recommendations](#13-recommendations)
14. [How to Run](#14-how-to-run)

---

## 1. Project Overview

| Item | Detail |
|------|--------|
| **Domain** | Retail Banking — Nigerian Market |
| **Task** | Binary Classification (Churn vs. Retained) |
| **Dataset** | 10,000 customer records, 35 raw features |
| **Models** | Logistic Regression, Random Forest, Gradient Boosting |
| **Best CV F1-Score** | 0.8352 (Tuned Random Forest) |
| **Best ROC-AUC** | 0.8436 (Logistic Regression) |
| **Estimated Campaign ROI** | 65,660% |

---

## 2. Business Problem

Customer churn is one of the costliest problems in retail banking. Acquiring a new customer costs 5–7× more than retaining an existing one. Identifying customers who are likely to leave before they actually do allows the bank to launch targeted, cost-effective retention campaigns.

**Goals set for this project:**
- Achieve F1-Score > 0.75
- Achieve ROC-AUC > 0.85
- Provide interpretable, actionable outputs for the retention team

---

## 3. Dataset

**File:** `nigerian_bank_churn_realistic_10k.xlsx`

| Property | Value |
|----------|-------|
| Rows | 10,000 |
| Raw Features | 35 |
| Target Variable | `CHURN_FLAG` (0 = Retained, 1 = Churned) |
| Churned Customers | 2,362 (23.62%) |
| Retained Customers | 7,638 (76.38%) |
| Class Imbalance Ratio | 3.23 : 1 |
| Missing Values | None |
| Duplicate Records | None |

### Feature Groups

| Group | Features |
|-------|----------|
| **Demographics** | STATE, GENDER, AGE, OCCUPATION, MARITAL_STATUS, EDUCATION_LEVEL, ACCOUNT_TYPE |
| **Account** | ACCOUNT_TENURE_MONTHS, NUM_PRODUCTS, HAS_CREDIT_CARD, HAS_LOAN, LOAN_DEFAULT |
| **Transactions** | AVG_MONTHLY_TRANSACTIONS, TRANSACTION_TREND, AVG_TRANSACTION_AMOUNT |
| **Balance** | CURRENT_BALANCE, BALANCE_CHANGE_3M, MIN_BALANCE_VIOLATIONS |
| **Digital Engagement** | HAS_MOBILE_APP, APP_LOGIN_FREQUENCY, USES_INTERNET_BANKING, DIGITAL_TRANSACTION_RATIO |
| **Customer Service** | COMPLAINTS_6M, COMPLAINTS_RESOLVED_RATIO, CS_CALLS_6M, BRANCH_VISITS_6M |
| **Competition** | HAS_OTHER_BANK_ACCOUNTS, NUM_OTHER_BANK_ACCOUNTS, OPENED_ACCOUNT_ELSEWHERE_3M |
| **Pre-computed Scores** | ESTIMATED_CLV, ENGAGEMENT_SCORE, RISK_SCORE, LOYALTY_SCORE |

---

## 4. Project Structure

```
Basilica Group 18 Project/
├── nigerian_bank_churn_complete_analysis_(1) (2).ipynb   # Main notebook (52 cells, 20 steps)
├── nigerian_bank_churn_realistic_10k.xlsx                 # Dataset
└── README.md                                              # This file
```

The entire analysis lives in a single Jupyter notebook divided into 20 clearly labelled steps plus a bonus hyperparameter tuning section.

---

## 5. Dependencies

```
pandas
numpy
scipy
matplotlib
seaborn
scikit-learn
imbalanced-learn   # SMOTE
shap               # Model explainability
openpyxl           # Reading .xlsx files
```

Install all dependencies:

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn imbalanced-learn shap openpyxl
```

> **Note:** The notebook was originally developed in Google Colab. If running locally in Jupyter, remove the `google.colab` file-upload cells and load the `.xlsx` file directly via `pd.read_excel('nigerian_bank_churn_realistic_10k.xlsx')`.

---

## 6. Pipeline Walkthrough

### Step 1 — Library Imports & Configuration
All libraries are imported and plot styles are configured (Seaborn theme, figure sizing).

### Step 2 — Load Dataset
The Excel file is loaded and an initial preview of shape, dtypes, and sample rows is printed.

### Step 3 — Data Quality Assessment
Checks for null values, duplicate rows, and data type consistency. Result: zero nulls, zero duplicates.

### Step 4 — Statistical Summary
Descriptive statistics (mean, std, min/max, quartiles) for all numerical features.

### Step 5 — Target Variable Analysis
- Churn rate: **23.62%**
- Visualised as bar chart and pie chart
- Imbalance ratio noted: 3.23:1

### Step 6 — Demographic Analysis
Churn rates broken down by age, gender, marital status, education level, and account type. Statistical tests (chi-square, t-tests) quantify significance.

### Step 7 — Account Behavior Analysis
- Average tenure: **49 months** (retained) vs **39 months** (churned)
- Churn by number of products:
  - 1 product → **33% churn**
  - 3+ products → **12% churn**
- Credit card penetration: 35.6%
- Loan customers: 31.04%

### Step 8 — Transaction & Balance Analysis
- Average monthly transactions: **24.6** (retained) vs **18.7** (churned)
- 3-month balance change trends
- Impact of minimum balance violations on churn

### Step 9 — Digital Engagement & Customer Service Analysis
- App login frequency and digital transaction ratio vs churn
- Complaint patterns over 6-month window
- Complaint resolution ratio vs churn rate
- Branch visits and CS calls frequency analysis

### Step 10 — Correlation Analysis
Heatmap of top correlated features with target:
- Strong correlations (> 0.2): 3 features
- Moderate correlations (0.1–0.2): 6 features
- Weak correlations (< 0.1): 18 features

### Step 11 — Target Variable Generation Logic
Validates that the `CHURN_FLAG` column is consistent with the underlying feature logic used to simulate the dataset.

### Step 12 — Feature Engineering
14 new features are created (see [Section 7](#7-feature-engineering)).

### Step 13 — Data Preparation
- Label encoding for categorical variables (`_ENCODED` suffix)
- 80/20 stratified train-test split
- StandardScaler fitted on training data only
- SMOTE applied to training set to balance classes

### Steps 14–16 — Model Training
Three models trained and evaluated on the held-out test set (see [Section 8](#8-models--results)).

### Step 17 — Model Comparison & Selection
Side-by-side comparison of Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Logistic Regression selected as the production model based on recall and ROC-AUC.

### Step 18 — Performance Visualizations
ROC curves, Precision-Recall curves, and confusion matrices plotted for all three models.

### Step 19 — SHAP Analysis
SHAP values computed for the selected model to explain which features drive individual predictions.

### Step 20 — Business Insights & Recommendations
Translates model outputs into retention strategies and quantifies ROI.

### Bonus — Hyperparameter Tuning
GridSearchCV applied to all three models. Best CV results shown in [Section 9](#9-hyperparameter-tuning).

---

## 7. Feature Engineering

14 new features created from the original 35:

| Feature | Description |
|---------|-------------|
| `TENURE_PRODUCTS_INTERACTION` | Tenure × number of products (loyalty signal) |
| `BALANCE_TRANSACTIONS_RATIO` | Current balance divided by avg monthly transactions |
| `DIGITAL_ENGAGEMENT_INDEX` | Composite of app logins, internet banking, digital transaction ratio |
| `IS_INACTIVE` | Binary flag for customers with very low transaction activity |
| `IS_DECLINING` | Binary flag where transaction trend is declining |
| `IS_HIGH_RISK` | Binary flag based on risk score threshold |
| `IS_LOW_LOYALTY` | Binary flag based on loyalty score threshold |
| `BALANCE_HEALTH` | Composite of balance change, violations, and current balance |
| `TRANSACTION_ACTIVITY_LEVEL` | Categorised transaction volume (Low / Medium / High) |
| `IS_PREMIUM_CUSTOMER` | Binary flag for premium account holders |
| `IS_AT_RISK_CUSTOMER` | Multi-factor risk flag combining several risk signals |
| `IS_NEW_CUSTOMER` | Tenure < 6 months |
| `IS_LONG_TERM_CUSTOMER` | Tenure > 48 months (4 years) |
| `SERVICE_QUALITY_INDEX` | Complaint volume weighted by resolution ratio |

**Final feature set: 47 features** (35 original + 14 engineered, after label encoding and dropping redundant columns).

---

## 8. Models & Results

### Data Split & Class Balancing

| Split | Original | After SMOTE |
|-------|----------|-------------|
| Training set | 8,000 rows | 10,088 rows (5,044 : 5,044) |
| Test set | 2,000 rows | 2,000 rows (unchanged) |

SMOTE is applied only to the training set; the test set preserves the real-world distribution.

### Baseline Model Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 75.90% | 49.30% | **74.79%** | 0.5943 | **0.8436** |
| Random Forest | 79.35% | 56.97% | 51.06% | 0.5385 | 0.8258 |
| Gradient Boosting | **80.25%** | **60.55%** | 46.82% | 0.5281 | 0.8320 |

### Selected Production Model: Logistic Regression

Logistic Regression was chosen despite lower accuracy for the following reasons:

- **Highest Recall (74.79%)** — catches the most churners, which directly maximises retention campaign effectiveness
- **Highest ROC-AUC (0.8436)** — best overall risk ranking across probability thresholds
- **Interpretability** — coefficients are explainable to business stakeholders
- In churn prediction, a missed churner (false negative) is far more costly than a false alarm (false positive)

---

## 9. Hyperparameter Tuning

GridSearchCV with 5-fold stratified cross-validation applied to all three models.

### Logistic Regression

| Parameter | Search Space | Best Value |
|-----------|-------------|-----------|
| `C` | [0.001, 0.01, 0.1, 1, 10] | 0.01 |
| `penalty` | ['l1', 'l2'] | l1 |
| `solver` | ['liblinear', 'lbfgs'] | liblinear |

**Best CV F1-Score: 0.7675**

### Random Forest

| Parameter | Search Space | Best Value |
|-----------|-------------|-----------|
| `n_estimators` | [100, 200, 300] | 300 |
| `max_depth` | [10, 20, 30, None] | None |
| `min_samples_split` | [2, 5, 10] | 2 |
| `min_samples_leaf` | [1, 2, 4] | 1 |

**Best CV F1-Score: 0.8352** ← Highest overall

### Gradient Boosting

| Parameter | Search Space | Best Value |
|-----------|-------------|-----------|
| `n_estimators` | [100, 200, 300] | 100 |
| `learning_rate` | [0.01, 0.1, 0.2] | 0.01 |
| `max_depth` | [3, 5, 7] | 7 |
| `subsample` | [0.8, 0.9, 1.0] | 0.9 |

**Best CV F1-Score: 0.7719**

---

## 10. Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) values are computed for the Logistic Regression model to explain:
- **Global feature importance** — which features matter most across all predictions
- **Direction of impact** — whether a feature pushes the prediction toward churn or retention
- **Individual prediction explanations** — why a specific customer was flagged as high-risk

This is critical for the bank's compliance and stakeholder communication requirements.

---

## 11. Key Business Insights

### Top Churn Risk Signals

| Signal | Churn Rate |
|--------|-----------|
| Opened account at a competitor (last 3 months) | **52%** |
| Loyalty score < 30 | **53%** |
| Declining transaction trend | **40%** |
| Multiple unresolved complaints | **35%+** |
| Single-product holder | **33%** |
| New customer (tenure < 6 months) | Elevated |

### Protective Factors (Reduce Churn)

| Factor | Effect |
|--------|--------|
| Tenure > 4 years | ~15% lower churn |
| 3+ products | ~12% churn (vs 33% for 1 product) |
| High digital engagement | ~20% churn reduction |
| Premium account | Significantly more loyal |

### Demographic Patterns

- Younger customers and those with lower education levels showed higher churn tendencies
- Single and unmarried customers churn at higher rates than married customers
- Customers with loan defaults have a markedly higher churn probability

---

## 12. ROI Analysis

The model identifies 1,766 high-risk customers (74.79% recall applied to 2,362 churners).

| Metric | Value |
|--------|-------|
| Customers contacted | 1,766 |
| Retention cost per customer | ₦5,000 |
| Total campaign cost | ₦8,830,000 |
| Intervention success rate (assumed) | 30% |
| Customers retained | ~441 |
| Average Customer Lifetime Value | ₦13,156,827 |
| Revenue saved | ₦5,806,595,098 |
| Net benefit | ₦5,797,765,098 |
| **ROI** | **65,660%** |
| **Return per ₦1 spent** | **₦657.60** |

Even with conservative assumptions, the financial case for deploying this model is overwhelming.

---

## 13. Recommendations

### Immediate (0–3 months)
1. **Deploy the Logistic Regression model** for real-time churn risk scoring on all active accounts
2. **Set up a monitoring dashboard** tracking daily predictions, recall, and precision
3. **Prioritise single-product customers** — offer them a second product with an incentive

### Short-Term (3–6 months)
4. **New customer onboarding programme** — intensive engagement in the first 6 months to reduce early churn
5. **Digital adoption campaign** — promote mobile app and internet banking to non-digital users
6. **Complaint escalation workflow** — ensure all complaints are resolved within 48 hours (unresolved complaints are a leading churn indicator)

### Long-Term (6–12 months)
7. **Loyalty programme** — reward long-tenure and multi-product customers
8. **Competitor monitoring** — trigger immediate outreach when a customer opens an account elsewhere
9. **Model retraining pipeline** — retrain monthly with new data and track performance drift
10. **Ensemble models** — explore stacking Logistic Regression with the tuned Random Forest to push ROC-AUC above the 0.85 target

---

## 14. How to Run

### Option A — Google Colab (Recommended)

1. Upload the notebook and the `.xlsx` file to Google Drive or Colab
2. Open the notebook in Colab
3. Run all cells in order (`Runtime → Run all`)
4. The Colab file-upload cell at the top will prompt you to upload the dataset

### Option B — Local Jupyter

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy scipy matplotlib seaborn scikit-learn imbalanced-learn shap openpyxl jupyter
   ```
3. Comment out the `google.colab` upload cell and replace with:
   ```python
   import pandas as pd
   df = pd.read_excel('nigerian_bank_churn_realistic_10k.xlsx')
   ```
4. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook "nigerian_bank_churn_complete_analysis_(1) (2).ipynb"
   ```
5. Run all cells in order

---

## Authors

**Group 18** — Master's Programme

---

*This project was developed as a master's capstone demonstrating a production-ready machine learning pipeline, from raw data exploration through to business ROI quantification.*
