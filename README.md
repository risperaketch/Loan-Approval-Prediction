# Loan Approval Prediction — End-to-End Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

---

## 📌 Project Overview

Built a full supervised machine learning pipeline on **4,269 real loan applications** to automate creditworthiness assessment for financial institutions. The project covers two complementary objectives:

| Part | Task | Method | Best Result |
|---|---|---|---|
| **1** | Predict approved loan amount | OLS Linear Regression | **R² = 0.881 on test set** |
| **2** | Predict loan approval decision | 4 Classification Models | **96% accuracy — Random Forest** |

**`cibil_score`** emerged as the dominant predictor across all four classification models (Decision Tree: 93.9% importance; Random Forest: 83.9%; Logistic Regression odds ratio: 59.0×), confirming that credit history is the single most decisive factor in loan decisions — a finding directly actionable for credit risk teams.

---

## 💼 Business Problem

Financial institutions process hundreds of loan applications daily. Manual review is inconsistent, slow, and exposes lenders to two costly errors:

| Error Type | Description | Business Cost |
|---|---|---|
| **Type I (False Rejection)** | Deny a creditworthy applicant | Lost revenue, reputational damage, regulatory exposure |
| **Type II (False Approval)** | Approve a high-risk applicant | Loan default, principal loss, increased credit portfolio risk |

This project builds data-driven models that flag each application as **Approved** or **Rejected**, and predict the appropriate **loan amount** — enabling faster, more consistent, and more defensible credit decisions.

---

## 🎯 Objectives

- Predict `loan_amount` (regression) and `loan_status` (classification) from applicant financial profiles
- Identify which features drive credit decisions using feature importance and odds ratios
- Compare four classification algorithms and select the model best suited for production deployment
- Handle class imbalance (62% Approved vs. 38% Rejected) using SMOTE and class weighting
- Achieve Accuracy ≥ 85% and Precision/Recall ≥ 75% for both classes

---

## 📊 Dataset

**Source:** `LoanApproval.csv` (4,269 applications · 12 variables · 0 missing values)

| Variable | Type | Description | Role |
|---|---|---|---|
| `cibil_score` | Numeric | Credit score (300–900) | **#1 predictor** across all models |
| `income_annum` | Numeric | Annual income ($) | **#1 driver of loan amount** |
| `loan_amount` | Numeric | Approved loan amount ($) | Regression target |
| `loan_status` | Categorical | Approved / Rejected | Classification target |
| `loan_term` | Numeric | Loan duration (months) | #2 classification predictor |
| `residential_assets_value` | Numeric | Residential asset value ($) | Asset collateral signal |
| `commercial_assets_value` | Numeric | Commercial asset value ($) | Asset collateral signal |
| `luxury_assets_value` | Numeric | Luxury asset value ($) | Secondary predictor |
| `bank_asset_value` | Numeric | Bank-held asset value ($) | Secondary predictor |
| `no_of_dependents` | Numeric | Number of financial dependents | Household burden |
| `education` | Nominal | Graduate / Not Graduate | Demographic indicator |
| `self_employed` | Nominal | Yes / No | Employment type |

**Target class distribution:**

| Class | Count | Proportion |
|---|---|---|
| Approved (1) | 2,656 | 62.2% |
| Rejected (0) | 1,613 | 37.8% |

---

## 🛠️ Methodology

```
Data Loading (4,269 rows × 12 columns)
       ↓
Preprocessing
  • Strip whitespace → encode target → identify variable types
  • Numeric: SimpleImputer(median) → StandardScaler
  • Nominal: OneHotEncoder(drop='first')
       ↓
EDA — 12 Visualizations
  • Loan amount distributions (original, log, sqrt)
  • Correlation heatmap · Box plots · Class distribution
  • K-Means elbow plot · Resampling comparison
       ↓
PART 1 — OLS Linear Regression
  • Compare original vs. sqrt-transformed target
  • Diagnostic plots: Fitted/Residuals · Q-Q · Scale-Location
  • 70/30 train-test split (random_state=123)
       ↓
PART 2 — Classification (4 Models)
  • Stratified 70/30 split (stratify=y)
  • GridSearchCV 5-fold cross-validation
  • Resampling: RUS · ROS · SMOTE
       ↓
Model Selection → Business Recommendation
```

---

## 📈 Results

### Part 1 — Linear Regression

| Metric | Original Model | Sqrt-Transformed Model |
|---|---|---|
| Training Adj. R² | 0.858 | **0.881** |
| Test R² | lower | **0.881** |
| Test MAE | higher | **368.81** (sqrt scale) |
| Test RMSE | higher | **441.87** (sqrt scale) |
| Residual Pattern | Strong fan (heteroscedasticity) | Substantially improved |

The **sqrt-transformed model** was selected. Back-transforming (squaring) predictions recovers dollar amounts, with typical prediction errors of ±$250K–$400K around the mean — acceptable given loan values up to $35M+.

**Top regression predictors:**

| Feature | Coefficient | Direction |
|---|---|---|
| `income_annum` | +1,197.62 | ↑ Higher income → larger loan |
| `commercial_assets_value` | +22.73 | ↑ Commercial assets support larger loans |
| `self_employed_Yes` | −19.82 | ↓ Self-employment reduces predicted amount |
| `no_of_dependents` | −12.26 | ↓ More dependents → smaller loan |
| `luxury_assets_value` | −17.74 | ↓ Negative on sqrt scale |

---

### Part 2 — Classification

**GridSearchCV Optimal Hyperparameters (5-fold CV):**

| Model | Best Configuration | CV Accuracy |
|---|---|---|
| Decision Tree | max_depth=4, min_samples_split=2 | 0.9528 |
| **Random Forest** | **n_estimators=50** | **0.9622** |
| KNN | k=46 | 0.9207 |
| Logistic Regression | liblinear, class_weight=balanced | — |

**Full Test Set Performance (n = 1,281):**

| Model | Accuracy | Precision (Reject) | Recall (Reject) | F1 (Reject) | Precision (Approve) | Recall (Approve) | F1 (Approve) |
|---|---|---|---|---|---|---|---|
| **Decision Tree** | 0.9578 | 0.9283 | **0.9628** | 0.9452 | 0.9769 | 0.9548 | 0.9657 |
| **Random Forest** | **0.9571** | **0.9535** | 0.9318 | 0.9425 | **0.9592** | **0.9724** | **0.9657** |
| KNN | 0.9262 | lower | lower | lower | lower | lower | lower |
| Logistic Regression | 0.81 | 0.75 | ~0.74 | ~0.74 | ~0.84 | ~0.85 | ~0.85 |

---

## ✅ Business Recommendation

**Deploy: Random Forest (n_estimators=50, class_weight='balanced')**

The Random Forest is recommended for the primary production model because:
- **96% overall accuracy** — highest balance across all metrics
- **95.4% precision on Rejected loans** — when it flags a rejection, it is almost always correct (very few false alarms on creditworthy applicants)
- **97.2% recall on Approved loans** — very few eligible applicants are incorrectly rejected
- **No overfitting signal** — unlike the Decision Tree's 96.3% Rejected recall, which risks memorising training data

**Logistic Regression as secondary model** for regulatory contexts: When an applicant receives an adverse action notice (required under ECOA/Regulation B), the Logistic Regression's odds ratios provide a plain-language explanation — `cibil_score` carries a **59× odds multiplier** for approval — that satisfies regulatory explainability requirements that Random Forest cannot natively provide.

---

## 🔑 Key Findings

> **`cibil_score` is the overwhelming determinant of loan decisions.**
> - Decision Tree feature importance: **93.9%**
> - Random Forest feature importance: **83.9%**
> - Logistic Regression odds ratio: **59.0×** (a one-unit increase in cibil_score multiplies approval odds by 59×)

This finding is directly actionable: credit teams should ensure CIBIL score data pipelines are accurate, current, and regularly validated. Model performance degrades significantly if this input is stale or incorrect.

**`loan_term` is the secondary predictor** (DT: 5.5%, RF: 5.3%) — longer loan terms signal higher repayment burden and are associated with lower approval likelihood at equivalent income levels.

---

## ⚠️ Challenges & Limitations

| Challenge | Impact | Mitigation Applied |
|---|---|---|
| Class imbalance (62/38%) | Models bias toward majority class | SMOTE oversampling + `class_weight='balanced'` |
| `loan_amount` right-skewed | OLS heteroscedasticity | Sqrt transformation — reduced fan pattern |
| No economic context variables | Misses macro risk factors | Future work — incorporate interest rate, GDP data |
| Single lending context | May not generalise | Retrain on institution-specific portfolios |

The residual diagnostics reveal that mild heteroscedasticity persists even after sqrt transformation — a more flexible model (Gradient Boosting, Log-linear regression) could further improve regression performance.

---

## 🚀 Future Improvements

- [ ] **Gradient Boosting / XGBoost** — expected to outperform Random Forest on tabular data with complex feature interactions
- [ ] **Threshold calibration** — lower the decision boundary below 0.50 to trade precision for recall based on institution-specific cost preferences
- [ ] **Macroeconomic features** — prime rate, GDP growth, regional unemployment as additional predictors
- [ ] **Real-time scoring API** — package model as a REST endpoint using FastAPI or Flask for integration with core banking systems
- [ ] **SHAP values** — generate granular per-applicant explanations for regulatory adverse action notices
- [ ] **Quarterly retraining pipeline** — automate model refresh as the loan book and economic conditions evolve

---

## 📊 Visualizations

| Figure | File | Description |
|---|---|---|
| 1 | `loan_amount_distributions.png` | Original · Log · Sqrt distributions with median lines |
| 2 | `correlation_heatmap.png` | Lower-triangle Pearson correlation of 8 numerical features |
| 3 | `loan_amount_by_category.png` | Box plots: loan amount vs. education and self-employment |
| 4 | `kmeans_elbow.png` | K-Means WSS elbow (k=2–25) — no clear elbow, segmentation not informative |
| 5 | `target_distribution.png` | Bar + pie: 62.2% Approved vs. 37.8% Rejected |
| 6 | `diagnostics_original.png` | OLS diagnostics: Fitted/Residuals · Q-Q · Scale-Location (original) |
| 7 | `diagnostics_sqrt.png` | OLS diagnostics: Fitted/Residuals · Q-Q · Scale-Location (sqrt model) |
| 8 | `actual_vs_predicted.png` | Scatter: actual vs. predicted loan amount (R² = 0.881, back-transformed $) |
| 9 | `resampling_distributions.png` | Class distribution: Original · RUS · ROS · SMOTE |
| 10 | `feature_importance.png` | Feature importance: DT · RF · Logistic Regression side-by-side |
| 11 | `confusion_matrices.png` | 2×2 confusion matrices for all 4 models |
| 12 | `model_accuracy_comparison.png` | Bar chart: accuracy across all 4 models vs. 85% target |

<img width="1632" height="557" alt="image" src="https://github.com/user-attachments/assets/840af0e4-469a-4f00-a953-638002a7d314" />
<img width="929" height="752" alt="image" src="https://github.com/user-attachments/assets/d63a037c-e67a-456f-8fd9-9fc33ad177e3" />
<img width="1302" height="668" alt="image" src="https://github.com/user-attachments/assets/257523ee-cf52-41b4-b67d-7eb256bd6596" />
<img width="1082" height="532" alt="image" src="https://github.com/user-attachments/assets/b4407298-92bd-481d-aa21-2783a1333ee4" />
<img width="1259" height="557" alt="image" src="https://github.com/user-attachments/assets/b79a990c-2d4f-49d7-b19f-7fb3a7150500" />
<img width="1522" height="557" alt="image" src="https://github.com/user-attachments/assets/c6a0d334-a6e5-47cd-8806-837ed12d7f97" />
<img width="1522" height="557" alt="image" src="https://github.com/user-attachments/assets/a8db7d2b-c298-499e-9772-e0b3d6e5a357" />
<img width="862" height="642" alt="image" src="https://github.com/user-attachments/assets/f5561a65-64c4-4dfc-ab65-4052df426551" />
<img width="1082" height="891" alt="image" src="https://github.com/user-attachments/assets/86769c93-ce0d-40d6-8824-6b57a86ab5a4" />
<img width="1742" height="557" alt="image" src="https://github.com/user-attachments/assets/d705a1f6-5632-4070-96c5-0e7f12108c83" />
<img width="1200" height="1113" alt="image" src="https://github.com/user-attachments/assets/b2dfb07c-04e1-4bc5-8de3-bf73fea36f03" />
<img width="1082" height="532" alt="image" src="https://github.com/user-attachments/assets/473a58a1-54dd-4edb-9405-03250c87f6ab" />

---

## 🧰 Tools & Technologies

| Category | Tools |
|---|---|
| **Language** | Python 3.10 |
| **Data Manipulation** | pandas, NumPy |
| **Machine Learning** | scikit-learn (Pipeline, GridSearchCV, ColumnTransformer) |
| **Imbalanced Learning** | imbalanced-learn (SMOTE, ROS, RUS) |
| **Statistics** | statsmodels (OLS, diagnostic plots) |
| **Visualization** | matplotlib, seaborn |
| **Environment** | Jupyter Notebook / Google Colab |

---

## 💡 Skills Demonstrated

- **End-to-end ML pipeline** from raw data to deployable model recommendation
- **OLS regression diagnostics** — residual analysis, heteroscedasticity detection, target transformation
- **Multi-model classification** — Decision Tree, Random Forest, KNN, Logistic Regression
- **Hyperparameter tuning** — GridSearchCV with 5-fold cross-validation (180+ model fits for DT alone)
- **Class imbalance handling** — three resampling strategies evaluated and compared
- **Feature importance analysis** — Gini reduction, odds ratios, coefficient interpretation
- **Business communication** — translating model output into credit policy recommendations
- **Regulatory awareness** — adverse action notices, ECOA, model explainability requirements

---

## 📁 Project Files

| File | Description |
|---|---|
| `Loan_Approval_Prediction.ipynb` | Complete analysis — EDA, regression, 4 classifiers, evaluation |
| `LoanApproval.csv` | Input dataset (4,269 applications) |
| `README.md` | Project documentation |
| `*.png` | 12 generated visualization files |

---

## 👤 Author

**Aketch Adhiambo Okoth**  
M.S. Business Analytics — Montclair State University (GPA 3.8)  
📧 aaketch1997@gmail.com | 📞 +1 (862) 405-9654  
🔗 [LinkedIn]([https://linkedin.com/in/aketchrisperokoth]) · [GitHub](https://github.com/risperaketch) · [Portfolio](https://your-portfolio-url.com)

*Seeking Data Analyst / Business Analyst roles — Spring 2026 · New Jersey / Remote*

---

*This project is part of a portfolio of 12 data analytics and machine learning projects covering clustering, NLP, sentiment analysis, HR analytics, and healthcare AI. See full portfolio at the link above.*
