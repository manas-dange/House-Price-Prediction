# 🏡 ResiVal: Intelligent Residential Property Valuation & Price Forecasting Using Multiple & Polynomial Regression Analysis

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)
![Course](https://img.shields.io/badge/Course-ML%20Fundamentals%20(Sem%205)-purple)
![Group](https://img.shields.io/badge/Group-Group%203-success)

---

## 📌 1. Project Overview & Attribution
- **Course**: Machine Learning Fundamentals (Semester 5)
- **Project Group**: **Group 3**
- **Approved Project Title**: **ResiVal: Intelligent Residential Property Valuation & Price Forecasting Using Multiple & Polynomial Regression Analysis**
- **Problem Statement**: Build an econometric Machine Learning model that estimates the selling price of residential properties based on square footage, bedrooms, bathrooms, neighborhood location, property age, parking capacity, stories, main road access, and furnishing conditions.
- **Primary Deliverable**: Master self-contained executed notebook (`Group3_House_Price_Prediction.ipynb`) and theme-adaptive interactive Streamlit web application (`app.py`).

---

## 🔄 2. Expected Machine Learning Lifecycle Flow
```
Problem Definition & Objectives
            │
            ▼
    Dataset Acquisition (housing_data.csv)
            │
            ▼
 Data Cleaning & Outlier Analysis (IQR & Winsorization)
            │
            ▼
 Exploratory Data Analysis (EDA) & Correlations
            │
            ▼
 Feature Engineering (bath_bed_ratio, is_new_construction, amenity_score)
            │
            ▼
 Leak-Free Pipeline Design & Train/Test Split (80/20)
            │
            ▼
 Regression Modeling (Simple, Multiple, Polynomial, Ridge)
            │
            ▼
 Performance Evaluation (MAE, MSE, RMSE, R², Adj. R²)
            │
            ▼
 Model Selection & Serialization (best_house_price_model.pkl)
            │
            ▼
 Streamlit Web Application Deployment (Theme-Adaptive app.py)
            │
            ▼
 Testing, Validation & Viva Defense
```

---

## 📊 3. Final Model Evaluation & Comparison Matrix

All candidate models were evaluated on an unseen holdout testing set ($n = 500$, 20% split):

| Candidate Model | Train $R^2$ | Test $R^2$ | Test Adj. $R^2$ | Test MAE ($) | Test RMSE ($) | Model Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Simple Linear Regression** | 0.7011 | 0.7145 | 0.7139 | $99,172.74 | $123,013.24 | Baseline (Area only; high bias) |
| **2. Multiple Linear Regression** | 0.9513 | 0.9591 | 0.9579 | $34,119.75 | $46,571.37 | Strong structural linear fit |
| **3. Polynomial Regression (Deg 2)** | 0.9682 | **0.9729** | **0.9644** | **$27,114.84** | **$37,889.93** | High variance across 104 terms |
| **4. Regularized Poly Ridge (L2)** | 0.9671 | **0.9710** | **0.9619** | **$28,194.33** | **$39,189.77** | 🏆 **Selected Winner** (Optimal bias-variance trade-off) |

### 🏆 Winning Model Justification:
**Model 4 (Regularized Polynomial Ridge)** was selected as the production model because:
1. It explains **97.10%** of unseen price variation with a low test RMSE of **$39,189.77**.
2. The $L_2$ shrinkage parameter ($\alpha = 12.0$) stabilizes matrix conditioning $(X^T X + \alpha I)^{-1}$, suppressing multicollinearity across the 104 polynomial features.
3. Residuals conform to a Gaussian bell curve centered at zero, satisfying the Gauss-Markov homoscedasticity assumptions.

---

## 💻 4. Installation & Local Execution

### Prerequisites
- Python 3.9 or higher (Anaconda recommended)
- Git

### Step 1: Clone or Navigate to Directory
```bash
cd /Users/manasdange/Documents/Subjects/sem5/ML_Fundamentals/HousePricePredictionGroup3
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Master Jupyter Notebook
```bash
jupyter notebook Group3_House_Price_Prediction.ipynb
```
*Note: The notebook is already fully pre-executed with all figures, equations, and tables intact.*

### Step 4: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. The interface automatically adapts between **Dark Mode** (crisp white text on dark cards) and **Light Mode** (black text on light cards).

---

## 📁 5. Repository File Structure

```
House-Price-Prediction/
│
├── Group3_House_Price_Prediction.ipynb  # Master end-to-end executed Jupyter Notebook
├── app.py                             # Theme-adaptive interactive Streamlit web app
├── best_house_price_model.pkl         # Serialized winning pipeline (Joblib)
├── metadata.json                      # Model metadata, performance metrics & feature ranges
├── housing_data.csv                   # Synthetic real-estate econometric dataset (2,500 rows)
├── build_notebook.py                  # Automated notebook generator and executor script
├── requirements.txt                   # Environment dependencies
├── README.md                          # Repository documentation & guide
└── PROJECT_REPORT.md                  # Comprehensive formal academic project report
```

---

## 🎯 6. Individual Viva Defense Preparation Guide

| Key Viva Question | Technical / Theoretical Answer |
| :--- | :--- |
| **Q1. Why does Polynomial Regression risk overfitting, and how does Ridge solve this?** | Polynomial degree 2 generates quadratic terms ($x_i^2$) and interaction terms ($x_i x_j$), increasing dimension from 13 to 104. This leads to near-singular $(X^T X)$ matrices and coefficient inflation. Ridge adds an $L_2$ penalty $\alpha \sum \beta_j^2$, shrinking coefficients toward zero without eliminating them, stabilizing condition numbers, and reducing test variance. |
| **Q2. Why is Winsorization preferred over dropping outliers in real estate?** | Real estate naturally features luxury estates and distressed properties. Dropping rows artificially truncates market variance and reduces sample size. Winsorization caps extreme values at $1.5 \times \text{IQR}$, curbing extreme leverage while preserving 100% of observations. |
| **Q3. Why use both RMSE and MAE?** | MAE gives the uniform average absolute error (robust to outliers). RMSE squares the errors before averaging, penalizing large financial forecasting errors heavily. In real estate pricing, large mistakes carry severe financial risk, making RMSE critical. |
| **Q4. How was data leakage prevented during scaling and encoding?** | All transformers (`StandardScaler`, `OneHotEncoder`) were fitted strictly on the 80% training set ($X_{train}$) inside a `ColumnTransformer` pipeline. Test samples ($X_{test}$) were strictly evaluated using the learned parameters. |

---

## 👥 7. Individual Contribution Record

| Group Member | Module Responsibility | Specific Contributions |
| :--- | :--- | :--- |
| **Member 1** | Problem Definition & Data Sourcing | Formulated real estate problem statement, designed mathematical data generation schema, verified feature bounds. |
| **Member 2** | Preprocessing, EDA & Feature Engineering | Conducted IQR & Z-score outlier audits, implemented Winsorization, constructed EDA visualizations, engineered layout and amenity features. |
| **Member 3** | Regression Modeling & Metrics | Implemented Simple Linear, Multiple Linear, Polynomial, and Ridge models; computed MAE, MSE, RMSE, and $R^2$; performed bias-variance analysis. |
| **Member 4** | Model Selection, Streamlit & Documentation | Carried out residual diagnostics, serialized production pipeline, developed theme-adaptive Streamlit UI, authored technical report. |
