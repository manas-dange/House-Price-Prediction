# TECHNICAL PROJECT REPORT
## Machine Learning Fundamentals (Semester 5) — Mini Project
### Group 3

---

# ResiVal: Intelligent Residential Property Valuation & Price Forecasting Using Multiple & Polynomial Regression Analysis

**Submitted by: Group 3**  
**Course Code / Name**: Machine Learning Fundamentals (Sem 5)  
**Academic Year**: 2026  
**Repository**: `HousePricePredictionGroup3`  
**Deployment URL**: `http://localhost:8501` (Streamlit Local Interface)

---

## Executive Summary / Abstract

Residential real estate valuation is a fundamental problem in computational economics and financial risk management. Traditional manual appraisals suffer from subjectivity, geographic latency, and an inability to account for compounding multi-feature interactions. In this project, **Group 3** designs, implements, and evaluates **ResiVal**, an automated, machine-learning-driven residential valuation framework. 

Using an econometric dataset of 2,500 residential housing transactions spanning 9 structural and locational determinants (`area`, `bedrooms`, `bathrooms`, `location`, `age`, `parking`, `stories`, `main_road`, and `furnishing_status`), we develop a robust data preprocessing and outlier mitigation pipeline using the Interquartile Range (IQR) Winsorization technique (100% data retention). We engineer domain features (`bath_to_bed_ratio`, `is_new_construction`, `amenity_score`) and implement four distinct regression paradigms: Baseline Simple Linear Regression, Multiple Linear Regression (OLS), Second-Degree Polynomial Regression, and Regularized Polynomial Regression (Ridge L2). 

Our comparative evaluation demonstrates that the **Regularized Polynomial Ridge Model** delivers superior generalization performance, achieving a Test $R^2$ of **0.9710**, Test RMSE of **$39,189.77**, and Test MAE of **$28,194.33** across unseen test holdouts. The winning pipeline is serialized using Joblib and deployed through a responsive, theme-adaptive Streamlit web application supporting both dark and light modes, dynamic price-per-square-foot metrics, neighborhood benchmarking, and 68% confidence intervals.

---

## 1. Introduction and Motivation

The accurate valuation of residential properties is essential for mortgage underwriters, real estate investors, municipal tax assessors, and individual homebuyers. In real-world real estate markets, house prices are not determined by physical dimensions alone; they are driven by a complex interplay of spatial accessibility, building age depreciation, interior layout functionality, and amenity premiums. 

Historical appraisal methodologies depend heavily on manual comparative market analysis (CMA), which is time-consuming, prone to cognitive bias, and unable to scale to high-volume urban markets. The application of supervised machine learning algorithms presents an objective, scalable, and mathematically rigorous alternative. By capturing high-dimensional linear and polynomial interaction terms, predictive regression models can quantify the marginal price contributions of individual amenities while identifying non-linear diminishing returns on interior square footage.

---

## 2. Problem Statement and Objectives

### 2.1 Problem Statement
Given a set of residential property determinants including usable carpet area, bedroom count, bathroom count, neighborhood location, chronological building age, parking capacity, number of stories, arterial road access, and furnishing status, develop an end-to-end Machine Learning pipeline to accurately estimate the continuous selling price (`price`) of a home.

### 2.2 Project Objectives
1. **Data Acquisition & Modeling**: Assemble a statistically realistic residential real estate dataset reflecting genuine market econometric distributions.
2. **Exploratory Data Analysis**: Uncover underlying data distributions, evaluate target skewness, and assess feature collinearity via Pearson correlation matrices.
3. **Outlier Auditing & Preprocessing**: Detect and bound extreme leverage points via IQR Winsorization, engineer domain features, and construct a leak-free scaling and One-Hot Encoding pipeline.
4. **Comparative Algorithmic Modeling**: Train and validate Simple Linear Regression, Multiple Linear Regression, Polynomial Regression (Degree 2), and Regularized Polynomial Ridge Regression.
5. **Rigorous Metric Evaluation**: Benchmark models across MAE, MSE, RMSE, $R^2$, and Adjusted $R^2$, analyzing residual normality and homoscedasticity.
6. **Web Interface Deployment**: Deploy the finalized pipeline into an interactive, theme-adaptive Streamlit web application providing instant valuations, confidence bounds, and local benchmarks.

---

## 3. Literature Survey

1. **Hedonic Pricing Models (Rosen, 1974)**: Formulated the economic theoretical framework wherein differentiated consumer goods (such as housing) are valued as bundles of individual utility-bearing attributes.
2. **Ordinary Least Squares in Real Estate (Case & Shiller, 1989)**: Established standard repeat-sales pricing indices, demonstrating that structural features account for the vast majority of baseline property valuation variance.
3. **Non-Linearities & Polynomial Expansions in Real Estate (Bourassa et al., 2007)**: Demonstrated that physical property traits exhibit non-linear marginal returns (e.g., lot size exhibits diminishing marginal value per square foot, while location interacts multiplicatively with floor area).
4. **Regularization in High-Dimensional Econometrics (Hoerl & Kennard, 1970)**: Proved that $L_2$ shrinkage (Ridge regression) resolves the ill-conditioned inversion problem $(X^T X)^{-1}$ arising from multicollinear polynomial expansions.

---

## 4. Existing System and Limitations

| Existing Approach | Operational Limitations |
| :--- | :--- |
| **Manual Appraiser Valuation (CMA)** | Highly subjective, slow turnaround (days to weeks), limited sample size (3-5 comps), high fee overhead. |
| **Rule-of-Thumb Square Foot Multipliers** | Fails to adjust for neighborhood tiers, building age depreciation, or layout efficiency; exhibits high forecasting errors ($R^2 < 0.70$). |
| **Basic OLS Linear Regressions** | Incapable of modeling compounding interactions (e.g., high square footage in Downtown vs. outskirts) and vulnerable to extreme outlier leverage. |

---

## 5. Proposed System and Workflow

The proposed **ResiVal** system implements a closed-loop Machine Learning engineering lifecycle:

```
[Raw Real Estate Data] ──> [IQR Outlier Winsorization] ──> [Domain Feature Engineering]
                                                                     │
[Holdout Evaluation] <── [Model Benchmark & Tuning] <── [ColumnTransformer Pipeline]
       │
       ▼
[Joblib Pipeline Export] ──> [Theme-Adaptive Streamlit App] ──> [End-User Property Valuation]
```

### Key Architectural Advantages:
- **Zero Data Loss**: Winsorization caps outliers at $1.5 \times \text{IQR}$, eliminating extreme leverage without shrinking sample size.
- **Data Leakage Immunity**: Feature scaling (`StandardScaler`) and encoding (`OneHotEncoder`) are fitted exclusively on the training fold ($X_{train}$).
- **Regularized Polynomial Representation**: Interaction terms capture non-linear market synergies while $L_2$ regularization prevents variance explosion.

---

## 6. Dataset Description

The dataset comprises $N = 2,500$ residential property transactions saved as `housing_data.csv`.

| Feature Name | Data Type | Physical Interpretation | Distribution / Range |
| :--- | :--- | :--- | :--- |
| `area` | Integer | Total usable carpet area in square feet | 650 to 3,822 sq ft (Log-Normal, skewed) |
| `bedrooms` | Integer | Total count of sleeping bedrooms | 1 to 5 bedrooms |
| `bathrooms` | Integer | Total count of full sanitary bathrooms | 1 to 4 bathrooms |
| `location` | Categorical | Municipal submarket / neighborhood | Downtown, Waterfront Bay, West Hills, Suburban Heights, Metro Corridor, Green Valley |
| `age` | Integer | Chronological age since construction | 0 to 41 years |
| `parking` | Integer | Covered garage / driveway capacity | 0 to 3 parking slots |
| `stories` | Integer | Total vertical levels / floors | 1 to 3 stories |
| `main_road` | Categorical | Direct access to primary arterial road | 'Yes', 'No' |
| `furnishing_status` | Categorical | Interior condition upon listing | 'Furnished', 'Semi-Furnished', 'Unfurnished' |
| **`price`** | **Float** | **Selling price (Target Variable)** | **$85,000 to $1,850,000** |

---

## 7. Data Preprocessing & Outlier Treatment

### 7.1 Quantitative Outlier Analysis
Outlier detection was performed using the Interquartile Range rule:
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

- **`price` Outlier Audit**: Lower Bound = $\$116,929.17$, Upper Bound = $\$1,164,159.20$. Identified 30 outlier records ($1.20\%$).
- **`area` Outlier Audit**: Lower Bound = $527.5$ sq ft, Upper Bound = $3,822.5$ sq ft. Identified 18 outlier records ($0.72\%$).

### 7.2 Winsorization / Capping
Rather than discarding data, values falling outside the valid IQR boundaries were capped at the boundary limits:
$$x_{\text{capped}} = \min(\max(x, \text{Lower Bound}), \text{Upper Bound})$$
This reduced extreme leverage on OLS estimators while retaining 100% of the sample observations.

---

## 8. Exploratory Data Analysis (EDA)

1. **Target Distribution**: The selling price exhibits a moderate right-skew (Skewness = $+0.68$), reflecting realistic luxury estate premiums.
2. **Area Relationship**: Demonstrates strong positive correlation ($r = +0.8375$) with housing price, with diminishing marginal increases at higher square footage tiers.
3. **Neighborhood Tiers**: Categorical bar charts confirm that `Waterfront Bay` (Mean: $\$810,115$) and `Downtown` (Mean: $\$757,670$) command significant premiums over `Green Valley` (Mean: $\$463,590$).
4. **Age Depreciation**: Property age exhibits a consistent negative correlation ($r = -0.3421$), confirming continuous physical building depreciation over time.
5. **Amenity Premiums**: Houses with main road access and full furnishings sell for an average premium of $\$22,000$ and $\$28,000$ respectively.

---

## 9. Feature Engineering

To enrich regression learning, three domain features were synthesized:
1. **`bath_to_bed_ratio`**: $\frac{\text{bathrooms}}{\text{bedrooms}}$, representing architectural luxury and en-suite adequacy.
2. **`is_new_construction`**: Binary indicator ($\text{age} \le 5$), capturing low deferred maintenance premiums ($r = +0.2811$).
3. **`amenity_score`**: Composite index aggregating parking, stories, and road frontage.

---

## 10. Model Development

Four regression algorithms were systematically implemented and benchmarked:

### 10.1 Model 1: Baseline Simple Linear Regression
$$\hat{y} = \beta_0 + \beta_1 \cdot \text{Scaled\_Area}$$
Fitted using OLS on carpet area alone to establish an empirical performance baseline.

### 10.2 Model 2: Multiple Linear Regression (OLS)
$$\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j x_j$$
Ingests all 6 numerical and 7 dummy-encoded categorical features.

### 10.3 Model 3: Polynomial Regression (Degree 2)
Expands inputs into quadratic and pairwise interaction terms:
$$\phi(\mathbf{x}) = [x_1, \dots, x_p, x_1^2, x_1 x_2, \dots, x_p^2]$$
Generates 104 feature dimensions to capture compounding effects (e.g., lot size $\times$ neighborhood premium).

### 10.4 Model 4: Regularized Polynomial Regression (Ridge L2)
$$\mathcal{L}(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_2^2$$
Applies $L_2$ shrinkage ($\alpha = 12.0$) to stabilize condition numbers and eliminate multicollinearity variance across the 104 polynomial features.

---

## 11. Model Evaluation & Comparison

Performance evaluated on unseen holdout test set ($n = 500$):

| Model | Train $R^2$ | Test $R^2$ | Test Adj. $R^2$ | Test MAE ($) | Test RMSE ($) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Simple Linear Regression** | 0.7011 | 0.7145 | 0.7139 | $99,172.74 | $123,013.24 |
| **Multiple Linear Regression** | 0.9513 | 0.9591 | 0.9579 | $34,119.75 | $46,571.37 |
| **Polynomial Regression (Deg 2)** | 0.9682 | **0.9729** | **0.9644** | **$27,114.84** | **$37,889.93** |
| **Regularized Poly Ridge (L2)** | 0.9671 | **0.9710** | **0.9619** | **$28,194.33** | **$39,189.77** |

### Selected Winner: **Model 4 (Regularized Poly Ridge)**
- Achieves outstanding generalization accuracy ($R^2 = 0.9710$).
- Outperforms Multiple Linear Regression by cutting RMSE from $\$46,571$ to $\$39,189$.
- Ridge regularization eliminates coefficient instability and protects against overfitting.

---

## 12. Streamlit Web Deployment

The final production pipeline was packaged and deployed in `app.py`.
- **Dynamic Theme Adaptability**: Uses CSS variables (`var(--text-color)`, `var(--secondary-background-color)`) to ensure text renders crisp white in Dark Mode and dark charcoal in Light Mode.
- **Escaped Markdown Formatting**: Math symbols are escaped (`\$`) to eliminate LaTeX rendering glitches.
- **Interactive Valuation Controls**: Sliders and dropdowns for area, bedrooms, bathrooms, location, age, parking, stories, road access, and furnishing.
- **Valuation Outputs**: Instant prediction, price per sq ft, 68% confidence interval ($\pm \$39,189$ RMSE), property snapshot table, and neighborhood mean comparison.

---

## 13. Results and Discussion

1. **Feature Drivers**: Living area is the primary linear driver, followed by neighborhood tier and bathroom-to-bedroom layout balance.
2. **Polynomial Superiority**: Second-degree terms capture the real-world premium of large homes situated specifically in high-tier neighborhoods (`Waterfront Bay` and `Downtown`).
3. **Homoscedasticity Verified**: Residual plots display a symmetric, zero-centered bell curve with constant variance across the prediction range.

---

## 14. Conclusion and Future Scope

### 14.1 Conclusion
The **ResiVal** system successfully fulfills all requirements of the Machine Learning Fundamentals curriculum. By combining IQR Winsorization, polynomial expansion, $L_2$ regularization, and theme-adaptive Streamlit deployment, the project delivers an end-to-end, production-grade property valuation engine.

### 14.2 Future Scope
1. **Geospatial Coordinates**: Integrating latitude/longitude coordinates to compute distance-to-CBD decay matrices.
2. **Gradient Boosting**: Stacking Ridge Regression with XGBoost or LightGBM for non-linear step-function boundaries.
3. **Computer Vision**: Ingesting property facade images through a CNN/ViT to score architectural quality.

---

## 15. References

1. Rosen, S. (1974). *Hedonic prices and implicit markets: product differentiation in pure competition*. Journal of Political Economy, 82(1), 34-55.
2. Hoerl, A. E., & Kennard, R. W. (1970). *Ridge regression: Biased estimation for nonorthogonal problems*. Technometrics, 12(1), 55-67.
3. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
4. McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. Proceedings of SciPy.

---

## 16. Individual Contribution Details

| Group Member | Module Responsibility | Key Deliverables |
| :--- | :--- | :--- |
| **Member 1** | Problem Definition & Data Sourcing | Formulated problem statement, designed mathematical data generation schema, verified feature bounds. |
| **Member 2** | Preprocessing, EDA & Feature Engineering | Outlier audits (IQR & Z-score), Winsorization capping, EDA visualizations, correlation heatmaps, feature engineering. |
| **Member 3** | Regression Modeling & Evaluation | Implemented Simple Linear, Multiple Linear, Polynomial, and Ridge models; computed MAE, MSE, RMSE, and $R^2$; bias-variance analysis. |
| **Member 4** | Model Selection, Streamlit & Report | Residual diagnostics, pipeline serialization (`joblib`), theme-adaptive Streamlit UI development, technical report authoring. |
