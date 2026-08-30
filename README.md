# E-Commerce Customer Churn: Data Preprocessing & Feature Selection

## 1. Project Title
Predicting E-Commerce Customer Churn : A From-Scratch Data Preprocessing and Statistical Feature Selection Pipeline

## 2. Problem Statement
Customer churn — when a customer stops using a company's platform — is one of the most costly problems in e-commerce, since acquiring a new customer is typically far more expensive than retaining an existing one. This project analyzes a real-world e-commerce customer dataset to understand the behavioral, demographic, and transactional factors associated with churn.

The objective is not to build a final predictive model, but to demonstrate a complete, justified, from-scratch understanding of the full preprocessing and feature-selection pipeline — from raw, messy data to a clean, statistically-validated set of predictive features — following the process: **Theory → Formula/Logic → From-Scratch Code → Output → Library Verification → Interpretation.**

- **Target variable:** `Churn` (1 = customer churned, 0 = customer retained)
- **Problem type:** Binary Classification

## 3. Group Members

| Student | Roll Number | Contribution |
|---|---|---|
| Anmol Pandey | 06 | B1,E1,E2,F2,H2,K,L,M4,M6,M7,M9,N |
| Drishti Nigam | 56 | A2,B2,D1,D2,F1,H1,L,M1,M3,M9,N|
|Abhishek Saini  | 02 | C1,C2,F3,G,I,J,L,M2,M8,M9,N |




## 4. Dataset Description

- **Total records:** 5,630 customers
- **Total raw features:** 18 input features + 1 target (`Churn`) + 1 Primary Key (`CustomerID`)
- **Numerical features (13):** Tenure, CityTier, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount
- **Categorical features (5):** PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus
- **Target variable:** Churn (binary — 0 = Retained, 1 = Churned)
- **Missing values:** 1,478 missing values spread across 7 numerical columns (Tenure, WarehouseToHome, HourSpendOnApp, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder — each with roughly 4.4%–5.3% missingness)
- **Duplicate records:** 0 (verified explicitly, not assumed)

**Column Name**                         | **Data Type**       | **Unique Values**   | **Missing Values**
|---|---|---|---|
CustomerID                          | int64           | 4504            | 0
Churn                               | int64           | 2               | 0
Tenure                              | float64         | 35              | 218
PreferredLoginDevice                | object          | 3               | 0
CityTier                            | int64           | 3               | 0
WarehouseToHome                     | float64         | 34              | 206
PreferredPaymentMode                | object          | 7               | 0
Gender                              | object          | 2               | 0
HourSpendOnApp                      | float64         | 6               | 198
NumberOfDeviceRegistered            | int64           | 6               | 0
PreferedOrderCat                    | object          | 6               | 0
SatisfactionScore                   | int64           | 5               | 0
MaritalStatus                       | object          | 3               | 0
NumberOfAddress                     | int64           | 13              | 0
Complain                            | int64           | 2               | 0
OrderAmountHikeFromlastYear         | float64         | 16              | 212
CouponUsed                          | float64         | 16              | 205
OrderCount                          | float64         | 16              | 202
DaySinceLastOrder                   | float64         | 22              | 237
CashbackAmount                      | float64         | 2464            | 0


## 5. Dataset Source
**Kaggle — Ecommerce Customer Churn Analysis and Prediction**
https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

## 6. Preprocessing Techniques Implemented

| Technique | Purpose | Implementation |
|---|---|---|
| Missing Value Treatment | Handle 1,478 missing values across 7 numeric columns | Mean/Median imputation, chosen per-column based on skew (mean-median gap) |
| Duplicate Detection | Verify data integrity | Manual ID-scan logic; 0 duplicates found |
| Inconsistent Data Handling | Fix spelling/casing/spacing issues in categorical text | Manual mapping dictionary + `.strip().title()` cleanup |
| Label Encoding | Encode binary `Gender` feature | Manual dictionary-based mapping |
| One-Hot Encoding | Encode multi-category nominal features | Manual binary column generation |
| Outlier Detection | Identify extreme values | IQR method (1,824 outliers) and Z-Score method (293 outliers) — from scratch |
| Outlier Treatment | Handle skew/outliers without deleting data | Log(x+1) transformation applied to `CashbackAmount` and `OrderAmountHikeFromlastYear` |
| Min-Max Normalization | Scale numeric features to [0,1] | From-scratch formula implementation |
| Standardization | Scale numeric features to mean=0, std=1 | From-scratch formula implementation |
| Train-Test Split | Create unbiased evaluation set | Manual random shuffle + index-based 80/20 split (no `train_test_split()`) |

## 7. Feature-Selection Techniques Implemented

| Technique | Applied To | Purpose |
|---|---|---|
| Variance Threshold | 13 numerical/encoded features | Detect zero/near-zero-variance features |
| Pearson Correlation | 13 numerical features vs. Churn | Measure linear feature–target and feature–feature relationships |
| Chi-Square Test | 7 categorical/discrete features vs. Churn | Test statistical dependence between categorical variables and target |
| ANOVA F-Test | 13 numerical features vs. Churn groups | Test whether feature means differ significantly across churn groups |

## 8. From-Scratch Implementations

All core statistical logic was implemented manually using base Python/NumPy/Pandas loops and formulas, then verified against the equivalent library function:

- Mean, Median, Mode, Variance, Standard Deviation (Task B2)
- Missing value imputation logic (mean/median/mode fill, no `SimpleImputer`)
- Label Encoding & One-Hot Encoding (no `LabelEncoder`/`OneHotEncoder`)
- IQR and Z-Score outlier detection (no library outlier detectors)
- Min-Max Normalization and Standardization formulas (no `MinMaxScaler`/`StandardScaler`)
- Train-Test Split via random shuffling and index slicing (no `train_test_split()`)
- Variance Threshold (no `VarianceThreshold`)
- Pearson Correlation coefficient formula
- Chi-Square contingency table, expected frequencies, and statistic (no `chi2_contingency` for the core calculation)
- ANOVA F-Test (SSB, SSW, F-statistic) using Task B2's mean/variance functions

Every from-scratch result was verified against its library equivalent (NumPy, Pandas, SciPy, Scikit-learn) for correctness.

## 9. Results

| Pipeline Stage | Number of Features |
|---|---|
| Original Dataset | 20 (18 input + ID + target) |
| After Preprocessing (Cleaning & Encoding) | 36 |
| After Feature Selection | 15 |

| Parameter | Before Preprocessing | After Preprocessing |
|---|---|---|
| Records | 5,630 | 4,504(train-split) |
| Features | 19 | 35 |
| Missing Values | 1,478 | 0 |
| Duplicate Records | 0 | 0 |
| Categorical Features (raw text) | 7 | 0 (fully encoded) |
| Outliers (IQR Method) | 1,824 | 1,824 detected — treated via transformation, not deletion |
| Outliers (Z-Score Method) | 293 | 293 detected |
| Selected Features | 18 | 26 |

## 10. Selected Features

**Kept (15):** Tenure, Complain, DaySinceLastOrder, CashbackAmount, SatisfactionScore, NumberOfDeviceRegistered, CityTier, WarehouseToHome, NumberOfAddress, OrderCount, PreferedOrderCat, MaritalStatus, PreferredPaymentMode, PreferredLoginDevice, Gender

**Removed (3):**
- **HourSpendOnApp** — ANOVA F-score (1.66) below the critical value (3.84); near-zero Pearson correlation (0.019). No evidence of relationship with churn.
- **CouponUsed** — ANOVA F-score (0.32) not significant; near-zero Pearson correlation (-0.008).
- **OrderAmountHikeFromlastYear** — ANOVA F-score essentially zero (0.0001); Pearson correlation effectively zero (-0.0002).

All three removed features failed significance under **two independent statistical tests** (ANOVA and Pearson), strengthening the justification for removal.

## 11. Key Findings

- **`Complain` is the single strongest churn indicator among categorical/binary features** — customers who raised a complaint churned at ~31.5%, nearly **3× the rate** of customers who didn't complain (~11.0%), confirmed by both Chi-Square (274.19) and ANOVA (291.98).
- **`Tenure` is the strongest overall predictor** (ANOVA F = 591.20, Pearson r = -0.34) — newer customers churn significantly more than long-tenured ones.
- Features with weak Pearson correlation (`CashbackAmount`, `WarehouseToHome`) were still found significant via ANOVA — confirming the assignment's key caution that **low linear correlation does not rule out a real (possibly nonlinear) relationship**.
- No feature showed zero or near-zero variance strongly enough to warrant removal on that basis alone — Variance Threshold served as a confirmatory check rather than a primary filter for this dataset.
- Dataset had 0 duplicate records — verified explicitly rather than assumed, consistent with rigorous data validation practice.

## 12. Instructions to Run the Code

1. Clone this repository:
   ```
   git clone https://github.com/DrishtiNigam1928/CUSTOMER_CHURN.git
   ```
2. Open `notebooks/main_analysis.ipynb` in Google Colab (or Jupyter).
3. Download the dataset from the Kaggle link above, or load directly from `dataset/train_dataset.csv` and `dataset/test_dataset.csv` if already generated.
4. Run all cells sequentially from top to bottom — each task (A2 through N) builds on outputs from earlier cells.
5. Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn` (used only for verification, not core logic).

## 13. Google Colab Link


---
**Repository:** https://github.com/DrishtiNigam1928/CUSTOMER_CHURN
