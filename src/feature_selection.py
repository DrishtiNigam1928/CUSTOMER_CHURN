import pandas as pd
import numpy as np
from preprocessing import df_cleaned
from scipy.stats import chi2 as chi2_dist
from scipy.stats import chi2_contingency
from scipy.stats import f as f_dist
import matplotlib.pyplot as plt
import seaborn as sns
# from IPython.display import display

# FILENAME = "CUSTOMER_CHURN/dataset/train_dataset.csv"
df = pd.read_csv("train_dataset.csv")

print("Shape:", df.shape)
df.head()

target_col = "Churn"
id_col = "CustomerID"

num_rows = 0
for _ in df.itertuples():
    num_rows += 1
print("Number of rows/observations:", num_rows)

# 2. Number of columns/features (without .shape)
column_list = []
for col in df:
    column_list.append(col)
num_cols = len(column_list)
print("Number of columns/features:", num_cols)
print("Column names:", column_list)

features_to_test = [
    'Tenure',
    'WarehouseToHome',
    'HourSpendOnApp',
    'NumberOfDeviceRegistered',
    'NumberOfAddress',
    'OrderCount',
    'CouponUsed',
    'DaySinceLastOrder',
    'CityTier',
    'SatisfactionScore',
    'OrderAmountHikeFromlastYear',
    'Complain',
    'CashbackAmount'
]

def calc_mean(values):
    return sum(values) / len(values)

def calc_min(values):
    m = values[0]
    for v in values:
        if v < m:
            m = v
    return m

def calc_max(values):
    m = values[0]
    for v in values:
        if v > m:
            m = v
    return m

def calc_median(values):
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    else:
        return sorted_vals[mid]

def calc_mode(values):
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    max_count = max(freq.values())
    for v, c in freq.items():
        if c == max_count:
            return v

def calc_variance(values, mean_val):
    squared_diffs = [(v - mean_val) ** 2 for v in values]
    return sum(squared_diffs) / len(values)   # population variance, matches the formula given

def calc_std(variance_val):
    return variance_val ** 0.5

def calc_range(min_val, max_val):
    return max_val - min_val


# Task M1
print("Total features being checked:", len(features_to_test))
print(features_to_test)
print()

# --- Calculate variance for each selected feature ---
variance_report = []
def calc_mean(values):
    return sum(values) / len(values)

def calc_variance(values, mean_val):
    squared_diffs = [(v - mean_val) ** 2 for v in values]
    return sum(squared_diffs) / len(values)
for col in features_to_test:
    values = df_cleaned[col].dropna().tolist()
    mean_val = calc_mean(values)
    var_val = calc_variance(values, mean_val)
    variance_report.append({"Feature": col, "Variance": var_val})

variance_table = pd.DataFrame(variance_report).sort_values("Variance").reset_index(drop=True)
print(variance_table)
print()

# --- Library Verification using NumPy variance (np.var) ---
lib_variance_report = []

for col in features_to_test:
    values = df_cleaned[col].dropna().values
    lib_var = np.var(values, ddof=0)
    lib_variance_report.append({"Feature": col, "Lib Variance": lib_var})

lib_check = pd.DataFrame(lib_variance_report)

merged_check = variance_table.merge(lib_check, on="Feature")
merged_check["Match"] = np.isclose(merged_check["Variance"], merged_check["Lib Variance"], atol=1e-6)

print("Library Verification using NumPy (np.var) for M1 Task")
print(merged_check[["Feature", "Variance", "Lib Variance", "Match"]])
print()


# --- Classify + decide Keep/Remove
def classify_variance(v):
    if v == 0:
        return "Zero-variance"
    elif v < 0.01:
        return "Near-zero-variance"
    elif v < 1:
        return "Low-variance"
    else:
        return "Acceptable"

variance_table["Category"] = variance_table["Variance"].apply(classify_variance)

variance_table["Decision"] = variance_table["Category"].apply(
    lambda c: "Remove" if c in ["Zero-variance", "Near-zero-variance"] else "Keep"
)

variance_table["Reason"] = variance_table.apply(
    lambda row: f"{row['Category']} ({row['Variance']:.4f}) -> little/no discriminatory information"
    if row["Decision"] == "Remove"
    else f"{row['Category']} ({row['Variance']:.4f}) -> sufficient spread, potentially useful",
    axis=1
)

print("Task M1: Variance Threshold completed successfully from scratch!\n")
print(variance_table)
print()


# M2: Pearson Correlation Function

print("Number of numerical features:", len(features_to_test))
print(features_to_test)
print()

def calc_pearson(x, y):

    # Remove rows where either value is missing
    paired_values = []

    for i in range(len(x)):
        if pd.notna(x[i]) and pd.notna(y[i]):
            paired_values.append((x[i], y[i]))

    x_clean = [pair[0] for pair in paired_values]
    y_clean = [pair[1] for pair in paired_values]

    # Calculate means
    mean_x = calc_mean(x_clean)
    mean_y = calc_mean(y_clean)

    # Calculate numerator and denominator
    numerator = 0
    sum_x_squared = 0
    sum_y_squared = 0

    for i in range(len(x_clean)):

        x_diff = x_clean[i] - mean_x
        y_diff = y_clean[i] - mean_y

        numerator += x_diff * y_diff
        sum_x_squared += x_diff ** 2
        sum_y_squared += y_diff ** 2

    denominator = (sum_x_squared * sum_y_squared) ** 0.5

    # Avoid division by zero
    if denominator == 0:
        return 0

    return numerator / denominator


# M2: Feature-Target Correlation
feature_target_results = []

for feature in features_to_test:

    x = df_cleaned[feature].tolist()
    y = df_cleaned[target_col].tolist()

    correlation = calc_pearson(x, y)

    feature_target_results.append({
        "Feature": feature,
        "Target": target_col,
        "Pearson Correlation": correlation
    })

feature_target_table = pd.DataFrame(feature_target_results)

feature_target_table["Pearson Correlation"] = \
    feature_target_table["Pearson Correlation"].round(4)

print(feature_target_table)
print()

# M2: Feature-Feature Correlation Matrix

correlation_matrix = pd.DataFrame(
    0.0,
    index=features_to_test,
    columns=features_to_test
)

for i in range(len(features_to_test)):

    for j in range(len(features_to_test)):

        feature_1 = features_to_test[i]
        feature_2 = features_to_test[j]

        x = df_cleaned[feature_1].tolist()
        y = df_cleaned[feature_2].tolist()

        correlation = calc_pearson(x, y)

        correlation_matrix.loc[
            feature_1,
            feature_2
        ] = correlation

print(correlation_matrix.round(4))
print()

# M2: Identify Potentially Redundant Features

correlation_threshold = 0.80

redundant_features = []

for i in range(len(features_to_test)):

    for j in range(i + 1, len(features_to_test)):

        feature_1 = features_to_test[i]
        feature_2 = features_to_test[j]

        correlation = correlation_matrix.loc[
            feature_1,
            feature_2
        ]

        if abs(correlation) >= correlation_threshold:

            redundant_features.append({
                "Feature 1": feature_1,
                "Feature 2": feature_2,
                "Pearson Correlation": round(correlation, 4)
            })

redundant_features_table = pd.DataFrame(
    redundant_features
)

if len(redundant_features_table) == 0:

    print(
        "No highly correlated feature pairs found "
        "at threshold", correlation_threshold
    )

else:

    print(redundant_features_table)
print()

# M2: Pearson Correlation Heatmap

# plt.figure(figsize=(14, 10))

# sns.heatmap(
#     correlation_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="coolwarm",
#     center=0,
#     linewidths=0.5
# )

# plt.title(
#     "Pearson Correlation Heatmap - Numerical Features",
#     fontsize=15,
#     fontweight="bold"
# )

# plt.tight_layout()
# plt.show()

# M2: Feature-Target Correlation Interpretation

def interpret_correlation(r):

    if abs(r) >= 0.80:
        strength = "Strong"
    elif abs(r) >= 0.50:
        strength = "Moderate"
    elif abs(r) >= 0.30:
        strength = "Weak"
    else:
        strength = "Very Weak / Near Zero"

    if r > 0:
        direction = "Positive"
    elif r < 0:
        direction = "Negative"
    else:
        direction = "Zero"

    return direction + " " + strength


interpretation_results = []

for row in feature_target_results:

    feature = row["Feature"]
    correlation = row["Pearson Correlation"]

    interpretation_results.append({
        "Feature": feature,
        "Correlation with Churn": round(correlation, 4),
        "Interpretation": interpret_correlation(correlation)
    })

interpretation_table = pd.DataFrame(
    interpretation_results
)

print(interpretation_table)
print()

# M2: Library Verification of Correlation Matrix

lib_correlation_matrix = df_cleaned[features_to_test].corr()

# Compare manual correlation matrix with library result
correlation_match = np.isclose(
    correlation_matrix.values,
    lib_correlation_matrix.values,
    atol=1e-6
)

print("All correlations match:",correlation_match.all())
print()

# --- Task M3: Chi-Square Feature Selection
categorical_features_for_chi2 = ["Gender", "PreferredLoginDevice", "PreferredPaymentMode",
                                  "PreferedOrderCat", "MaritalStatus", "CityTier", "Complain"
                                ]

categorical_features_for_chi2 = [c for c in categorical_features_for_chi2 if c in df_cleaned.columns]
print("Features to test:", categorical_features_for_chi2)
print()

# Contingency table + Observed frequencies ---
def build_contingency_table(df, feature_col, target_col):
    contingency = {}
    for _, row in df[[feature_col, target_col]].dropna().iterrows():
        f_val = row[feature_col]
        t_val = row[target_col]
        if f_val not in contingency:
            contingency[f_val] = {}
        contingency[f_val][t_val] = contingency[f_val].get(t_val, 0) + 1
    return contingency

def contingency_to_dataframe(contingency, target_categories):
    rows = []
    for f_val, counts in contingency.items():
        row = {"Category": f_val}
        for t_val in target_categories:
            row[f"Churn={t_val}"] = counts.get(t_val, 0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Category")
#  Expected frequencies, contributions, total Chi-Square, degrees of freedom ---
def calc_chi_square(contingency_df):
    row_totals = contingency_df.sum(axis=1)
    col_totals = contingency_df.sum(axis=0)
    grand_total = contingency_df.values.sum()

    expected_df = pd.DataFrame(index=contingency_df.index, columns=contingency_df.columns, dtype=float)
    chi2_contrib_df = pd.DataFrame(index=contingency_df.index, columns=contingency_df.columns, dtype=float)

    for row_cat in contingency_df.index:
        for col_cat in contingency_df.columns:
            O = contingency_df.loc[row_cat, col_cat]
            E = (row_totals[row_cat] * col_totals[col_cat]) / grand_total   # Expected frequency formula
            expected_df.loc[row_cat, col_cat] = E
            chi2_contrib_df.loc[row_cat, col_cat] = ((O - E) ** 2) / E if E != 0 else 0

    total_chi2 = chi2_contrib_df.values.sum()
    r = contingency_df.shape[0]
    c = contingency_df.shape[1]
    degrees_of_freedom = (r - 1) * (c - 1)

    return expected_df, chi2_contrib_df, total_chi2, degrees_of_freedom

  # --- Apply Chi-Square to every candidate feature ---
target_categories = sorted(df_cleaned["Churn"].dropna().unique())
chi2_summary = []
chi2_details = {}

for col in categorical_features_for_chi2:
    contingency_raw = build_contingency_table(df_cleaned, col, "Churn")
    contingency_df = contingency_to_dataframe(contingency_raw, target_categories)

    expected_df, chi2_contrib_df, total_chi2, dof = calc_chi_square(contingency_df)

    chi2_details[col] = {
        "Observed": contingency_df,
        "Expected": expected_df,
        "Contributions": chi2_contrib_df
    }

    chi2_summary.append({
        "Feature": col,
        "Chi-Square Statistic": round(total_chi2, 3),
        "Degrees of Freedom": dof
    })

chi2_table = pd.DataFrame(chi2_summary).sort_values("Chi-Square Statistic", ascending=False)
print(chi2_table)
print()

# --- Interpretation: compare p-value against alpha (significance level) ---


alpha = 0.05

chi2_table["p-value"] = chi2_table.apply(
    lambda row: 1 - chi2_dist.cdf(row["Chi-Square Statistic"], row["Degrees of Freedom"]),
    axis=1
)
chi2_table["Significant (p < alpha)"] = chi2_table["p-value"] < alpha

print(f"Using alpha (significance level) = {alpha}")
print(chi2_table)
print()


# --- Display one feature's full tables
sample_feature = chi2_table.iloc[0]["Feature"]

print(f"--- {sample_feature}: Observed Frequencies (Contingency Table) ---")
print(chi2_details[sample_feature]["Observed"])

print(f"\n--- {sample_feature}: Expected Frequencies ---")
print(chi2_details[sample_feature]["Expected"].round(2))

print(f"\n--- {sample_feature}: Chi-Square Contributions per Cell ---")
print(chi2_details[sample_feature]["Contributions"].round(3))
print()

verify_rows = []
for col in categorical_features_for_chi2:
    ct = pd.crosstab(df_cleaned[col], df_cleaned["Churn"])
    lib_chi2, lib_p, lib_dof, _ = chi2_contingency(ct, correction=False)   # <-- disable Yates' correction

    my_row = chi2_table[chi2_table["Feature"] == col].iloc[0]
    verify_rows.append({
        "Feature": col,
        "My Chi2": my_row["Chi-Square Statistic"],
        "Lib Chi2 (no correction)": round(lib_chi2, 3),
        "Match": np.isclose(my_row["Chi-Square Statistic"], lib_chi2, atol=0.01)
    })

print(pd.DataFrame(verify_rows))
print()

# M4: ANOVA F-TEST





features_for_anova = [col for col in features_to_test if col in df_cleaned.columns]
anova_df = df_cleaned[features_for_anova + ['Churn']].dropna()

anova_results_scratch = []

n = len(anova_df)
groups = anova_df['Churn'].unique()
k = len(groups)

# Degrees of freedom
df_between = k - 1
df_within = n - k

alpha = 0.05
critical_f_value = f_dist.ppf(1 - alpha, df_between, df_within)

for col in features_for_anova:
    all_values = anova_df[col].tolist()

    # Mean using custom B2 functions
    overall_mean = calc_mean(all_values)

    # Calculating SSB and SSW
    ssb = 0.0
    ssw = 0.0
    for g in groups:
        group_data = anova_df[anova_df['Churn'] == g][col].tolist()
        n_g = len(group_data)

        if n_g > 0:
            # Group Mean using custom B2 function
            g_mean = calc_mean(group_data)
            ssb += n_g * ((g_mean - overall_mean) ** 2)

            # SSW using custom B2 function
            g_var = calc_variance(group_data, g_mean)
            ssw += g_var * (n_g - 1)

    # Mean Squares
    ms_between = ssb / df_between if df_between > 0 else 0
    ms_within = ssw / df_within if df_within > 0 else 0

    # F-Statistic
    f_score = ms_between / ms_within if ms_within > 0 else 0.0

    is_significant = f_score > critical_f_value

    anova_results_scratch.append({
        'Feature': col,
        'F-Score (Scratch)': round(f_score, 4),
        'Critical F': round(critical_f_value, 4),
        'Significant (Alpha = 0.05)': is_significant,
        'SSB': round(ssb, 4),
        'SSW': round(ssw, 4)
    })

anova_scratch_table = pd.DataFrame(anova_results_scratch).sort_values(by='F-Score (Scratch)', ascending=False)

print("Task M4: ANOVA F-Test computed successfully from scratch using B2 functions!")
print(anova_scratch_table)
print()

# TASK M6: FEATURE SELECTION MAPPING TABLE



feature_selection_mapping_table = [
    {
        "Method": "Variance Threshold",
        "Features": "Tenure, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered, NumberOfAddress, OrderCount, CouponUsed, DaySinceLastOrder, CityTier, SatisfactionScore, OrderAmountHikeFromlastYear, Complain, CashbackAmount",
        "Feature Type": "Numerical",
        "Target Type": "Not required",
        "Main Purpose": "Detect low-variance features"
    },
    {
        "Method": "Pearson Correlation",
        "Features": "Tenure, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered, NumberOfAddress, OrderCount, CouponUsed, DaySinceLastOrder, CityTier, SatisfactionScore, OrderAmountHikeFromlastYear, Complain, CashbackAmount",
        "Feature Type": "Numerical",
        "Target Type": "Numerical",
        "Main Purpose": "Linear relationship evaluation"
    },
    {
        "Method": "Chi-Square Test",
        "Features": "Gender, PreferredLoginDevice, PreferredPaymentMode, PreferedOrderCat, MaritalStatus, CityTier, Complain",
        "Feature Type": "Categorical/Discrete",
        "Target Type": "Categorical",
        "Main Purpose": "Statistical dependence of categories"
    },
    {
        "Method": "ANOVA F-Test",
        "Features": "Tenure, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered, NumberOfAddress, OrderCount, CouponUsed, DaySinceLastOrder, CityTier, SatisfactionScore, OrderAmountHikeFromlastYear, Complain, CashbackAmount",
        "Feature Type": "Numerical",
        "Target Type": "Categorical",
        "Main Purpose": "Difference across target groups [Churn]"
    }
]

pd.set_option('display.max_colwidth', None)
df_mapping_table = pd.DataFrame(feature_selection_mapping_table)

print("Task M6: Updated Feature Selection Technique & Feature Mapping Table")
print(df_mapping_table)
print()

# TASK M7: FINAL DECISION TABLE



final_decision_table = [
    {
        "Feature": "Tenure",
        "Data Type": "Numerical",
        "Tests Used": "Var, Pearson, ANOVA",
        "Scores": "Var: 70.10 | r: -0.3406 | F: 591.20",
        "Keep/Remove": "Keep",
        "Justification": "High variance spread; strong negative linear correlation and massive ANOVA F-score (Sig)."
    },
    {
        "Feature": "Complain",
        "Data Type": "Binary / Categorical",
        "Tests Used": "Var, Pearson, Chi2, ANOVA",
        "Scores": "Var: 0.20 | r: 0.2467 | Chi2: 274.19 | F: 291.98",
        "Keep/Remove": "Keep",
        "Justification": "Exceptional statistical dependence with churn across all metric tests."
    },
    {
        "Feature": "DaySinceLastOrder",
        "Data Type": "Numerical",
        "Tests Used": "Var, Pearson, ANOVA",
        "Scores": "Var: 12.91 | r: -0.1572 | F: 114.07",
        "Keep/Remove": "Keep",
        "Justification": "Healthy variance, meaningful negative correlation, and high ANOVA F-score (Sig)."
    },
    {
        "Feature": "CashbackAmount",
        "Data Type": "Numerical",
        "Tests Used": "Var, Pearson, ANOVA",
        "Scores": "Var: 2424.58 | r: -0.1564 | F: 112.89",
        "Keep/Remove": "Keep",
        "Justification": "Massive variance spread, robust linear correlation, and significant ANOVA F-score."
    },
    {
        "Feature": "SatisfactionScore",
        "Data Type": "Numerical / Discrete",
        "Tests Used": "Var, Pearson, Chi2, ANOVA",
        "Scores": "Var: 1.92 | r: 0.1134 | Chi2: 63.77 | F: 58.66",
        "Keep/Remove": "Keep",
        "Justification": "Passed variance check; strong performance across both Chi-Square and ANOVA."
    },
    {
        "Feature": "NumberOfDeviceRegistered",
        "Data Type": "Numerical",
        "Tests Used": "Var, Pearson, ANOVA",
        "Scores": "Var: 1.03 | r: 0.1114 | F: 56.55",
        "Keep/Remove": "Keep",
        "Justification": "Acceptable variance and well above the critical ANOVA threshold (3.84)."
    },
    {
        "Feature": "CityTier",
        "Data Type": "Numerical / Discrete",
        "Tests Used": "Var, Pearson, Chi2, ANOVA",
        "Scores": "Var: 0.84 | r: 0.0769 | Chi2: 27.92 | F: 26.81",
        "Keep/Remove": "Keep",
        "Justification": "Sufficient variance and solid statistical dependence indicators."
    },
    {
        "Feature": "WarehouseToHome",
        "Data Type": "Numerical",
        "Tests Used": "Var, Pearson, ANOVA",
        "Scores": "Var: 70.24 | r: 0.0674 | F: 20.58",
        "Keep/Remove": "Keep",
        "Justification": "High variance spread and clear significance in ANOVA testing."
    },
    {
        "Feature": "NumberOfAddress",
        "Data Type": "Numerical",
        "Tests Used": "Var, Pearson, ANOVA",
        "Scores": "Var: 6.63 | r: 0.0460 | F: 9.54",
        "Keep/Remove": "Keep",
        "Justification": "Acceptable variance and clears the critical ANOVA threshold."
    },
    {
        "Feature": "OrderCount",
        "Data Type": "Numerical",
        "Tests Used": "Var, Pearson, ANOVA",
        "Scores": "Var: 8.39 | r: -0.0317 | F: 4.54",
        "Keep/Remove": "Keep",
        "Justification": "Sufficient variance; marginally passes the critical ANOVA threshold (4.54 > 3.84)."
    },
    {
        "Feature": "HourSpendOnApp",
        "Data Type": "Numerical",
        "Tests Used": "Var, Pearson, ANOVA",
        "Scores": "Var: 0.50 | r: 0.0192 | F: 1.6593 (False)",
        "Keep/Remove": "Remove",
        "Justification": "F-score below critical value (3.84) and near-zero correlation; lacks target separation."
    },
    {
        "Feature": "CouponUsed",
        "Data Type": "Numerical",
        "Tests Used": "Var, Pearson, ANOVA",
        "Scores": "Var: 3.36 | r: -0.0084 | F: 0.3166 (False)",
        "Keep/Remove": "Remove",
        "Justification": "Negligible linear correlation and failed ANOVA significance test."
    },
    {
        "Feature": "OrderAmountHikeFromlastYear",
        "Data Type": "Numerical",
        "Tests Used": "Var, Pearson, ANOVA",
        "Scores": "Var: 12.91 | r: -0.0002 | F: 0.0001 (False)",
        "Keep/Remove": "Remove",
        "Justification": "Flat Pearson correlation (-0.0002) and virtually zero ANOVA F-score."
    },
    {
        "Feature": "PreferedOrderCat",
        "Data Type": "Categorical",
        "Tests Used": "Chi-Square Test",
        "Scores": "Chi2: 242.62 (DF: 4)",
        "Keep/Remove": "Keep",
        "Justification": "Extremely high Chi-Square statistic confirming strong categorical dependence."
    },
    {
        "Feature": "MaritalStatus",
        "Data Type": "Categorical",
        "Tests Used": "Chi-Square Test",
        "Scores": "Chi2: 154.81 (DF: 2)",
        "Keep/Remove": "Keep",
        "Justification": "Strong statistical association with customer churn behavior."
    },
    {
        "Feature": "PreferredPaymentMode",
        "Data Type": "Categorical",
        "Tests Used": "Chi-Square Test",
        "Scores": "Chi2: 41.85 (DF: 4)",
        "Keep/Remove": "Keep",
        "Justification": "Significant Chi-Square value indicating payment mode preference affects target."
    },
    {
        "Feature": "PreferredLoginDevice",
        "Data Type": "Categorical",
        "Tests Used": "Chi-Square Test",
        "Scores": "Chi2: 16.16 (DF: 1)",
        "Keep/Remove": "Keep",
        "Justification": "Exceeds critical value; confirms dependence on login device."
    },
    {
        "Feature": "Gender",
        "Data Type": "Categorical",
        "Tests Used": "Chi-Square Test",
        "Scores": "Chi2: 4.24 (DF: 1)",
        "Keep/Remove": "Keep",
        "Justification": "Passes critical threshold (alpha = 0.05, critical approx 3.84)."
    }
]

df_decision_table = pd.DataFrame(final_decision_table)
print("Task M7: Final Decision Table using Exact Notebook Statistics")
print(df_decision_table)
print()


# M8: Before vs After Feature Selection

features_after_preprocessing=df_cleaned.shape[1]

before_vs_after_feature_selection = [
    {
        "Pipeline Stage": "Original Dataset",
        "Number of Features": 20,
        "Description": "Raw input dataset containing all initial numerical, categorical, and identifier columns."
    },
    {
        "Pipeline Stage": "After Preprocessing (Cleaning & Encoding)",
        "Number of Features": features_after_preprocessing,
        "Description": "Dataset after handling missing values and encoding categorical features (e.g., Gender_Encoded added while maintaining column count or accounting for dropped IDs)."
    },
    {
        "Pipeline Stage": "After Feature Selection (ANOVA / Chi2 / Variance)",
        "Number of Features": 15,
        "Description": "Final optimized feature subset after systematically dropping non-significant features and identifiers."
    }
]
after_vs_before_table = pd.DataFrame(before_vs_after_feature_selection)
pd.set_option('display.max_colwidth', None)

print("Task M8: Before vs After Feature Selection")
print(after_vs_before_table)
print()


# Task N: Before vs After Preprocessing Comparison Table



preprocessing_exact_data = [
    {
        "Parameter": "Records",
        "Before Preprocessing": "5,630",
        "After Preprocessing": "4,504"
    },
    {
        "Parameter": "Features",
        "Before Preprocessing": "19",
        "After Preprocessing": "35"
    },
    {
        "Parameter": "Missing Values",
        "Before Preprocessing": "1,478",
        "After Preprocessing": "0"
    },
    {
        "Parameter": "Duplicate Records",
        "Before Preprocessing": "0",
        "After Preprocessing": "0"
    },
    {
        "Parameter": "Categorical Features",
        "Before Preprocessing": "7",
        "After Preprocessing": "0"
    },
    {
        "Parameter": "Outliers (IQR Method)",
        "Before Preprocessing": "1,824",
        "After Preprocessing": "1,824"
    },
    {
        "Parameter": "Outliers (Z-Score Method)",
        "Before Preprocessing": "293",
        "After Preprocessing": "293"
    },
    {
        "Parameter": "Selected Features",
        "Before Preprocessing": "18",
        "After Preprocessing": "26"
    }
]

pd.set_option('display.max_colwidth', None)
df_preprocessing_exact = pd.DataFrame(preprocessing_exact_data)

print("Task N: Before vs After Preprocessing Comparison Table")
print(df_preprocessing_exact)
print()