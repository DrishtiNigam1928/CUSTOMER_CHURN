import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist
# from IPython.display import display

# FILENAME = "CUSTOMER_CHURN/dataset/train_dataset.csv"
df = pd.read_csv("train_dataset.csv")

print("Shape:", df.shape)
df.head()

#A2: Input features & Target variable
target_col = "Churn"
id_col = "CustomerID"

input_features = [c for c in df.columns if c not in [target_col, id_col]]

print("Target variable:", target_col)
print("Number of input features:", len(input_features))
print("Input features:", input_features)


#  A2: Numerical vs Categorical variables -
numerical_vars = df[input_features].select_dtypes(include=np.number).columns.tolist()
categorical_vars = df[input_features].select_dtypes(exclude=np.number).columns.tolist()

print("Numerical variables (", len(numerical_vars), "):", numerical_vars)
print()
print("Categorical variables (", len(categorical_vars), "):", categorical_vars)
print("\n")

# A2: Source of dataset -
print("Source: Kaggle - Ecommerce Customer Churn Analysis and Prediction")
print("URL: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction")


#  All Numerical features
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

def execute_task_B1(dataframe):
  print("Task B1: Initial Data Exploration and Structure Inspection\n")

  print(df.head(3))
  print("\n")
  print(df.tail(3))
  print("\n")
  rows, cols = dataframe.shape
  print(f"Total Number of Rows: {rows}\n")
  print(f"Total Number of Columns: {cols}\n")

  print(f"{'Column Name':<35} | {'Data Type':<15} | {'Unique Values':<15} | {'Missing Values'}")
  for col in dataframe.columns:
    col_name=col
    dtype=str(dataframe[col].dtype)
    unique_vals = dataframe[col].nunique()
    missing_vals = dataframe[col].isnull().sum()
    print(f"{col_name:<35} | {dtype:<15} | {unique_vals:<15} | {missing_vals}")

execute_task_B1(df)
print("\nTask B1 Completed Successfully!")
print("\n")


# B2
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

stats_dict = {}

for col in features_to_test:
    values = [v for v in df[col] if pd.notna(v)]

    col_mean = calc_mean(values)
    col_min = calc_min(values)
    col_max = calc_max(values)
    col_median = calc_median(values)
    col_mode = calc_mode(values)
    col_variance = calc_variance(values, col_mean)
    col_std = calc_std(col_variance)
    col_range = calc_range(col_min, col_max)

    stats_dict[col] = {
        "Mean": col_mean, "Median": col_median, "Mode": col_mode,
        "Min": col_min, "Max": col_max, "Range": col_range,
        "Variance": col_variance, "Std Dev": col_std
    }

stats_table = pd.DataFrame(stats_dict).T
print(stats_table)
print("\n")


# C1
missing_count = df.isnull().sum()
missing_pct = ((missing_count / len(df)) * 100).round(2)

missing_table = pd.DataFrame({
    "Feature": missing_count.index,
    "Missing Values": missing_count.values,
    "Missing %": missing_pct.values
})
missing_table = missing_table[missing_table["Missing Values"] >= 0].sort_values("Missing %", ascending=False)

num_missing_vals = (missing_table["Missing Values"]).sum() # For Task N

print(missing_table)
print("\n")


# Task C2: Handling Missing Values

df_cleaned = df.copy()

# Mean used on HourSpendOnApp (Approximately symmetric)
df_cleaned["HourSpendOnApp"] = df_cleaned["HourSpendOnApp"].fillna(
    stats_dict["HourSpendOnApp"]["Mean"]
)

# Median used on Tenure, WarehouseToHome, OrderAmountHikeFromlastYear and DaySinceLastOrder (skewed data)
for col in ["Tenure", "WarehouseToHome",
            "OrderAmountHikeFromlastYear", "DaySinceLastOrder"]:
    df_cleaned[col] = df_cleaned[col].fillna(
        stats_dict[col]["Median"]
    )

# Mode used on CouponUsed and OrderCount (Discrete/count variable)
for col in ["CouponUsed", "OrderCount"]:
    df_cleaned[col] = df_cleaned[col].fillna(
        stats_dict[col]["Mode"]
    )
print("\n")

# Before → After Missing Value Treatment
missing_features = ["Tenure","WarehouseToHome","HourSpendOnApp","OrderAmountHikeFromlastYear","CouponUsed","OrderCount","DaySinceLastOrder"]

before = df[missing_features].isnull().sum()
after = df_cleaned[missing_features].isnull().sum()

comparison = pd.DataFrame({
    "Feature": missing_features,
    "Before Missing Value Treatment": before.values,
    "After Missing Value Treatment": after.values
})

print(comparison)
print("\n")


# D1: Duplicate Detection
import pandas as pd

original_records = len(df_cleaned)

seen_ids = set()
duplicate_indices = []
duplicate_count = 0

for index, customer_id in enumerate(df_cleaned["CustomerID"]):
    if customer_id in seen_ids:
        duplicate_count += 1
        duplicate_indices.append(index)
    else:
        seen_ids.add(customer_id)

df_cleaned = df_cleaned.drop(duplicate_indices).reset_index(drop=True)
records_after_treatment = len(df_cleaned)

duplicate_summary_table = pd.DataFrame({
    "Item": ["Original Records", "Duplicate Records", "Records After Treatment"],
    "Value": [original_records, duplicate_count, records_after_treatment]
})

print(duplicate_summary_table)
print("\n")


#  D2: Standardize categorical inconsistencies
standardization_map = {
    "PreferredLoginDevice": {
        "Mobile Phone": "Phone",
        "Phone": "Phone",
        "Computer": "Computer"
    },
    "PreferredPaymentMode": {
        "CC": "Credit Card",
        "Credit Card": "Credit Card",
        "COD": "Cash on Delivery",
        "Cash on Delivery": "Cash on Delivery",
        "Debit Card": "Debit Card",
        "E wallet": "E Wallet",
        "UPI": "UPI"
    },
    "PreferedOrderCat": {
        "Mobile Phone": "Mobile",
        "Mobile": "Mobile",
        "Laptop & Accessory": "Laptop & Accessory",
        "Fashion": "Fashion",
        "Grocery": "Grocery",
        "Others": "Others"
    }
}

for col, mapping in standardization_map.items():
    before_unique = df_cleaned[col].nunique()

    # Manual replacement logic
    new_column = []
    for value in df_cleaned[col]:
        if pd.isna(value):
            new_column.append(value)
        else:
            cleaned_value = str(value).strip()          # remove extra spaces
            new_column.append(mapping.get(cleaned_value, cleaned_value))
    df_cleaned[col] = new_column

    after_unique = df_cleaned[col].nunique()
    print(f"{col}: {before_unique} unique values -> {after_unique} unique values after standardization")

print()


# D2: General cleanup for ALL categorical columns
for col in categorical_vars:
    new_column = []
    for value in df_cleaned[col]:
        if pd.isna(value):
            new_column.append(value)
        else:
            cleaned_value = str(value).strip().title()   # remove spaces
            new_column.append(cleaned_value)
    df_cleaned[col] = new_column

print("After general cleanup:")
for col in categorical_vars:
    print(f"{col}: {sorted(df_cleaned[col].unique())}")

print()


# --- D2: Check numeric columns for impossible/negative values
invalid_report = []
for col in features_to_test:
    if col in df_cleaned.columns:
        negative_count = (df_cleaned[col] < 0).sum()
        invalid_report.append({"Feature": col, "Negative Values": negative_count})

print(pd.DataFrame(invalid_report))
print()


# Task E1: Label Encoding for 'Gender'
target_label_col="Gender"

# Extracting unique values and sorting them for deterministic mapping
unique_gender_vals = sorted(list(df_cleaned[target_label_col].dropna().unique()))

# Creating mapping Dictionary
gender_mapping = {val: i for i, val in enumerate(unique_gender_vals)}

encoded_gender= [None]* len(df_cleaned)

for idx, val in enumerate(df_cleaned[target_label_col]):
  if pd.notna(val):
    encoded_gender[idx]=gender_mapping.get(val)

df_cleaned["Gender_Encoded"] = encoded_gender

print("Task E1: Label Encoding Completed Successfully!\n")
print(f"Mapping used for '{target_label_col}': {gender_mapping}\n")
print(df_cleaned[[target_label_col, "Gender_Encoded"]].head(8))
print("\n")
print(df_cleaned["Gender_Encoded"].value_counts())

print()


# Task E2: One Hot Encoding

one_hot_cols = ["PreferredLoginDevice", "PreferredPaymentMode", "PreferedOrderCat", "MaritalStatus"]

for col in one_hot_cols:
  if col in df_cleaned.columns:
    # Get sorted unique values
    unique_vals = sorted(list(df_cleaned[col].dropna().unique()))

    for val in unique_vals:
      # format a clean column name by replacing spaces with underscores like PreferredLoginDevice_Phone
      clean_val_name = str(val).replace(" ", "_")
      new_col_name = f"{col}_{clean_val_name}"

      binary_column = []
      for row_val in df_cleaned[col]:
        if pd.isna(row_val):
          binary_column.append(0)
        elif row_val==val:
          binary_column.append(1)
        else:
          binary_column.append(0)

      # assign the new binary column to the df_cleaned
      df_cleaned[new_col_name] = binary_column

print("Task E2: One-Hot Encoding Completed Successfully!\n")
print(f"Total columns in dataframe after OHE: {len(df_cleaned.columns)}")

# verification of the columns using display() function
ohe_sample_cols = [c for c in df_cleaned.columns if any(col in c for col in one_hot_cols)][:20]
print(df_cleaned[ohe_sample_cols].head(10))

print()


# F1: from-scratch IQR calculation

def calc_percentile(sorted_values, percentile):
    """Calculates the percentile position using linear interpolation (same method pandas uses by default)."""
    n = len(sorted_values)
    index = percentile * (n - 1)
    lower_index = int(index)
    upper_index = lower_index + 1

    if upper_index >= n:
        return sorted_values[lower_index]

    fraction = index - lower_index
    return sorted_values[lower_index] + fraction * (sorted_values[upper_index] - sorted_values[lower_index])

def calc_q1(values):
    sorted_vals = sorted(values)
    return calc_percentile(sorted_vals, 0.25)

def calc_q3(values):
    sorted_vals = sorted(values)
    return calc_percentile(sorted_vals, 0.75)


# F1: from-scratch IQR calculation
def calc_percentile(sorted_values, percentile):
    """Calculates the percentile position using linear interpolation (same method pandas uses by default)."""
    n = len(sorted_values)
    index = percentile * (n - 1)
    lower_index = int(index)
    upper_index = lower_index + 1

    if upper_index >= n:
        return sorted_values[lower_index]

    fraction = index - lower_index
    return sorted_values[lower_index] + fraction * (sorted_values[upper_index] - sorted_values[lower_index])

def calc_q1(values):
    sorted_vals = sorted(values)
    return calc_percentile(sorted_vals, 0.25)

def calc_q3(values):
    sorted_vals = sorted(values)
    return calc_percentile(sorted_vals, 0.75)

# Apply IQR method to each numerical feature
iqr_report = []
outlier_indices_by_col = {}   # store row indices for later decisions

for col in features_to_test:
    values = df_cleaned[col].dropna().tolist()

    q1 = calc_q1(values)
    q3 = calc_q3(values)
    iqr = q3 - q1                       # IQR = Q3 - Q1
    lower_bound = q1 - 1.5 * iqr        # Lower Bound
    upper_bound = q3 + 1.5 * iqr        # Upper Bound

    # Identify observations outside these limits
    outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
    outlier_count = len(outliers)

    outlier_indices_by_col[col] = outliers.index.tolist()

    iqr_report.append({
        "Feature": col, "Q1": round(q1, 2), "Q3": round(q3, 2), "IQR": round(iqr, 2),
        "Lower Bound": round(lower_bound, 2), "Upper Bound": round(upper_bound, 2),
        "Outliers Detected": outlier_count
    })

iqr_table = pd.DataFrame(iqr_report).sort_values("Outliers Detected", ascending=False)

tot_outliers_count_iqr = (iqr_table["Outliers Detected"]).sum() # For Task N

print(iqr_table)
print()

#  Library verification (NumPy quantile)
verify_rows = []
for col in features_to_test:
    values = df_cleaned[col].dropna()
    lib_q1 = np.quantile(values, 0.25)
    lib_q3 = np.quantile(values, 0.75)

    my_row = next(r for r in iqr_report if r["Feature"] == col)
    verify_rows.append({
        "Feature": col,
        "Q1 match": np.isclose(my_row["Q1"], lib_q1, atol=0.01),
        "Q3 match": np.isclose(my_row["Q3"], lib_q3, atol=0.01)
    })

print(pd.DataFrame(verify_rows))
print()


# F2: Applying Z-Score Method to each Numerical Feature

z_score_report = []
z_outlier_indices_by_col = {} # store row indices for Z-score outliers
z_threshold = 3.0 # using standard threshold i.e. [3.0]

for col in features_to_test:
  values = df_cleaned[col].dropna().tolist()
  # resuing B2 tasks custom funtions for calculating mean and standard deviation
  mean_val = calc_mean(values)
  variance_val = calc_variance(values, mean_val)
  std_val = calc_std(variance_val)

  # edge case where standard deviation is 0 (i.e. all values identical)
  if std_val==0:
    outlier_count=0
    z_outlier_indices_by_col[col]=[]
  else:
    # checking for observations with absolute Z-score > 3.0
    # condition is being applied using lambda function
    outliers = df_cleaned[df_cleaned[col].dropna().apply(lambda x: abs((x - mean_val)/std_val) > z_threshold)]
    outlier_count = len(outliers)
    z_outlier_indices_by_col[col] = outliers.index.tolist()

  z_score_report.append({
     "Feature": col,
     "Mean": round(mean_val, 2),
     "Std Dev": round(std_val, 2),
     "Outliers Detected": outlier_count
  })

z_table = pd.DataFrame(z_score_report).sort_values("Outliers Detected", ascending=False)
tot_outliers_count_z_score = (z_table["Outliers Detected"]).sum() # For Task N
print(z_table)
print()

# Library Verification using SciPy zscore for F2 Task

verify_z_rows = []
for col in features_to_test:
  clean_series = df_cleaned[col].dropna()
  lib_z_scores = zscore(clean_series)
  lib_outlier_count = sum(abs(lib_z_scores) > z_threshold)

  my_row = next(r for r in z_score_report if r["Feature"] == col)

  verify_z_rows.append({
      "Feature": col,
      "Custom Outliers": my_row["Outliers Detected"],
      "Library Outliers": lib_outlier_count,
      "Counts Match": my_row["Outliers Detected"] == lib_outlier_count
  })

print(pd.DataFrame(verify_z_rows))
print()

# F3 + G - Outlier Treatment using Transformation

import numpy as np
df_outlier_transform = df_cleaned.copy()

#Apply Log(x + 1) Transformation
features = ["CashbackAmount","OrderAmountHikeFromlastYear"]

for col in features:
    new_col = col + "_LogTransform"
    transformed_values = []

    for x in df_outlier_transform[col]:

        if pd.notna(x):
            transformed_values.append(np.log(x + 1))
        else:
            transformed_values.append(x)

    df_outlier_transform[new_col] = transformed_values


# Show Original and Transformed Values
print(df_outlier_transform[["CashbackAmount","CashbackAmount_LogTransform","OrderAmountHikeFromlastYear","OrderAmountHikeFromlastYear_LogTransform"]].head(10))
print()


# H1: Min-Max Normalization for numerical values
exclude_always = ["CustomerID"]
exclude_ordinal = ["CityTier"]

df_normalized = df_cleaned.copy()
normalized_cols = []

for col in features_to_test:
    if col in exclude_always or col in exclude_ordinal:
        continue

    values = df_normalized[col].dropna().tolist()

    # FROM B2
    min_val = min(values)
    max_val = max(values)

    new_col_name = col + "_Normalized"
    normalized_cols.append(new_col_name)

    scaled_values = []
    for x in df_normalized[col]:
        if pd.isna(x):
            scaled_values.append(x)
        elif max_val == min_val:
            scaled_values.append(0.0)
        else:
            scaled_values.append((x - min_val) / (max_val - min_val))

    df_normalized[new_col_name] = scaled_values

print(df_normalized[normalized_cols].head(10))
print()


# --- H1 verification: confirm normalized values lie within [0, 1] (tabular form) ---
verification_rows = []
for col in normalized_cols:
    col_min = df_normalized[col].min()
    col_max = df_normalized[col].max()
    in_range = (col_min >= 0) and (col_max <= 1)
    verification_rows.append({
        "Feature": col,
        "Min": round(col_min, 4),
        "Max": round(col_max, 4),
        "Within [0,1]": in_range
    })

verification_table = pd.DataFrame(verification_rows)
print(verification_table)
print()


# H2: Standardization for numerical values [Feature Scaling]

df_standardized = df_cleaned.copy()
standardized_cols = []

for col in features_to_test:
    values = df_standardized[col].dropna().tolist()

    # Reusing B2 custom functions
    mean_val = calc_mean(values)
    variance_val = calc_variance(values, mean_val)
    std_val = calc_std(variance_val)

    new_col_name = col + "_Standardized"
    standardized_cols.append(new_col_name)

    scaled_values = []
    for x in df_standardized[col]:
        if pd.isna(x):
            scaled_values.append(x)
        elif std_val == 0:
            # Avoiding division by zero if all values are identical
            scaled_values.append(0.0)
        else:
            scaled_val = (x - mean_val) / std_val
            scaled_values.append(scaled_val)

    df_standardized[new_col_name] = scaled_values

print("Task H2: Standardization completed successfully from scratch!")
print(df_standardized[standardized_cols].head(10))
print()


# Library Verification using StandardScaler from scikit-learn for H2 Task
verify_std_rows = []
scaler = StandardScaler()

for col in features_to_test:

    clean_data = df_cleaned[[col]].dropna()
    lib_scaled = scaler.fit_transform(clean_data).flatten()

    # Custom standardization results
    my_scaled = df_standardized[col + "_Standardized"].dropna().values

    # Check if they closely match using atol = tolerance for floating-point precision
    matches = np.allclose(my_scaled, lib_scaled, atol=1e-5)

    verify_std_rows.append({
        "Feature": col,
        "Feature": col,
        "Custom Mean": round(df_standardized[col + "_Standardized"].mean(), 5), # Should be ~0
        "Custom Std": round(df_standardized[col + "_Standardized"].std(), 5),   # Should be ~1
        "Matches Sklearn": matches
    })

print(pd.DataFrame(verify_std_rows))
print()


# # Task I - Histogram for Tenure, WarehouseToHome and HourSpendOnApp
# histogram_features = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp']

# # Setting up a 1-row, 3-column canvas using subplot function
# fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# for i, col in enumerate(histogram_features):
#     ax = axes[i]
#     data = df_cleaned[col].dropna()

#     ax.hist(data, bins=15, color='skyblue', edgecolor='black', alpha=0.8)

#     ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
#     ax.set_xlabel(col, fontsize=10)
#     ax.set_ylabel('Frequency', fontsize=10)

#     ax.grid(True, linestyle='--', alpha=0.5)

# plt.suptitle('Univariate & Skewness Analysis: Histograms of Continuous Features', fontsize=14, fontweight='bold', y=1.03)
# plt.tight_layout()
# plt.show()
# print()

# # Task I - Box Plot
# boxplot_features = ['CashbackAmount', 'OrderCount', 'DaySinceLastOrder']

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# for i, col in enumerate(boxplot_features):
#     ax = axes[i]
#     not_churned = df_cleaned[df_cleaned["Churn"] == 0][col].dropna()
#     churned = df_cleaned[df_cleaned["Churn"] == 1][col].dropna()

#     ax.boxplot([not_churned, churned], tick_labels=["Not Churned", "Churned"])
#     ax.set_title(f'{col} by Churn Status', fontsize=12, fontweight='bold')
#     ax.set_xlabel('Churn Status', fontsize=10)
#     ax.set_ylabel(col, fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.5)

# plt.suptitle('Outlier & Spread Analysis: Box Plots by Churn Status', fontsize=14, fontweight='bold', y=1.03)
# plt.tight_layout()
# plt.show()
# print()


# # Task I - Scatter Plot of Tenure vs CashBackAmount and WarehouseToHome vs HourSpendOnApp

# import matplotlib.pyplot as plt

# # Setting up a 1-row, 2-column canvas using subplot function
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ax1 = axes[0]
# ax1.scatter(df_cleaned['Tenure'], df_cleaned['CashbackAmount'],
#             color='skyblue', alpha=0.6, edgecolors='black', s=50)
# ax1.set_title('Tenure vs. Cashback Amount', fontsize=12, fontweight='bold')
# ax1.set_xlabel('Tenure (Months)', fontsize=10)
# ax1.set_ylabel('Cashback Amount', fontsize=10)
# ax1.grid(True, linestyle='--', alpha=0.5)

# ax2 = axes[1]
# ax2.scatter(df_cleaned['WarehouseToHome'], df_cleaned['HourSpendOnApp'],
#             color='teal', alpha=0.6, edgecolors='w', s=50)
# ax2.set_title('Warehouse Distance vs. App Hours', fontsize=12, fontweight='bold')
# ax2.set_xlabel('Warehouse To Home (Km)', fontsize=10)
# ax2.set_ylabel('Hour Spend On App', fontsize=10)
# ax2.grid(True, linestyle='--', alpha=0.5)

# plt.suptitle('Bivariate Analysis: Scatter Plots', fontsize=14, fontweight='bold', y=1.03)
# plt.tight_layout()
# plt.show()
# print()


# # Task I - Bar chart for PreferredLoginDevice, PreferredPaymentMode and CityTier

# import matplotlib.pyplot as plt

# bar_features = ['PreferredLoginDevice', 'PreferredPaymentMode', 'CityTier']

# # Setting up a 1-row, 3-column canvas using subplot function
# fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# for i, col in enumerate(bar_features):
#     ax = axes[i]

#     counts = df_cleaned[col].value_counts()
#     categories = [str(cat) for cat in counts.index]
#     values = counts.values

#     ax.bar(categories, values, color=['coral', 'cornflowerblue', 'mediumseagreen'], edgecolor='black', alpha=0.8)

#     ax.set_title(f'Frequency of {col}', fontsize=12, fontweight='bold')
#     ax.set_xlabel(col, fontsize=10)
#     ax.set_ylabel('Count', fontsize=10)

#     # Rotating x-tick labels because the text is long (especially for payment modes)
#     plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

#     ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# plt.suptitle('Categorical Analysis: Bar Charts', fontsize=14, fontweight='bold', y=1.03)
# plt.tight_layout()
# plt.show()
# print()


# # Task I  - Pie Charts for Churn and Complain proportions
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # --- Pie 1: Churn proportion ---
# churn_counts = df_cleaned["Churn"].value_counts()
# churn_labels = ["Retained", "Churned"]

# axes[0].pie(churn_counts, labels=churn_labels, autopct='%1.1f%%',
#             colors=['mediumseagreen', 'indianred'], startangle=90,
#             wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
# axes[0].set_title('Proportion of Retained vs Churned Customers', fontsize=12, fontweight='bold')

# # Pie 2: Complain proportion
# complain_counts = df_cleaned["Complain"].value_counts()
# complain_labels = ["No Complaint", "Complaint Raised"]

# axes[1].pie(complain_counts, labels=complain_labels, autopct='%1.1f%%',
#             colors=['cornflowerblue', 'orange'], startangle=90,
#             wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
# axes[1].set_title('Proportion of Customers Who Raised a Complaint', fontsize=12, fontweight='bold')

# plt.suptitle('Target & Complaint Distribution', fontsize=14, fontweight='bold', y=1.02)
# plt.tight_layout()
# plt.show()
# print()


# # Task I - Correlation Heatmap of Numerical Features

# correlation_matrix = pd.DataFrame(0.0,index=features_to_test,columns=features_to_test)

# for col1 in features_to_test:
#     for col2 in features_to_test:

#         x = df_cleaned[col1].values
#         y = df_cleaned[col2].values

#         mean_x = np.mean(x)
#         mean_y = np.mean(y)

#         numerator = 0
#         denominator_x = 0
#         denominator_y = 0

#         for i in range(len(x)):
#             numerator += (x[i] - mean_x) * (y[i] - mean_y)
#             denominator_x += (x[i] - mean_x) ** 2
#             denominator_y += (y[i] - mean_y) ** 2

#         correlation = numerator / np.sqrt(
#             denominator_x * denominator_y
#         )

#         correlation_matrix.loc[col1, col2] = correlation

# correlation_matrix

# # Plot heatmap
# plt.figure(figsize=(14, 10))

# sns.heatmap(correlation_matrix,annot=True,fmt=".2f",cmap="coolwarm",linewidths=0.5,center=0)

# plt.title("Correlation Heatmap of Numerical Features",fontsize=16,fontweight="bold")

# plt.tight_layout()
# plt.show()
# print()

#  Train-Test Split (80/20)
# import random

# def train_test_split_manual(df, test_ratio=0.20, seed=42):

#     total_rows = len(df)

#     #  Get all row indices as a list
#     indices = list(df.index)

#     #  Shuffle them randomly
#     random.seed(seed)
#     random.shuffle(indices)

#     #  Calculate split point
#     test_size = int(total_rows * test_ratio)

#     # Slice the shuffled indices
#     test_indices = indices[:test_size]
#     train_indices = indices[test_size:]

#     #  Using these indices to create actual train/test dataframes
#     train_df = df.loc[train_indices]
#     test_df = df.loc[test_indices]

#     return train_df, test_df

# train_df, test_df = train_test_split_manual(df_cleaned, test_ratio=0.20, seed=42)

# # Save train and test datasets as CSV files
# train_df.to_csv("train_dataset.csv", index=False)
# test_df.to_csv("test_dataset.csv", index=False)

# print("Original dataset size:", len(df_cleaned))
# print("Training set size:", len(train_df), f"({len(train_df)/len(df_cleaned)*100:.1f}%)")
# print("Testing set size:", len(test_df), f"({len(test_df)/len(df_cleaned)*100:.1f}%)")

#  Library verification
# from sklearn.model_selection import train_test_split

# train_lib, test_lib = train_test_split(df_cleaned, test_size=0.20, random_state=42)

# print("Manual split sizes match sklearn split sizes:",
#       len(train_df) == len(train_lib) and len(test_df) == len(test_lib))

