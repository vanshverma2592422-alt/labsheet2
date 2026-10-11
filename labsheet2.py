import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder

DATA_FILE = "data.csv"

def load_dataset():
    return pd.read_csv(DATA_FILE)

def numeric_columns(data):
    return data.select_dtypes(include=np.number).columns.tolist()


# Program 1: Load dataset and identify missing values
df = load_dataset()
missing_values = df.isnull().sum()

# Program 2: Display percentage of missing values in every feature
df = load_dataset()
missing_percentage = df.isnull().mean() * 100

# Program 3: Remove rows containing missing values
df = load_dataset()
df_without_missing_rows = df.dropna()

# Program 4: Remove columns having more than 50% missing values
df = load_dataset()
missing_percentage = df.isnull().mean() * 100
df_reduced = df.loc[:, missing_percentage <= 50].copy()

# Program 5: Replace missing numerical values using the mean
df = load_dataset()
df_mean = df.copy()
for col in numeric_columns(df_mean):
    df_mean[col] = df_mean[col].fillna(df_mean[col].mean())

# Program 6: Replace missing numerical values using the median
df = load_dataset()
df_median = df.copy()
for col in numeric_columns(df_median):
    df_median[col] = df_median[col].fillna(df_median[col].median())

# Program 7: Replace missing categorical values using the mode
df = load_dataset()
df_mode = df.copy()
for col in df_mode.select_dtypes(include=["object", "category", "bool"]).columns:
    mode_values = df_mode[col].mode(dropna=True)
    if not mode_values.empty:
        df_mode[col] = df_mode[col].fillna(mode_values.iloc[0])

# Program 8: Fill missing values using forward fill
df = load_dataset()
df_forward_fill = df.ffill()

# Program 9: Fill missing values using backward fill
df = load_dataset()
df_backward_fill = df.bfill()

# Program 10: Compare dataset before and after handling missing values
df_before = load_dataset()
df_after = df_before.copy()
for col in numeric_columns(df_after):
    df_after[col] = df_after[col].fillna(df_after[col].median())
for col in df_after.select_dtypes(include=["object", "category", "bool"]).columns:
    mode_values = df_after[col].mode(dropna=True)
    if not mode_values.empty:
        df_after[col] = df_after[col].fillna(mode_values.iloc[0])
missing_comparison = pd.DataFrame({
    "Before": df_before.isnull().sum(),
    "After": df_after.isnull().sum()
})

# Program 11: Detect outliers using the IQR method
df = load_dataset()
iqr_outliers = {}
for col in numeric_columns(df):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    iqr_outliers[col] = df[col].lt(q1 - 1.5 * iqr) | df[col].gt(q3 + 1.5 * iqr)

# Program 12: Detect outliers using the Z-score method
df = load_dataset()
num_cols = numeric_columns(df)
z_scores = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std().replace(0, np.nan)
z_score_outliers = z_scores.abs() > 3

# Program 13: Visualize outliers using a Box Plot
df = load_dataset()
num_cols = numeric_columns(df)
if num_cols:
    sns.boxplot(data=df[num_cols])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Program 14: Visualize outliers using a Scatter Plot
df = load_dataset()
num_cols = numeric_columns(df)
if len(num_cols) >= 2:
    plt.scatter(df[num_cols[0]], df[num_cols[1]])
    plt.xlabel(num_cols[0])
    plt.ylabel(num_cols[1])
    plt.tight_layout()
    plt.show()

# Program 15: Remove outliers using the IQR method
df = load_dataset()
df_iqr_filtered = df.copy()
for col in numeric_columns(df_iqr_filtered):
    q1, q3 = df_iqr_filtered[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df_iqr_filtered = df_iqr_filtered[
        df_iqr_filtered[col].between(low, high) | df_iqr_filtered[col].isna()
    ]

# Program 16: Replace outliers with the median value
df = load_dataset()
df_outliers_median = df.copy()
for col in numeric_columns(df_outliers_median):
    q1, q3 = df_outliers_median[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = df_outliers_median[col].lt(low) | df_outliers_median[col].gt(high)
    df_outliers_median.loc[mask, col] = df_outliers_median[col].median()

# Program 17: Cap outliers using percentile-based capping
df = load_dataset()
df_capped = df.copy()
for col in numeric_columns(df_capped):
    low, high = df_capped[col].quantile([0.01, 0.99])
    df_capped[col] = df_capped[col].clip(low, high)

# Program 18: Compare dataset before and after outlier treatment
df_before = load_dataset()
df_after = df_before.copy()
num_cols = numeric_columns(df_after)
for col in num_cols:
    q1, q3 = df_after[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df_after[col] = df_after[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
outlier_comparison = pd.DataFrame({
    "Before_Mean": df_before[num_cols].mean(),
    "After_Mean": df_after[num_cols].mean(),
    "Before_Std": df_before[num_cols].std(),
    "After_Std": df_after[num_cols].std()
})

# Program 19: Apply Min-Max Normalization
df = load_dataset()
num_cols = numeric_columns(df)
df_minmax = df.copy()
if num_cols:
    df_minmax[num_cols] = MinMaxScaler().fit_transform(df_minmax[num_cols])

# Program 20: Apply Standardization (Z-score Scaling)
df = load_dataset()
num_cols = numeric_columns(df)
df_standardized = df.copy()
if num_cols:
    df_standardized[num_cols] = StandardScaler().fit_transform(df_standardized[num_cols])

# Program 21: Apply Robust Scaling
df = load_dataset()
num_cols = numeric_columns(df)
df_robust = df.copy()
if num_cols:
    df_robust[num_cols] = RobustScaler().fit_transform(df_robust[num_cols])

# Program 22: Apply Max Absolute Scaling
df = load_dataset()
num_cols = numeric_columns(df)
df_maxabs = df.copy()
if num_cols:
    df_maxabs[num_cols] = MaxAbsScaler().fit_transform(df_maxabs[num_cols])

# Program 23: Compare original and normalized datasets
df_original = load_dataset()
num_cols = numeric_columns(df_original)
df_normalized = df_original.copy()
if num_cols:
    df_normalized[num_cols] = MinMaxScaler().fit_transform(df_normalized[num_cols])
original_summary = df_original[num_cols].describe()
normalized_summary = df_normalized[num_cols].describe()

# Program 24: Visualize normalization using histograms
df = load_dataset()
num_cols = numeric_columns(df)
if num_cols:
    col = num_cols[0]
    original_values = df[col].dropna()
    normalized_values = MinMaxScaler().fit_transform(original_values.to_frame()).ravel()
    plt.hist(original_values, bins=20, alpha=0.5, label="Original")
    plt.hist(normalized_values, bins=20, alpha=0.5, label="Normalized")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Program 25: Visualize scaling using box plots
df = load_dataset()
num_cols = numeric_columns(df)
if num_cols:
    complete_data = df[num_cols].dropna()
    scaled_data = StandardScaler().fit_transform(complete_data)
    scaled_df = pd.DataFrame(scaled_data, columns=num_cols)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=complete_data, ax=axes[0])
    sns.boxplot(data=scaled_df, ax=axes[1])
    axes[0].set_title("Original Data")
    axes[1].set_title("Standardized Data")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

# Program 26: Compare different scaling techniques
df = load_dataset()
num_cols = numeric_columns(df)
scaling_comparison = {}
if num_cols:
    values = df[num_cols].dropna()
    for name, scaler in {
        "MinMax": MinMaxScaler(),
        "Standard": StandardScaler(),
        "Robust": RobustScaler(),
        "MaxAbs": MaxAbsScaler()
    }.items():
        scaling_comparison[name] = pd.DataFrame(
            scaler.fit_transform(values), columns=num_cols, index=values.index
        )

# Program 27: Encode categorical variables using Label Encoding
df = load_dataset()
df_label_encoded = df.copy()
for col in df_label_encoded.select_dtypes(include=["object", "category", "bool"]).columns:
    mask = df_label_encoded[col].notna()
    encoder = LabelEncoder()
    df_label_encoded.loc[mask, col] = encoder.fit_transform(
        df_label_encoded.loc[mask, col].astype(str)
    )

# Program 28: Encode categorical variables using One-Hot Encoding
df = load_dataset()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
df_one_hot = pd.get_dummies(df, columns=cat_cols)

# Program 29: Perform Binary Encoding on categorical data
df = load_dataset()
df_binary = df.copy()
cat_cols = df_binary.select_dtypes(include=["object", "category", "bool"]).columns
for col in cat_cols:
    categories = df_binary[col].dropna().astype(str).unique()
    mapping = {value: i + 1 for i, value in enumerate(categories)}
    codes = df_binary[col].astype(str).map(mapping).fillna(0).astype(int)
    bits = max(1, max(mapping.values(), default=0).bit_length())
    for bit in range(bits):
        df_binary[f"{col}_bit_{bit}"] = (codes >> bit) & 1
    df_binary.drop(columns=[col], inplace=True)

# Program 30: Create a new feature by combining two existing columns
df = load_dataset()
if len(df.columns) >= 2:
    col1, col2 = df.columns[:2]
    df["Combined_Feature"] = df[col1].fillna("").astype(str) + "_" + df[col2].fillna("").astype(str)

# Program 31: Extract year, month, and day from a date column
df = load_dataset()
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

# Program 32: Create a new feature using mathematical transformations
df = load_dataset()
num_cols = numeric_columns(df)
if num_cols:
    col = num_cols[0]
    df[f"{col}_Squared"] = df[col] ** 2

# Program 33: Apply Log Transformation to skewed data
df = load_dataset()
num_cols = numeric_columns(df)
if num_cols:
    col = num_cols[0]
    shift = max(0, -df[col].min()) if df[col].notna().any() else 0
    df[f"{col}_Log"] = np.log1p(df[col] + shift)

# Program 34: Perform Feature Selection using correlation analysis
df = load_dataset()
numeric_df = df.select_dtypes(include=np.number)
correlation_matrix = numeric_df.corr()
correlation_pairs = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
).stack()
selected_correlations = correlation_pairs[correlation_pairs.abs() >= 0.5]

# Program 35: Create a final preprocessed dataset ready for Machine Learning
df = load_dataset()
for col in numeric_columns(df):
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
    modes = df[col].mode(dropna=True)
    if not modes.empty:
        df[col] = df[col].fillna(modes.iloc[0])
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
df_preprocessed = pd.get_dummies(df, columns=cat_cols)
num_cols = numeric_columns(df_preprocessed)
if num_cols:
    df_preprocessed[num_cols] = StandardScaler().fit_transform(df_preprocessed[num_cols])
df_preprocessed.to_csv("preprocessed_data.csv", index=False)
