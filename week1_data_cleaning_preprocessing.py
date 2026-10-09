"""
Week 1 Task: Data Acquisition, Cleaning, and Preprocessing
Dataset: scikit-learn Diabetes Progression dataset

Note:
The "SIMULATED ISSUES" section intentionally inserts missing values, malformed text,
and extreme values to demonstrate cleaning. These are not claims about defects in
the original public dataset. Remove that section when cleaning an authentic file.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

# 1. Acquire the public dataset
dataset = load_diabetes(as_frame=True)
df = dataset.frame.copy()
df["record_id"] = range(1, len(df) + 1)

# Preserve a raw-style copy for auditability
df.to_csv("diabetes_raw_sample.csv", index=False)

# 2. SIMULATED ISSUES FOR THIS EDUCATIONAL DEMONSTRATION ONLY
rng = np.random.default_rng(42)
for col in ["bmi", "bp", "s5", "target"]:
    idx = rng.choice(df.index, 5, replace=False)
    df.loc[idx, col] = np.nan

df["bmi"] = df["bmi"].astype(object)
df.loc[rng.choice(df.index, 2, replace=False), "bmi"] = "unknown"
df["bp"] = df["bp"].astype(object)
df.loc[rng.choice(df.index, 2, replace=False), "bp"] = "invalid"

extreme_idx = rng.choice(df.index, 4, replace=False)
df.loc[extreme_idx[:2], "bmi"] = 9.99
df.loc[extreme_idx[2:], "bp"] = -9.99

df.to_csv("diabetes_dirty_sample.csv", index=False)

# 3. Initial inspection and numeric type conversion
df = pd.read_csv("diabetes_dirty_sample.csv")
print("Initial shape:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values before conversion:\n", df.isna().sum())

numeric_cols = [col for col in df.columns if col != "record_id"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Missing values after conversion:\n", df.isna().sum())

# 4. Duplicate check excludes the generated identifier
duplicate_count = df.drop(columns=["record_id"]).duplicated().sum()
print("Duplicate content rows:", int(duplicate_count))

# 5. Outlier detection with the IQR rule
outlier_cols = ["bmi", "bp", "s5", "target"]
for col in outlier_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flags = df[col].lt(lower) | df[col].gt(upper)
    print(f"{col}: {int(flags.sum())} potential outliers")

# 6. Median imputation
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 7. IQR capping on selected fields
for col in outlier_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[col] = df[col].clip(lower=lower, upper=upper)

# 8. Validate and export
print("Final shape:", df.shape)
print("Missing cells after cleaning:", int(df.isna().sum().sum()))
print(df.describe().T)
df.to_csv("diabetes_cleaned_sample.csv", index=False)
print("Saved diabetes_cleaned_sample.csv")
