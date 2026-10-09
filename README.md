Week 1: Data Acquisition, Cleaning, and Preprocessing

Project Overview

This project demonstrates data acquisition, data cleaning, and preprocessing using Python. The Diabetes Progression Dataset provided by scikit-learn is used to demonstrate data preparation techniques.

Objectives

- Acquire a public dataset.
- Explore the dataset using Pandas.
- Identify missing values and invalid entries.
- Handle missing values using median imputation.
- Detect potential outliers using the Interquartile Range (IQR) method.
- Apply outlier capping.
- Export the cleaned dataset.

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn

Project Files

- "week1_data_cleaning_preprocessing.py" – Python preprocessing script.
- "diabetes_dirty_sample.csv" – Dataset containing simulated data-quality issues.
- "diabetes_cleaned_sample.csv" – Preprocessed dataset.
- "Week_1_Data_Acquisition_Cleaning_Preprocessing_Report.docx" – Detailed project report.

Methodology

1. Load the public dataset.
2. Inspect the dataset structure and data types.
3. Convert invalid numeric entries into missing values.
4. Fill missing numeric values using the median.
5. Detect potential outliers using the IQR method.
6. Apply outlier capping to selected features.
7. Validate and export the cleaned dataset.

Important Note

Missing values, malformed entries, and extreme values were deliberately introduced into a copy of the original dataset for educational demonstration. They are not claimed to be defects in the original public dataset.

Dataset Source

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html

Conclusion

This project demonstrates a reproducible data preprocessing workflow and explains how cleaning decisions can influence subsequent analysis and machine learning.
