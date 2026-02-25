"""
Handling Missing Data
=====================
Detecting and handling NaN values in DataFrames.

Topics:
- Detecting missing data
- Removing missing data (dropna)
- Filling missing data (fillna)
- Imputation strategies
- Missing data visualization

Run: python 01_missing_data.py
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 60)
    print("Handling Missing Data")
    print("=" * 60)

    # 1. Detecting missing data
    print("\n1. Detecting Missing Data")
    print("-" * 40)

    # Create DataFrame with missing values
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5],
        'D': [np.nan, np.nan, np.nan, np.nan, np.nan]
    })

    print("DataFrame with missing values:")
    print(df)

    # Check for missing values
    print("\nMissing values (boolean mask):")
    print(df.isnull())

    # Count missing values
    print("\nMissing count per column:")
    print(df.isnull().sum())

    # Percentage of missing values
    print("\nMissing percentage per column:")
    print((df.isnull().sum() / len(df) * 100).round(2))

    # Total missing values
    print(f"\nTotal missing values: {df.isnull().sum().sum()}")

    # 2. Removing missing data
    print("\n2. Removing Missing Data (dropna)")
    print("-" * 40)

    print("Original DataFrame:")
    print(df)

    # Drop rows with any missing values
    df_dropped_rows = df.dropna()
    print("\nAfter dropna() - removes any row with NaN:")
    print(df_dropped_rows)

    # Drop columns with any missing values
    df_dropped_cols = df.dropna(axis=1)
    print("\nAfter dropna(axis=1) - removes any column with NaN:")
    print(df_dropped_cols)

    # Drop rows where all values are missing
    df_dropped_all = df.dropna(how='all')
    print("\nAfter dropna(how='all') - removes rows where all values are NaN:")
    print(df_dropped_all)

    # Drop rows with less than N non-null values
    df_thresh = df.dropna(thresh=3)
    print("\nAfter dropna(thresh=3) - requires at least 3 non-null values:")
    print(df_thresh)

    # 3. Filling missing data
    print("\n3. Filling Missing Data (fillna)")
    print("-" * 40)

    print("Original DataFrame:")
    print(df)

    # Fill with a scalar value
    df_filled_zero = df.fillna(0)
    print("\nFilled with 0:")
    print(df_filled_zero)

    # Fill with different values per column
    fill_values = {'A': 0, 'B': -1, 'D': 999}
    df_filled_dict = df.fillna(fill_values)
    print("\nFilled with dictionary (different values per column):")
    print(df_filled_dict)

    # Forward fill (use previous value)
    df_ffill = df.fillna(method='ffill')
    print("\nForward fill (ffill) - use previous value:")
    print(df_ffill)

    # Backward fill (use next value)
    df_bfill = df.fillna(method='bfill')
    print("\nBackward fill (bfill) - use next value:")
    print(df_bfill)

    # 4. Statistical imputation
    print("\n4. Statistical Imputation")
    print("-" * 40)

    # Create sample data
    np.random.seed(42)
    df_stats = pd.DataFrame({
        'Age': [25, np.nan, 35, 30, np.nan, 28, 32, np.nan, 27],
        'Salary': [50000, 60000, np.nan, 55000, 70000, np.nan, 65000, 58000, np.nan],
        'Score': [85, 90, 78, np.nan, 88, 92, np.nan, 79, 86]
    })

    print("Original data:")
    print(df_stats)
    print(f"\nMissing values: {df_stats.isnull().sum().sum()}")

    # Fill with mean
    df_mean = df_stats.copy()
    df_mean['Age'].fillna(df_mean['Age'].mean(), inplace=True)
    df_mean['Salary'].fillna(df_mean['Salary'].mean(), inplace=True)
    df_mean['Score'].fillna(df_mean['Score'].mean(), inplace=True)

    print("\nFilled with mean:")
    print(df_mean)
    print("\nMeans used:")
    print(f"  Age: {df_stats['Age'].mean():.2f}")
    print(f"  Salary: {df_stats['Salary'].mean():.2f}")
    print(f"  Score: {df_stats['Score'].mean():.2f}")

    # Fill with median
    df_median = df_stats.copy()
    for col in df_median.columns:
        df_median[col].fillna(df_median[col].median(), inplace=True)

    print("\nFilled with median:")
    print(df_median)

    # Fill with mode (most frequent value)
    df_mode = df_stats.copy()
    for col in df_mode.columns:
        mode_value = df_mode[col].mode()
        if len(mode_value) > 0:
            df_mode[col].fillna(mode_value[0], inplace=True)

    print("\nFilled with mode:")
    print(df_mode)

    # 5. Interpolation
    print("\n5. Interpolation")
    print("-" * 40)

    # Time series data
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    ts = pd.Series([100, np.nan, np.nan, 103, 104, np.nan, 106, 107, np.nan, 110],
                   index=dates)

    print("Original time series:")
    print(ts)

    # Linear interpolation
    ts_interp = ts.interpolate(method='linear')
    print("\nLinear interpolation:")
    print(ts_interp)

    # Polynomial interpolation
    ts_poly = ts.interpolate(method='polynomial', order=2)
    print("\nPolynomial interpolation (order 2):")
    print(ts_poly.round(2))

    # 6. Group-based imputation
    print("\n6. Group-Based Imputation")
    print("-" * 40)

    # Create data with groups
    df_groups = pd.DataFrame({
        'Department': ['IT', 'IT', 'HR', 'HR', 'Finance', 'Finance', 'IT', 'HR'],
        'Salary': [70000, np.nan, 50000, np.nan, 80000, 75000, np.nan, 52000]
    })

    print("Original data:")
    print(df_groups)

    # Fill missing values with group mean
    df_groups['Salary_Filled'] = df_groups.groupby('Department')['Salary'].transform(
        lambda x: x.fillna(x.mean())
    )

    print("\nFilled with department mean:")
    print(df_groups)

    print("\nDepartment means:")
    print(df_groups.groupby('Department')['Salary'].mean())

    # 7. Creating missing value indicators
    print("\n7. Missing Value Indicators")
    print("-" * 40)

    df_indicator = pd.DataFrame({
        'Value': [1, 2, np.nan, 4, np.nan, 6]
    })

    print("Original data:")
    print(df_indicator)

    # Add indicator column
    df_indicator['Value_Missing'] = df_indicator['Value'].isnull().astype(int)

    # Fill missing values
    df_indicator['Value_Filled'] = df_indicator['Value'].fillna(df_indicator['Value'].mean())

    print("\nWith missing indicator and filled values:")
    print(df_indicator)

    print("\n" + "=" * 60)
    print("Summary - Missing Data Strategies:")
    print("  1. dropna(): Remove missing values")
    print("  2. fillna(value): Fill with constant")
    print("  3. fillna(method='ffill'): Forward fill")
    print("  4. fillna(mean/median/mode): Statistical imputation")
    print("  5. interpolate(): Interpolation for time series")
    print("  6. Group-based filling: Use group statistics")
    print("  7. Missing indicators: Flag missing values before filling")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create DataFrame with 20% missing values, analyze patterns")
    print("2. Compare mean vs median imputation on skewed data")
    print("3. Implement KNN imputation using similar rows")
    print("4. Create visualization showing missing data patterns")
    print("=" * 60)

if __name__ == "__main__":
    main()
