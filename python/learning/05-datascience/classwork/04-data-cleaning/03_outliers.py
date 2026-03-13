"""
Outlier Detection and Handling
===============================
Identifying and handling outliers in datasets.

Topics:
- Statistical methods (Z-score, IQR)
- Visualization of outliers
- Outlier removal
- Outlier capping/winsorization
- Domain-specific considerations

Run: python 03_outliers.py
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 60)
    print("Outlier Detection and Handling")
    print("=" * 60)

    # 1. Understanding outliers
    print("\n1. What Are Outliers?")
    print("-" * 40)

    np.random.seed(42)
    # Normal data with outliers
    data = np.concatenate([
        np.random.normal(100, 10, 95),  # Normal data
        [150, 155, 45, 40, 160]  # Outliers
    ])

    df = pd.DataFrame({'Value': data})

    print(f"Data points: {len(df)}")
    print(f"Mean: {df['Value'].mean():.2f}")
    print(f"Median: {df['Value'].median():.2f}")
    print(f"Std: {df['Value'].std():.2f}")
    print(f"Min: {df['Value'].min():.2f}")
    print(f"Max: {df['Value'].max():.2f}")

    print("\nBasic statistics:")
    print(df.describe())

    # 2. Z-score method
    print("\n2. Z-Score Method")
    print("-" * 40)

    # Calculate Z-scores
    df['Z_Score'] = (df['Value'] - df['Value'].mean()) / df['Value'].std()

    print("Data with Z-scores (first 10):")
    print(df.head(10))

    # Identify outliers (|Z| > 3)
    threshold = 3
    outliers_z = df[np.abs(df['Z_Score']) > threshold]

    print(f"\nOutliers (|Z-score| > {threshold}):")
    print(outliers_z)
    print(f"Number of outliers: {len(outliers_z)}")

    # 3. IQR method
    print("\n3. Interquartile Range (IQR) Method")
    print("-" * 40)

    Q1 = df['Value'].quantile(0.25)
    Q3 = df['Value'].quantile(0.75)
    IQR = Q3 - Q1

    print(f"Q1 (25th percentile): {Q1:.2f}")
    print(f"Q3 (75th percentile): {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"\nOutlier bounds:")
    print(f"  Lower: {lower_bound:.2f}")
    print(f"  Upper: {upper_bound:.2f}")

    # Identify outliers
    outliers_iqr = df[(df['Value'] < lower_bound) | (df['Value'] > upper_bound)]

    print(f"\nOutliers by IQR method:")
    print(outliers_iqr[['Value']])
    print(f"Number of outliers: {len(outliers_iqr)}")

    # 4. Removing outliers
    print("\n4. Removing Outliers")
    print("-" * 40)

    print(f"Original data: {len(df)} points")

    # Remove using IQR method
    df_clean = df[(df['Value'] >= lower_bound) & (df['Value'] <= upper_bound)]

    print(f"After removal: {len(df_clean)} points")
    print(f"Removed: {len(df) - len(df_clean)} points")

    print("\nStatistics before removal:")
    print(f"  Mean: {df['Value'].mean():.2f}")
    print(f"  Median: {df['Value'].median():.2f}")
    print(f"  Std: {df['Value'].std():.2f}")

    print("\nStatistics after removal:")
    print(f"  Mean: {df_clean['Value'].mean():.2f}")
    print(f"  Median: {df_clean['Value'].median():.2f}")
    print(f"  Std: {df_clean['Value'].std():.2f}")

    # 5. Capping outliers (Winsorization)
    print("\n5. Capping Outliers (Winsorization)")
    print("-" * 40)

    df_capped = df.copy()

    # Cap at 5th and 95th percentiles
    lower_cap = df['Value'].quantile(0.05)
    upper_cap = df['Value'].quantile(0.95)

    print(f"Capping bounds:")
    print(f"  Lower (5th percentile): {lower_cap:.2f}")
    print(f"  Upper (95th percentile): {upper_cap:.2f}")

    # Apply capping
    df_capped['Value_Capped'] = df_capped['Value'].clip(lower_cap, upper_cap)

    print("\nBefore and after capping (extreme values):")
    extreme_idx = df_capped['Value'].nlargest(5).index.tolist() + \
                  df_capped['Value'].nsmallest(5).index.tolist()
    print(df_capped.loc[extreme_idx, ['Value', 'Value_Capped']].sort_values('Value'))

    # 6. Domain-specific considerations
    print("\n6. Domain-Specific Considerations")
    print("-" * 40)

    # Example: Salary data
    salaries = pd.DataFrame({
        'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'CEO'],
        'Salary': [60000, 65000, 58000, 62000, 61000, 500000]
    })

    print("Salary data:")
    print(salaries)

    print("\nStatistical outlier detection:")
    Q1 = salaries['Salary'].quantile(0.25)
    Q3 = salaries['Salary'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    print(f"IQR upper bound: ${upper_bound:,.0f}")

    statistical_outliers = salaries[salaries['Salary'] > upper_bound]
    print(f"\nStatistical outliers:")
    print(statistical_outliers)

    print("\nNote: CEO salary is a statistical outlier but valid!")
    print("Domain knowledge is crucial - not all outliers should be removed.")

    # 7. Outlier treatment strategies
    print("\n7. Outlier Treatment Strategies Summary")
    print("-" * 40)

    print("Strategy 1 - Remove: Delete outlier rows")
    print("  Pros: Clean data")
    print("  Cons: Loss of information")

    print("\nStrategy 2 - Cap (Winsorize): Limit to threshold values")
    print("  Pros: Retain all data points")
    print("  Cons: Distorts distribution")

    print("\nStrategy 3 - Transform: Log, sqrt to reduce impact")
    print("  Pros: Preserves relationships")
    print("  Cons: Changes interpretation")

    print("\nStrategy 4 - Flag: Keep but mark for special handling")
    print("  Pros: Flexible, no data loss")
    print("  Cons: Requires model support")

    print("\n" + "=" * 60)
    print("Summary - Outlier Detection Methods:")
    print("  1. Z-score: |Z| > 3 (assumes normal distribution)")
    print("  2. IQR: Values beyond Q1-1.5*IQR or Q3+1.5*IQR")
    print("  3. Percentile: Cap at specific percentiles (5th, 95th)")
    print("  4. Domain knowledge: Always consider context")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Detect outliers in multi-column dataset")
    print("2. Compare Z-score vs IQR methods on skewed data")
    print("3. Implement robust statistics (median, MAD)")
    print("4. Handle multivariate outliers using Mahalanobis distance")
    print("=" * 60)

if __name__ == "__main__":
    main()
