"""
Descriptive Statistics
======================
Measures of central tendency, dispersion, and distribution shape.

Topics:
- Mean, median, mode
- Variance and standard deviation
- Quantiles and percentiles
- Range and IQR
- Skewness and kurtosis
- Summary statistics

Run: python 01_descriptive_stats.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Descriptive Statistics")
    print("=" * 60)

    # Create sample dataset
    np.random.seed(42)
    data = np.array([23, 45, 67, 34, 56, 78, 90, 12, 45, 67,
                     89, 23, 56, 78, 45, 34, 67, 89, 23, 45])

    # 1. Measures of Central Tendency
    print("\n1. Measures of Central Tendency")
    print("-" * 40)

    mean = np.mean(data)
    median = np.median(data)
    mode_result = stats.mode(data, keepdims=True)
    mode = mode_result.mode[0]

    print(f"Dataset: {data[:10]}... (n={len(data)})")
    print(f"\nMean (average): {mean:.2f}")
    print(f"  Sum of all values / count")
    print(f"  ({data.sum()}) / {len(data)} = {mean:.2f}")

    print(f"\nMedian (middle value): {median:.2f}")
    print(f"  50th percentile when sorted")
    sorted_data = np.sort(data)
    print(f"  Sorted data middle: {sorted_data[len(sorted_data)//2-1:len(sorted_data)//2+1]}")

    print(f"\nMode (most frequent): {mode}")
    print(f"  Value appearing {mode_result.count[0]} times")

    # 2. Measures of Dispersion
    print("\n2. Measures of Dispersion (Spread)")
    print("-" * 40)

    variance = np.var(data, ddof=1)  # Sample variance
    std_dev = np.std(data, ddof=1)   # Sample standard deviation
    data_range = np.ptp(data)        # Range (peak to peak)

    print(f"Range (max - min): {data_range}")
    print(f"  Min: {data.min()}, Max: {data.max()}")
    print(f"  {data.max()} - {data.min()} = {data_range}")

    print(f"\nVariance: {variance:.2f}")
    print(f"  Average squared deviation from mean")
    deviations = (data - mean) ** 2
    print(f"  Sum of squared deviations: {deviations.sum():.2f}")
    print(f"  Variance = {deviations.sum():.2f} / {len(data)-1} = {variance:.2f}")

    print(f"\nStandard Deviation: {std_dev:.2f}")
    print(f"  Square root of variance")
    print(f"  sqrt({variance:.2f}) = {std_dev:.2f}")
    print(f"  About {(std_dev/mean*100):.1f}% of the mean")

    # Coefficient of variation
    cv = (std_dev / mean) * 100
    print(f"\nCoefficient of Variation: {cv:.2f}%")
    print(f"  (Std Dev / Mean) * 100")
    print(f"  Measures relative variability")

    # 3. Quantiles and Percentiles
    print("\n3. Quantiles and Percentiles")
    print("-" * 40)

    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)  # Same as median
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    print("Quartiles (divide data into 4 equal parts):")
    print(f"  Q1 (25th percentile): {q1:.2f}")
    print(f"  Q2 (50th percentile/median): {q2:.2f}")
    print(f"  Q3 (75th percentile): {q3:.2f}")

    print(f"\nInterquartile Range (IQR): {iqr:.2f}")
    print(f"  Q3 - Q1 = {q3:.2f} - {q1:.2f} = {iqr:.2f}")
    print(f"  Contains middle 50% of data")

    # Additional percentiles
    print("\nOther percentiles:")
    for p in [10, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(data, p):.2f}")

    # Outlier detection using IQR
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = data[(data < lower_fence) | (data > upper_fence)]

    print(f"\nOutlier detection (IQR method):")
    print(f"  Lower fence: {lower_fence:.2f}")
    print(f"  Upper fence: {upper_fence:.2f}")
    print(f"  Number of outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Outliers: {outliers}")

    # 4. Shape of Distribution
    print("\n4. Shape of Distribution")
    print("-" * 40)

    skewness = stats.skew(data)
    kurt = stats.kurtosis(data)

    print(f"Skewness: {skewness:.3f}")
    if abs(skewness) < 0.5:
        skew_type = "approximately symmetric"
    elif skewness > 0:
        skew_type = "right-skewed (positive skew)"
    else:
        skew_type = "left-skewed (negative skew)"
    print(f"  {skew_type}")
    print(f"  0 = symmetric, >0 = right tail, <0 = left tail")

    print(f"\nKurtosis: {kurt:.3f}")
    if kurt > 0:
        kurt_type = "leptokurtic (heavy tails)"
    elif kurt < 0:
        kurt_type = "platykurtic (light tails)"
    else:
        kurt_type = "mesokurtic (normal)"
    print(f"  {kurt_type}")
    print(f"  0 = normal, >0 = heavy tails, <0 = light tails")

    # 5. Five Number Summary
    print("\n5. Five Number Summary")
    print("-" * 40)

    print("Classic five numbers:")
    print(f"  Minimum: {data.min()}")
    print(f"  Q1: {q1:.2f}")
    print(f"  Median: {median:.2f}")
    print(f"  Q3: {q3:.2f}")
    print(f"  Maximum: {data.max()}")

    # 6. Using Pandas for Descriptive Stats
    print("\n6. Pandas Descriptive Statistics")
    print("-" * 40)

    df = pd.DataFrame({
        'values': data
    })

    print("Complete statistical summary:")
    print(df.describe())

    # Additional pandas stats
    print(f"\nAdditional statistics:")
    print(f"  Sum: {df['values'].sum()}")
    print(f"  Count: {df['values'].count()}")
    print(f"  Median: {df['values'].median():.2f}")
    print(f"  MAD (Mean Absolute Deviation): {(abs(df['values'] - df['values'].mean())).mean():.2f}")

    # 7. Multiple Variables Example
    print("\n7. Descriptive Stats for Multiple Variables")
    print("-" * 40)

    np.random.seed(42)
    df_multi = pd.DataFrame({
        'Age': np.random.randint(20, 65, 50),
        'Salary': np.random.randint(30000, 120000, 50),
        'Experience': np.random.randint(0, 30, 50)
    })

    print("Dataset preview:")
    print(df_multi.head())

    print("\nSummary statistics for all variables:")
    print(df_multi.describe())

    print("\nCorrelation between variables:")
    print(df_multi.corr())

    # 8. Visualization
    print("\n8. Visualizing Descriptive Statistics")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram with stats
    axes[0, 0].hist(data, bins=10, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    axes[0, 0].axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Histogram with Mean and Median')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot
    axes[0, 1].boxplot(data, vert=True)
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Box Plot (Five Number Summary)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Multiple box plots
    axes[1, 0].boxplot([df_multi['Age'], df_multi['Experience']],
                       labels=['Age', 'Experience'])
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Multiple Variables Box Plot')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Distribution plot
    axes[1, 1].hist(data, bins=10, density=True, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Distribution with Spread Indicators')
    axes[1, 1].axvline(mean, color='red', linestyle='--', label='Mean')
    axes[1, 1].axvline(mean + std_dev, color='orange', linestyle=':', label='±1 SD')
    axes[1, 1].axvline(mean - std_dev, color='orange', linestyle=':')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/descriptive_stats.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("Created visualizations:")
    print("  - Histogram with mean and median")
    print("  - Box plot showing five number summary")
    print("  - Multiple variables comparison")
    print("  - Distribution with spread indicators")
    print("  Saved to: /tmp/descriptive_stats.png")

    print("\n" + "=" * 60)
    print("Summary:")
    print("Descriptive statistics summarize and describe data features")
    print("- Central tendency: where data clusters (mean, median, mode)")
    print("- Dispersion: how spread out data is (std, variance, range)")
    print("- Shape: symmetry and tail behavior (skewness, kurtosis)")
    print("- Position: relative standing (percentiles, quartiles)")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Calculate descriptive stats for [10, 20, 30, 40, 50, 60]")
    print("2. Find mean, median, mode of: [5, 5, 6, 7, 8, 8, 8, 9, 10]")
    print("3. Calculate variance and standard deviation manually")
    print("4. Identify outliers in: [10, 12, 15, 18, 100, 22, 25]")
    print("5. Create box plot comparing two datasets")
    print("=" * 60)

if __name__ == "__main__":
    main()
