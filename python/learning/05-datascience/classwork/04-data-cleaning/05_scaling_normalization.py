"""
Scaling and Normalization
==========================
Transforming features to similar scales for machine learning models.

Topics:
- Standardization (Z-score scaling)
- Min-Max scaling
- Robust scaling
- Normalization (L1, L2)
- When to use each method
- Fitting and transforming

Run: python 05_scaling_normalization.py
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 60)
    print("Scaling and Normalization")
    print("=" * 60)

    # 1. Understanding the need for scaling
    print("\n1. Why Scaling Matters")
    print("-" * 40)

    # Different scales
    df = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'Salary': [50000, 60000, 75000, 80000, 90000],
        'Years_Experience': [2, 5, 8, 12, 15]
    })

    print("Original data with different scales:")
    print(df)

    print("\nStatistics:")
    print(df.describe())

    print("\nNote: Features have very different ranges")
    print("  Age: ~25-45")
    print("  Salary: ~50k-90k")
    print("  Years_Experience: ~2-15")

    # 2. Standardization (Z-score scaling)
    print("\n2. Standardization (Z-Score Scaling)")
    print("-" * 40)

    print("Formula: (X - mean) / std")
    print("Result: mean=0, std=1")

    df_std = df.copy()

    # Manual standardization
    for col in df_std.columns:
        mean = df_std[col].mean()
        std = df_std[col].std()
        df_std[col] = (df_std[col] - mean) / std

    print("\nStandardized data:")
    print(df_std.round(3))

    print("\nVerify: mean should be ~0, std should be ~1")
    print(df_std.describe().round(3))

    # Using sklearn-style implementation
    print("\n3. StandardScaler Implementation")
    print("-" * 40)

    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.std_ = None

        def fit(self, X):
            self.mean_ = X.mean()
            self.std_ = X.std()
            return self

        def transform(self, X):
            return (X - self.mean_) / self.std_

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    # Apply to data
    df_test = pd.DataFrame({
        'Feature1': [10, 20, 30, 40, 50],
        'Feature2': [100, 200, 300, 400, 500]
    })

    print("Original data:")
    print(df_test)

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_test),
        columns=df_test.columns
    )

    print("\nAfter StandardScaler:")
    print(df_scaled.round(3))

    print("\nScaler parameters:")
    print(f"  Mean: {scaler.mean_.values}")
    print(f"  Std: {scaler.std_.values}")

    # 4. Min-Max scaling
    print("\n4. Min-Max Scaling")
    print("-" * 40)

    print("Formula: (X - min) / (max - min)")
    print("Result: values in range [0, 1]")

    df_minmax = df.copy()

    # Manual min-max scaling
    for col in df_minmax.columns:
        min_val = df_minmax[col].min()
        max_val = df_minmax[col].max()
        df_minmax[col] = (df_minmax[col] - min_val) / (max_val - min_val)

    print("\nMin-Max scaled data:")
    print(df_minmax.round(3))

    print("\nVerify: min should be 0, max should be 1")
    print(f"Min values: {df_minmax.min().values}")
    print(f"Max values: {df_minmax.max().values}")

    # 5. MinMaxScaler implementation
    print("\n5. MinMaxScaler Implementation")
    print("-" * 40)

    class MinMaxScaler:
        def __init__(self, feature_range=(0, 1)):
            self.feature_range = feature_range
            self.min_ = None
            self.max_ = None

        def fit(self, X):
            self.min_ = X.min()
            self.max_ = X.max()
            return self

        def transform(self, X):
            X_std = (X - self.min_) / (self.max_ - self.min_)
            X_scaled = X_std * (self.feature_range[1] - self.feature_range[0])
            X_scaled = X_scaled + self.feature_range[0]
            return X_scaled

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    # Scale to [0, 1]
    print("Scaling to [0, 1]:")
    scaler_01 = MinMaxScaler(feature_range=(0, 1))
    df_01 = pd.DataFrame(
        scaler_01.fit_transform(df_test),
        columns=df_test.columns
    )
    print(df_01.round(3))

    # Scale to [-1, 1]
    print("\nScaling to [-1, 1]:")
    scaler_11 = MinMaxScaler(feature_range=(-1, 1))
    df_11 = pd.DataFrame(
        scaler_11.fit_transform(df_test),
        columns=df_test.columns
    )
    print(df_11.round(3))

    # 6. Robust scaling
    print("\n6. Robust Scaling")
    print("-" * 40)

    print("Formula: (X - median) / IQR")
    print("Result: Less sensitive to outliers")

    # Data with outliers
    df_outliers = pd.DataFrame({
        'Value': [10, 12, 11, 13, 12, 14, 100, 11, 13, 12]
    })

    print("Data with outlier:")
    print(df_outliers.T)

    # Standard scaling (affected by outliers)
    mean = df_outliers['Value'].mean()
    std = df_outliers['Value'].std()
    df_outliers['Standard_Scaled'] = (df_outliers['Value'] - mean) / std

    # Robust scaling (less affected)
    median = df_outliers['Value'].median()
    Q1 = df_outliers['Value'].quantile(0.25)
    Q3 = df_outliers['Value'].quantile(0.75)
    IQR = Q3 - Q1
    df_outliers['Robust_Scaled'] = (df_outliers['Value'] - median) / IQR

    print("\nComparison:")
    print(df_outliers.round(3))

    print(f"\nOutlier (value=100):")
    print(f"  Standard scaled: {df_outliers.loc[6, 'Standard_Scaled']:.3f}")
    print(f"  Robust scaled: {df_outliers.loc[6, 'Robust_Scaled']:.3f}")

    # 7. Normalization (L1 and L2)
    print("\n7. Normalization (L1 and L2)")
    print("-" * 40)

    df_norm = pd.DataFrame({
        'Feature1': [3, 4, 5],
        'Feature2': [4, 3, 12]
    })

    print("Original data:")
    print(df_norm)

    # L2 normalization (Euclidean norm)
    print("\nL2 Normalization (unit vector):")
    df_l2 = df_norm.copy()
    for idx in df_l2.index:
        row = df_l2.loc[idx]
        l2_norm = np.sqrt((row ** 2).sum())
        df_l2.loc[idx] = row / l2_norm

    print(df_l2.round(4))

    # Verify unit vectors
    print("\nVerify L2 norm = 1:")
    for idx in df_l2.index:
        row = df_l2.loc[idx]
        l2_norm = np.sqrt((row ** 2).sum())
        print(f"  Row {idx}: {l2_norm:.4f}")

    # L1 normalization (Manhattan norm)
    print("\nL1 Normalization:")
    df_l1 = df_norm.copy()
    for idx in df_l1.index:
        row = df_l1.loc[idx]
        l1_norm = np.abs(row).sum()
        df_l1.loc[idx] = row / l1_norm

    print(df_l1.round(4))

    # Verify L1 norm = 1
    print("\nVerify L1 norm = 1:")
    for idx in df_l1.index:
        row = df_l1.loc[idx]
        l1_norm = np.abs(row).sum()
        print(f"  Row {idx}: {l1_norm:.4f}")

    # 8. Choosing the right scaler
    print("\n8. When to Use Each Method")
    print("-" * 40)

    comparison_data = pd.DataFrame({
        'Normal': np.random.normal(50, 10, 100),
        'Skewed': np.random.exponential(10, 100),
        'With_Outliers': np.concatenate([np.random.normal(50, 5, 95), [200, 210, 215, 220, 225]])
    })

    print("Sample data statistics:")
    print(comparison_data.describe().round(2))

    # Apply different scalers
    results = pd.DataFrame({
        'Original_Normal': [comparison_data['Normal'].mean(), comparison_data['Normal'].std()],
        'Standardized': [0, 1]
    }, index=['Mean', 'Std'])

    print("\nStandardization results:")
    print(results.round(3))

    # 9. Train-test split consideration
    print("\n9. Important: Fit on Training, Transform on Test")
    print("-" * 40)

    # Simulate train-test split
    train_data = pd.DataFrame({
        'Feature': [10, 20, 30, 40, 50, 60, 70, 80]
    })

    test_data = pd.DataFrame({
        'Feature': [25, 55, 75]
    })

    print("Training data:")
    print(train_data.T)

    print("\nTest data:")
    print(test_data.T)

    # Fit on training data
    scaler = StandardScaler()
    scaler.fit(train_data)

    print(f"\nScaler fitted on training data:")
    print(f"  Mean: {scaler.mean_.values[0]:.2f}")
    print(f"  Std: {scaler.std_.values[0]:.2f}")

    # Transform both using training statistics
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    print("\nScaled training data:")
    print(train_scaled.round(3).T)

    print("\nScaled test data (using training stats):")
    print(test_scaled.round(3).T)

    print("\nWARNING: Never fit scaler on test data!")
    print("This would cause data leakage.")

    print("\n" + "=" * 60)
    print("Summary - Scaling Methods:")
    print("  1. StandardScaler: (X - mean) / std")
    print("     Use: When features are normally distributed")
    print("  2. MinMaxScaler: (X - min) / (max - min)")
    print("     Use: When you need bounded values [0,1]")
    print("  3. RobustScaler: (X - median) / IQR")
    print("     Use: When data has outliers")
    print("  4. Normalizer: Scale to unit norm (L1 or L2)")
    print("     Use: When direction matters more than magnitude")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Compare scalers on dataset with outliers")
    print("2. Implement MaxAbsScaler (scale to [-1, 1] by max abs)")
    print("3. Scale features with different distributions")
    print("4. Demonstrate data leakage when fitting on test data")
    print("5. Implement inverse_transform for scalers")
    print("=" * 60)

if __name__ == "__main__":
    main()
