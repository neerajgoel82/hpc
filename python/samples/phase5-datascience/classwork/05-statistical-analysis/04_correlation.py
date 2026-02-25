"""
Correlation Analysis
====================
Measuring relationships between variables.

Topics:
- Pearson correlation (linear relationships)
- Spearman correlation (monotonic relationships)
- Correlation matrices
- Scatter plots with correlation
- Significance testing
- Correlation vs causation

Run: python 04_correlation.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 60)
    print("Correlation Analysis")
    print("=" * 60)

    # 1. Pearson Correlation
    print("\n1. Pearson Correlation Coefficient")
    print("-" * 40)

    print("Pearson correlation (r):")
    print("  - Measures linear relationship between two variables")
    print("  - Range: -1 to +1")
    print("  - r = +1: Perfect positive linear relationship")
    print("  - r = 0: No linear relationship")
    print("  - r = -1: Perfect negative linear relationship")
    print("  - Assumes: Linear relationship, continuous data, no outliers")

    # Example: Study hours vs test scores
    np.random.seed(42)
    study_hours = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    test_scores = 50 + 5 * study_hours + np.random.normal(0, 5, len(study_hours))

    print(f"\nExample: Study hours vs test scores")
    print(f"Study hours: {study_hours}")
    print(f"Test scores: {test_scores.astype(int)}")

    # Calculate Pearson correlation
    r_pearson, p_value = stats.pearsonr(study_hours, test_scores)

    print(f"\nPearson correlation coefficient: {r_pearson:.4f}")
    print(f"p-value: {p_value:.4f}")

    # Interpret strength
    if abs(r_pearson) >= 0.9:
        strength = "very strong"
    elif abs(r_pearson) >= 0.7:
        strength = "strong"
    elif abs(r_pearson) >= 0.5:
        strength = "moderate"
    elif abs(r_pearson) >= 0.3:
        strength = "weak"
    else:
        strength = "very weak"

    direction = "positive" if r_pearson > 0 else "negative"
    print(f"\nInterpretation: {strength} {direction} linear relationship")
    print(f"R² (coefficient of determination): {r_pearson**2:.4f}")
    print(f"  {r_pearson**2*100:.1f}% of variance in scores explained by study hours")

    # 2. Spearman Correlation
    print("\n2. Spearman Rank Correlation")
    print("-" * 40)

    print("Spearman correlation (ρ):")
    print("  - Measures monotonic relationship (not just linear)")
    print("  - Based on ranks, not actual values")
    print("  - More robust to outliers")
    print("  - Works with ordinal data")

    # Example with non-linear relationship
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])  # x squared

    print(f"\nExample: x vs x² (non-linear but monotonic)")
    print(f"x: {x}")
    print(f"y (x²): {y}")

    # Calculate both correlations
    r_pearson_nl, _ = stats.pearsonr(x, y)
    r_spearman, p_value_sp = stats.spearmanr(x, y)

    print(f"\nPearson correlation: {r_pearson_nl:.4f}")
    print(f"  (lower because relationship is non-linear)")
    print(f"Spearman correlation: {r_spearman:.4f}")
    print(f"  (perfect because relationship is perfectly monotonic)")
    print(f"  p-value: {p_value_sp:.4f}")

    # 3. Correlation with Outliers
    print("\n3. Effect of Outliers on Correlation")
    print("-" * 40)

    # Data with outlier
    x_clean = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_clean = np.array([2, 4, 5, 4, 5, 7, 8, 9, 9, 10])
    x_outlier = np.append(x_clean, 15)
    y_outlier = np.append(y_clean, 2)

    print("Dataset without outlier:")
    r_clean_p, _ = stats.pearsonr(x_clean, y_clean)
    r_clean_s, _ = stats.spearmanr(x_clean, y_clean)
    print(f"  Pearson: {r_clean_p:.4f}")
    print(f"  Spearman: {r_clean_s:.4f}")

    print("\nDataset with outlier (15, 2):")
    r_outlier_p, _ = stats.pearsonr(x_outlier, y_outlier)
    r_outlier_s, _ = stats.spearmanr(x_outlier, y_outlier)
    print(f"  Pearson: {r_outlier_p:.4f} (changed by {abs(r_outlier_p - r_clean_p):.4f})")
    print(f"  Spearman: {r_outlier_s:.4f} (changed by {abs(r_outlier_s - r_clean_s):.4f})")
    print(f"\nSpearman is more robust to outliers")

    # 4. Correlation Matrix
    print("\n4. Correlation Matrix (Multiple Variables)")
    print("-" * 40)

    # Create dataset with multiple variables
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        'Age': np.random.randint(20, 65, n),
        'Income': np.random.randint(30000, 120000, n),
        'Education_Years': np.random.randint(12, 20, n),
        'Job_Satisfaction': np.random.randint(1, 11, n)
    })

    # Add some correlations
    df['Income'] = df['Age'] * 1000 + df['Education_Years'] * 2000 + np.random.normal(0, 10000, n)
    df['Job_Satisfaction'] = (df['Income'] / 10000 +
                              np.random.normal(0, 2, n)).clip(1, 10).astype(int)

    print("Dataset preview:")
    print(df.head())

    print("\nPearson correlation matrix:")
    corr_matrix = df.corr(method='pearson')
    print(corr_matrix.round(3))

    # Identify strong correlations
    print("\nStrong correlations (|r| > 0.5):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r_val = corr_matrix.iloc[i, j]
            if abs(r_val) > 0.5:
                print(f"  {corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {r_val:.3f}")

    # 5. Testing Correlation Significance
    print("\n5. Testing Correlation Significance")
    print("-" * 40)

    var1 = df['Age'].values
    var2 = df['Income'].values

    r, p_val = stats.pearsonr(var1, var2)

    print(f"Correlation between Age and Income:")
    print(f"  r = {r:.4f}")
    print(f"  p-value = {p_val:.4f}")

    alpha = 0.05
    print(f"\nHypothesis test (α={alpha}):")
    print(f"  H0: No correlation (ρ = 0)")
    print(f"  H1: Correlation exists (ρ ≠ 0)")

    if p_val < alpha:
        print(f"  Result: p-value < α, reject H0")
        print(f"  Conclusion: Correlation is statistically significant")
    else:
        print(f"  Result: p-value >= α, fail to reject H0")
        print(f"  Conclusion: Correlation is not statistically significant")

    # 6. Correlation vs Causation
    print("\n6. Correlation vs Causation")
    print("-" * 40)

    print("Important distinctions:")
    print("  - Correlation: Two variables move together")
    print("  - Causation: One variable causes change in another")
    print("\nCorrelation does NOT imply causation!")
    print("\nPossible explanations for correlation:")
    print("  1. A causes B")
    print("  2. B causes A")
    print("  3. C causes both A and B (confounding variable)")
    print("  4. Coincidence/spurious correlation")

    print("\nExample scenarios:")
    print("  Ice cream sales ↔ Drowning incidents")
    print("    Correlation: Yes (both increase in summer)")
    print("    Causation: No (temperature is confounding variable)")

    # 7. Partial Correlation
    print("\n7. Interpreting Correlation Strength")
    print("-" * 40)

    print("General guidelines for |r|:")
    print("  0.00 - 0.19: Very weak")
    print("  0.20 - 0.39: Weak")
    print("  0.40 - 0.59: Moderate")
    print("  0.60 - 0.79: Strong")
    print("  0.80 - 1.00: Very strong")

    print("\nContext matters!")
    print("  - In physics: r=0.95 might be expected")
    print("  - In social sciences: r=0.50 can be very meaningful")
    print("  - Small correlations with large samples can be significant")

    # 8. Visualization
    print("\n8. Visualizing Correlations")
    print("-" * 40)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Strong positive correlation
    ax1 = fig.add_subplot(gs[0, 0])
    x1 = np.random.randn(50)
    y1 = x1 + np.random.randn(50) * 0.3
    r1, _ = stats.pearsonr(x1, y1)
    ax1.scatter(x1, y1, alpha=0.6)
    ax1.plot(np.unique(x1), np.poly1d(np.polyfit(x1, y1, 1))(np.unique(x1)), 'r--')
    ax1.set_title(f'Strong Positive (r={r1:.2f})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)

    # No correlation
    ax2 = fig.add_subplot(gs[0, 1])
    x2 = np.random.randn(50)
    y2 = np.random.randn(50)
    r2, _ = stats.pearsonr(x2, y2)
    ax2.scatter(x2, y2, alpha=0.6)
    ax2.set_title(f'No Correlation (r={r2:.2f})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)

    # Strong negative correlation
    ax3 = fig.add_subplot(gs[0, 2])
    x3 = np.random.randn(50)
    y3 = -x3 + np.random.randn(50) * 0.3
    r3, _ = stats.pearsonr(x3, y3)
    ax3.scatter(x3, y3, alpha=0.6)
    ax3.plot(np.unique(x3), np.poly1d(np.polyfit(x3, y3, 1))(np.unique(x3)), 'r--')
    ax3.set_title(f'Strong Negative (r={r3:.2f})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.grid(True, alpha=0.3)

    # Study hours vs scores
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(study_hours, test_scores, s=100, alpha=0.6)
    ax4.plot(np.unique(study_hours),
             np.poly1d(np.polyfit(study_hours, test_scores, 1))(np.unique(study_hours)),
             'r--', linewidth=2, label=f'r={r_pearson:.3f}')
    ax4.set_xlabel('Study Hours')
    ax4.set_ylabel('Test Score')
    ax4.set_title('Study Hours vs Test Scores')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Non-linear relationship
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(x, y, s=100, alpha=0.6)
    ax5.plot(x, y, 'r--', alpha=0.5)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y (X²)')
    ax5.set_title(f'Non-linear: Pearson={r_pearson_nl:.2f}, Spearman={r_spearman:.2f}')
    ax5.grid(True, alpha=0.3)

    # Outlier effect
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(x_clean, y_clean, s=100, alpha=0.6, label='Data')
    ax6.scatter([15], [2], s=200, color='red', marker='X', label='Outlier')
    ax6.plot(np.unique(x_outlier),
             np.poly1d(np.polyfit(x_outlier, y_outlier, 1))(np.unique(x_outlier)),
             'r--', label=f'With outlier: r={r_outlier_p:.2f}')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_title('Effect of Outliers')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Correlation matrix heatmap
    ax7 = fig.add_subplot(gs[2, :])
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=ax7, cbar_kws={'label': 'Correlation'})
    ax7.set_title('Correlation Matrix Heatmap')

    plt.savefig('/tmp/correlation.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("Created visualizations:")
    print("  - Different correlation strengths and directions")
    print("  - Real-world example (study hours vs scores)")
    print("  - Linear vs non-linear relationships")
    print("  - Effect of outliers on correlation")
    print("  - Correlation matrix heatmap")
    print("  Saved to: /tmp/correlation.png")

    print("\n" + "=" * 60)
    print("Summary:")
    print("Correlation measures relationship strength between variables")
    print("- Pearson: Linear relationships, continuous data")
    print("- Spearman: Monotonic relationships, robust to outliers")
    print("- Range: -1 (perfect negative) to +1 (perfect positive)")
    print("- Correlation ≠ Causation (always consider confounders)")
    print("- Test significance with p-values")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Calculate Pearson r for X=[1,2,3,4,5], Y=[2,4,6,8,10]")
    print("2. Compare Pearson vs Spearman for ranked data")
    print("3. Create scatter plot showing r=-0.8 relationship")
    print("4. Test if correlation r=0.45 (n=30) is significant")
    print("5. Find strongest correlation in a dataset with 4+ variables")
    print("=" * 60)

if __name__ == "__main__":
    main()
