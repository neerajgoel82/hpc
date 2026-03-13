"""
Statistical Analysis Exercises
==============================
Comprehensive exercises covering all statistical analysis topics.

Topics covered:
- Descriptive statistics
- Probability distributions
- Hypothesis testing
- Correlation analysis
- Chi-square tests
- ANOVA

Run: python exercises.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def exercise_1_descriptive_stats():
    """Exercise 1: Calculate descriptive statistics for a dataset."""
    print("\n" + "=" * 60)
    print("Exercise 1: Descriptive Statistics")
    print("=" * 60)

    # Dataset: Student exam scores
    np.random.seed(42)
    scores = np.array([78, 85, 92, 67, 88, 91, 73, 95, 82, 89,
                       76, 90, 84, 79, 93, 87, 71, 94, 86, 81])

    print("\nDataset: Student exam scores")
    print(f"Data: {scores}")

    print("\nTask: Calculate the following:")
    print("1. Mean, median, mode")
    print("2. Variance and standard deviation")
    print("3. Quartiles and IQR")
    print("4. Skewness and kurtosis")
    print("5. Identify any outliers using IQR method")

    print("\n" + "-" * 40)
    print("Solutions:")
    print("-" * 40)

    # 1. Central tendency
    mean = np.mean(scores)
    median = np.median(scores)
    mode_result = stats.mode(scores, keepdims=True)
    mode = mode_result.mode[0] if mode_result.count[0] > 1 else "No mode"

    print(f"\n1. Central Tendency:")
    print(f"   Mean: {mean:.2f}")
    print(f"   Median: {median:.2f}")
    print(f"   Mode: {mode}")

    # 2. Dispersion
    variance = np.var(scores, ddof=1)
    std_dev = np.std(scores, ddof=1)

    print(f"\n2. Dispersion:")
    print(f"   Variance: {variance:.2f}")
    print(f"   Standard Deviation: {std_dev:.2f}")

    # 3. Quartiles
    q1 = np.percentile(scores, 25)
    q2 = np.percentile(scores, 50)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1

    print(f"\n3. Quartiles:")
    print(f"   Q1 (25th percentile): {q1:.2f}")
    print(f"   Q2 (50th percentile): {q2:.2f}")
    print(f"   Q3 (75th percentile): {q3:.2f}")
    print(f"   IQR: {iqr:.2f}")

    # 4. Shape
    skewness = stats.skew(scores)
    kurtosis = stats.kurtosis(scores)

    print(f"\n4. Distribution Shape:")
    print(f"   Skewness: {skewness:.3f}")
    print(f"   Kurtosis: {kurtosis:.3f}")

    # 5. Outliers
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = scores[(scores < lower_fence) | (scores > upper_fence)]

    print(f"\n5. Outlier Detection:")
    print(f"   Lower fence: {lower_fence:.2f}")
    print(f"   Upper fence: {upper_fence:.2f}")
    print(f"   Outliers: {outliers if len(outliers) > 0 else 'None'}")

def exercise_2_distributions():
    """Exercise 2: Work with probability distributions."""
    print("\n" + "=" * 60)
    print("Exercise 2: Probability Distributions")
    print("=" * 60)

    print("\nTask: Answer the following using appropriate distributions:")
    print("1. Normal(μ=100, σ=15): Find P(X > 115)")
    print("2. Binomial(n=20, p=0.3): Find P(X = 5)")
    print("3. Poisson(λ=4): Find P(X <= 3)")
    print("4. Generate 1000 samples from each and verify parameters")

    print("\n" + "-" * 40)
    print("Solutions:")
    print("-" * 40)

    # 1. Normal distribution
    mu, sigma = 100, 15
    p_greater = 1 - stats.norm.cdf(115, mu, sigma)
    print(f"\n1. Normal Distribution N({mu}, {sigma}):")
    print(f"   P(X > 115) = {p_greater:.4f} ({p_greater*100:.2f}%)")

    # 2. Binomial distribution
    n, p = 20, 0.3
    p_exact = stats.binom.pmf(5, n, p)
    print(f"\n2. Binomial Distribution B({n}, {p}):")
    print(f"   P(X = 5) = {p_exact:.4f}")

    # 3. Poisson distribution
    lambda_param = 4
    p_leq = stats.poisson.cdf(3, lambda_param)
    print(f"\n3. Poisson Distribution (λ={lambda_param}):")
    print(f"   P(X <= 3) = {p_leq:.4f}")

    # 4. Generate samples
    np.random.seed(42)
    normal_samples = np.random.normal(mu, sigma, 1000)
    binomial_samples = np.random.binomial(n, p, 1000)
    poisson_samples = np.random.poisson(lambda_param, 1000)

    print(f"\n4. Generated Samples (n=1000 each):")
    print(f"   Normal: mean={normal_samples.mean():.2f}, std={normal_samples.std():.2f}")
    print(f"           Expected: mean={mu}, std={sigma}")
    print(f"   Binomial: mean={binomial_samples.mean():.2f}, std={binomial_samples.std():.2f}")
    print(f"             Expected: mean={n*p:.2f}, std={np.sqrt(n*p*(1-p)):.2f}")
    print(f"   Poisson: mean={poisson_samples.mean():.2f}, std={poisson_samples.std():.2f}")
    print(f"            Expected: mean={lambda_param}, std={np.sqrt(lambda_param):.2f}")

def exercise_3_hypothesis_testing():
    """Exercise 3: Perform hypothesis tests."""
    print("\n" + "=" * 60)
    print("Exercise 3: Hypothesis Testing")
    print("=" * 60)

    print("\nScenario: A company claims average delivery time is 3 days.")
    print("Sample of 25 deliveries has mean=3.4 days, std=0.8 days.")
    print("\nTask:")
    print("1. Test if actual delivery time differs from claimed (α=0.05)")
    print("2. Calculate 95% confidence interval")
    print("3. Interpret the results")

    print("\n" + "-" * 40)
    print("Solutions:")
    print("-" * 40)

    # Given data
    sample_mean = 3.4
    sample_std = 0.8
    n = 25
    claimed_mean = 3.0
    alpha = 0.05

    # Calculate t-statistic
    se = sample_std / np.sqrt(n)
    t_stat = (sample_mean - claimed_mean) / se
    df = n - 1

    # p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    print(f"\n1. Hypothesis Test:")
    print(f"   H0: μ = {claimed_mean} days")
    print(f"   H1: μ ≠ {claimed_mean} days")
    print(f"   Sample mean: {sample_mean} days")
    print(f"   Standard error: {se:.4f}")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_value:.4f}")

    print(f"\n   Decision (α={alpha}):")
    if p_value < alpha:
        print(f"   ✓ Reject H0: Delivery time differs significantly from {claimed_mean} days")
    else:
        print(f"   ✗ Fail to reject H0: No significant difference from {claimed_mean} days")

    # 2. Confidence interval
    ci = stats.t.interval(0.95, df, loc=sample_mean, scale=se)

    print(f"\n2. 95% Confidence Interval:")
    print(f"   [{ci[0]:.2f}, {ci[1]:.2f}] days")
    print(f"   We are 95% confident the true mean delivery time is in this range")

    # 3. Interpretation
    print(f"\n3. Interpretation:")
    print(f"   The sample mean ({sample_mean} days) is significantly higher than")
    print(f"   the claimed {claimed_mean} days (p={p_value:.4f}).")
    print(f"   The 95% CI does not contain {claimed_mean}, confirming the result.")

def exercise_4_correlation():
    """Exercise 4: Analyze correlations."""
    print("\n" + "=" * 60)
    print("Exercise 4: Correlation Analysis")
    print("=" * 60)

    # Dataset: Study time vs exam score
    np.random.seed(42)
    study_time = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    exam_scores = 40 + 4 * study_time + np.random.normal(0, 5, len(study_time))

    print("\nDataset: Study time (hours) vs Exam scores")
    print(f"Study time: {study_time}")
    print(f"Exam scores: {exam_scores.astype(int)}")

    print("\nTask:")
    print("1. Calculate Pearson correlation coefficient")
    print("2. Test if correlation is significant (α=0.05)")
    print("3. Calculate coefficient of determination (R²)")
    print("4. Interpret the strength and direction")

    print("\n" + "-" * 40)
    print("Solutions:")
    print("-" * 40)

    # 1. Pearson correlation
    r, p_value = stats.pearsonr(study_time, exam_scores)

    print(f"\n1. Pearson Correlation:")
    print(f"   r = {r:.4f}")

    # 2. Significance test
    alpha = 0.05
    print(f"\n2. Significance Test:")
    print(f"   p-value = {p_value:.4f}")
    print(f"   α = {alpha}")
    if p_value < alpha:
        print(f"   ✓ Correlation is statistically significant")
    else:
        print(f"   ✗ Correlation is not statistically significant")

    # 3. R-squared
    r_squared = r ** 2
    print(f"\n3. Coefficient of Determination:")
    print(f"   R² = {r_squared:.4f}")
    print(f"   {r_squared*100:.1f}% of variance in exam scores explained by study time")

    # 4. Interpretation
    if abs(r) >= 0.9:
        strength = "very strong"
    elif abs(r) >= 0.7:
        strength = "strong"
    elif abs(r) >= 0.5:
        strength = "moderate"
    else:
        strength = "weak"

    direction = "positive" if r > 0 else "negative"

    print(f"\n4. Interpretation:")
    print(f"   Strength: {strength}")
    print(f"   Direction: {direction}")
    print(f"   Conclusion: There is a {strength} {direction} linear relationship")
    print(f"   between study time and exam scores.")

def exercise_5_chi_square():
    """Exercise 5: Chi-square test of independence."""
    print("\n" + "=" * 60)
    print("Exercise 5: Chi-Square Test")
    print("=" * 60)

    print("\nScenario: Survey of 200 people on smartphone preference by age group")

    # Create contingency table
    data = {
        'Brand_A': [30, 45, 25],
        'Brand_B': [40, 30, 30]
    }
    contingency = pd.DataFrame(data, index=['Young', 'Middle', 'Senior'])

    print("\nObserved frequencies:")
    print(contingency)

    print("\nTask:")
    print("1. Test if phone preference is independent of age group")
    print("2. Calculate expected frequencies")
    print("3. Interpret the results (α=0.05)")

    print("\n" + "-" * 40)
    print("Solutions:")
    print("-" * 40)

    # Perform chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    print(f"\n1. Chi-Square Test:")
    print(f"   H0: Phone preference and age are independent")
    print(f"   H1: Phone preference and age are dependent")
    print(f"   Chi-square statistic: {chi2:.4f}")
    print(f"   p-value: {p_value:.4f}")
    print(f"   Degrees of freedom: {dof}")

    # 2. Expected frequencies
    print(f"\n2. Expected Frequencies:")
    expected_df = pd.DataFrame(expected, index=contingency.index,
                              columns=contingency.columns)
    print(expected_df.round(2))

    # 3. Decision
    alpha = 0.05
    print(f"\n3. Interpretation (α={alpha}):")
    if p_value < alpha:
        print(f"   ✓ Reject H0: Phone preference depends on age group")
    else:
        print(f"   ✗ Fail to reject H0: No significant association")

    # Effect size
    n = contingency.sum().sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    print(f"\n   Cramér's V: {cramers_v:.4f}")

def exercise_6_anova():
    """Exercise 6: One-way ANOVA test."""
    print("\n" + "=" * 60)
    print("Exercise 6: ANOVA")
    print("=" * 60)

    print("\nScenario: Test effectiveness of three different training programs")
    print("on employee productivity scores.")

    # Create data
    np.random.seed(42)
    program_a = np.array([75, 78, 82, 80, 77, 79, 81, 76, 83, 78])
    program_b = np.array([82, 85, 88, 84, 86, 83, 87, 85, 89, 84])
    program_c = np.array([79, 81, 78, 80, 82, 79, 81, 80, 83, 78])

    print(f"\nProgram A: {program_a}")
    print(f"Program B: {program_b}")
    print(f"Program C: {program_c}")

    print("\nTask:")
    print("1. Perform one-way ANOVA")
    print("2. Calculate effect size (eta-squared)")
    print("3. Interpret results (α=0.05)")
    print("4. If significant, identify which programs differ")

    print("\n" + "-" * 40)
    print("Solutions:")
    print("-" * 40)

    # 1. One-way ANOVA
    f_stat, p_value = stats.f_oneway(program_a, program_b, program_c)

    print(f"\n1. One-Way ANOVA:")
    print(f"   H0: μA = μB = μC (all programs equally effective)")
    print(f"   H1: At least one program differs")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   p-value: {p_value:.6f}")

    # 2. Effect size
    all_data = np.concatenate([program_a, program_b, program_c])
    grand_mean = all_data.mean()

    ss_between = (len(program_a) * (program_a.mean() - grand_mean)**2 +
                  len(program_b) * (program_b.mean() - grand_mean)**2 +
                  len(program_c) * (program_c.mean() - grand_mean)**2)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total

    print(f"\n2. Effect Size:")
    print(f"   Eta-squared (η²): {eta_squared:.4f}")
    print(f"   {eta_squared*100:.1f}% of variance explained by program type")

    # 3. Decision
    alpha = 0.05
    print(f"\n3. Interpretation (α={alpha}):")
    if p_value < alpha:
        print(f"   ✓ Reject H0: At least one program differs significantly")
    else:
        print(f"   ✗ Fail to reject H0: No significant difference")

    # 4. Pairwise comparisons (if significant)
    if p_value < alpha:
        print(f"\n4. Post-hoc Pairwise Comparisons:")
        programs = [program_a, program_b, program_c]
        names = ['Program A', 'Program B', 'Program C']

        for i in range(len(programs)):
            for j in range(i+1, len(programs)):
                t_stat, p_pair = stats.ttest_ind(programs[i], programs[j])
                print(f"   {names[i]} vs {names[j]}: p={p_pair:.4f}", end="")
                if p_pair < alpha:
                    print(" *significant*")
                else:
                    print()

def exercise_7_comprehensive():
    """Exercise 7: Comprehensive analysis of a dataset."""
    print("\n" + "=" * 60)
    print("Exercise 7: Comprehensive Statistical Analysis")
    print("=" * 60)

    print("\nDataset: Customer satisfaction survey with demographics")

    # Create dataset
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'Age': np.random.randint(18, 70, n),
        'Income': np.random.randint(20000, 120000, n),
        'Satisfaction': np.random.randint(1, 11, n),
        'Gender': np.random.choice(['M', 'F'], n),
        'Product': np.random.choice(['A', 'B', 'C'], n)
    })

    # Add some correlations
    df['Satisfaction'] = (df['Income'] / 15000 + np.random.normal(0, 2, n)).clip(1, 10).astype(int)

    print("\nDataset preview:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")

    print("\nTask: Perform comprehensive analysis:")
    print("1. Descriptive statistics for numerical variables")
    print("2. Test correlation between Age and Satisfaction")
    print("3. Chi-square test: Gender vs Product preference")
    print("4. ANOVA: Satisfaction across Product types")

    print("\n" + "-" * 40)
    print("Solutions:")
    print("-" * 40)

    # 1. Descriptive statistics
    print("\n1. Descriptive Statistics:")
    print(df[['Age', 'Income', 'Satisfaction']].describe())

    # 2. Correlation
    r, p = stats.pearsonr(df['Age'], df['Satisfaction'])
    print(f"\n2. Correlation Analysis:")
    print(f"   Age vs Satisfaction: r={r:.4f}, p={p:.4f}")
    if p < 0.05:
        print(f"   Correlation is significant")
    else:
        print(f"   Correlation is not significant")

    # 3. Chi-square test
    contingency = pd.crosstab(df['Gender'], df['Product'])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

    print(f"\n3. Chi-Square Test (Gender vs Product):")
    print(f"   Contingency table:")
    print(contingency)
    print(f"   Chi-square: {chi2:.4f}, p={p_chi:.4f}")
    if p_chi < 0.05:
        print(f"   Significant association found")
    else:
        print(f"   No significant association")

    # 4. ANOVA
    groups = [df[df['Product'] == prod]['Satisfaction'].values
              for prod in ['A', 'B', 'C']]
    f_stat, p_anova = stats.f_oneway(*groups)

    print(f"\n4. ANOVA (Satisfaction across Products):")
    for prod in ['A', 'B', 'C']:
        mean_sat = df[df['Product'] == prod]['Satisfaction'].mean()
        print(f"   Product {prod}: mean={mean_sat:.2f}")
    print(f"   F-statistic: {f_stat:.4f}, p={p_anova:.4f}")
    if p_anova < 0.05:
        print(f"   Significant difference in satisfaction across products")
    else:
        print(f"   No significant difference")

def exercise_8_visualization():
    """Exercise 8: Create comprehensive statistical visualizations."""
    print("\n" + "=" * 60)
    print("Exercise 8: Statistical Visualizations")
    print("=" * 60)

    print("\nTask: Create visualizations for statistical concepts")

    # Generate data
    np.random.seed(42)

    # Normal distribution
    normal_data = np.random.normal(50, 10, 1000)

    # Two groups for t-test
    group1 = np.random.normal(100, 15, 50)
    group2 = np.random.normal(110, 15, 50)

    # Correlation data
    x = np.linspace(0, 10, 50)
    y = 2 * x + np.random.normal(0, 2, 50)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Distribution with descriptive stats
    ax1 = axes[0, 0]
    ax1.hist(normal_data, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(normal_data.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {normal_data.mean():.1f}')
    ax1.axvline(np.median(normal_data), color='green', linestyle='--',
               linewidth=2, label=f'Median: {np.median(normal_data):.1f}')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution with Central Tendency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Two-sample comparison
    ax2 = axes[0, 1]
    ax2.boxplot([group1, group2], labels=['Group 1', 'Group 2'])
    t_stat, p_val = stats.ttest_ind(group1, group2)
    ax2.set_ylabel('Value')
    ax2.set_title(f'Two-Sample Comparison\n(t={t_stat:.2f}, p={p_val:.4f})')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Correlation scatter plot
    ax3 = axes[1, 0]
    r, p = stats.pearsonr(x, y)
    ax3.scatter(x, y, alpha=0.6)
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    ax3.plot(x, p_line(x), 'r--', linewidth=2)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title(f'Correlation Analysis\n(r={r:.3f}, p={p:.4f})')
    ax3.grid(True, alpha=0.3)

    # 4. ANOVA comparison
    ax4 = axes[1, 1]
    anova_a = np.random.normal(20, 3, 30)
    anova_b = np.random.normal(25, 3, 30)
    anova_c = np.random.normal(22, 3, 30)
    f, p_anova = stats.f_oneway(anova_a, anova_b, anova_c)

    data_anova = [anova_a, anova_b, anova_c]
    ax4.boxplot(data_anova, labels=['Group A', 'Group B', 'Group C'])
    ax4.set_ylabel('Value')
    ax4.set_title(f'ANOVA Comparison\n(F={f:.2f}, p={p_anova:.4f})')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/tmp/exercises_visualizations.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("\nCreated comprehensive statistical visualizations")
    print("  Saved to: /tmp/exercises_visualizations.png")

def main():
    print("=" * 60)
    print("Statistical Analysis Exercises")
    print("=" * 60)

    print("\nThis module contains 8 comprehensive exercises covering:")
    print("  1. Descriptive Statistics")
    print("  2. Probability Distributions")
    print("  3. Hypothesis Testing")
    print("  4. Correlation Analysis")
    print("  5. Chi-Square Tests")
    print("  6. ANOVA")
    print("  7. Comprehensive Dataset Analysis")
    print("  8. Statistical Visualizations")

    # Run all exercises
    exercise_1_descriptive_stats()
    exercise_2_distributions()
    exercise_3_hypothesis_testing()
    exercise_4_correlation()
    exercise_5_chi_square()
    exercise_6_anova()
    exercise_7_comprehensive()
    exercise_8_visualization()

    print("\n" + "=" * 60)
    print("All Exercises Completed!")
    print("=" * 60)

    print("\nKey Takeaways:")
    print("1. Descriptive stats summarize data characteristics")
    print("2. Choose appropriate distribution based on data type")
    print("3. Hypothesis tests help make data-driven decisions")
    print("4. Correlation measures relationships, not causation")
    print("5. Chi-square tests analyze categorical associations")
    print("6. ANOVA compares multiple group means simultaneously")
    print("7. Always check assumptions before applying tests")
    print("8. Visualizations aid in understanding statistical results")

    print("\n" + "=" * 60)
    print("Practice Challenges:")
    print("1. Apply these methods to your own datasets")
    print("2. Interpret results in context of research questions")
    print("3. Check and validate statistical assumptions")
    print("4. Combine multiple tests for comprehensive analysis")
    print("5. Create publication-quality visualizations")
    print("=" * 60)

if __name__ == "__main__":
    main()
