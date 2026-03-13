"""
Hypothesis Testing
==================
Statistical tests to make inferences about populations.

Topics:
- Null and alternative hypotheses
- t-tests (one-sample, two-sample, paired)
- z-tests
- p-values and significance levels
- Confidence intervals
- Type I and Type II errors

Run: python 03_hypothesis_testing.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Hypothesis Testing")
    print("=" * 60)

    # 1. Hypothesis Testing Basics
    print("\n1. Hypothesis Testing Fundamentals")
    print("-" * 40)

    print("Components of hypothesis testing:")
    print("  - Null hypothesis (H0): No effect or difference")
    print("  - Alternative hypothesis (H1): There is an effect")
    print("  - Significance level (α): Usually 0.05 (5%)")
    print("  - p-value: Probability of observing data if H0 is true")
    print("  - Decision: Reject H0 if p-value < α")

    print("\nTypes of errors:")
    print("  - Type I error (α): Reject H0 when it's true (false positive)")
    print("  - Type II error (β): Fail to reject H0 when it's false (false negative)")
    print("  - Power (1-β): Probability of correctly rejecting false H0")

    # 2. One-Sample t-test
    print("\n2. One-Sample t-test")
    print("-" * 40)

    # Sample data: test scores
    np.random.seed(42)
    scores = np.array([85, 88, 92, 79, 95, 87, 91, 83, 89, 94,
                       86, 90, 88, 92, 87, 85, 91, 89, 93, 88])
    hypothesized_mean = 85

    print(f"Sample: Test scores (n={len(scores)})")
    print(f"Data: {scores[:10]}...")
    print(f"\nH0: μ = {hypothesized_mean} (population mean is {hypothesized_mean})")
    print(f"H1: μ ≠ {hypothesized_mean} (population mean is different)")

    # Perform one-sample t-test
    t_stat, p_value = stats.ttest_1samp(scores, hypothesized_mean)

    print(f"\nSample statistics:")
    print(f"  Sample mean: {scores.mean():.2f}")
    print(f"  Sample std: {scores.std(ddof=1):.2f}")
    print(f"  Sample size: {len(scores)}")

    print(f"\nTest results:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Degrees of freedom: {len(scores) - 1}")

    alpha = 0.05
    print(f"\nDecision (α={alpha}):")
    if p_value < alpha:
        print(f"  p-value ({p_value:.4f}) < α ({alpha})")
        print(f"  ✓ Reject H0: Mean is significantly different from {hypothesized_mean}")
    else:
        print(f"  p-value ({p_value:.4f}) >= α ({alpha})")
        print(f"  ✗ Fail to reject H0: No significant difference from {hypothesized_mean}")

    # 3. Two-Sample Independent t-test
    print("\n3. Two-Sample Independent t-test")
    print("-" * 40)

    # Compare two groups
    group_a = np.array([23, 25, 27, 29, 31, 24, 28, 26, 30, 25])
    group_b = np.array([31, 33, 35, 32, 36, 34, 37, 33, 35, 32])

    print("Comparing two independent groups:")
    print(f"  Group A (n={len(group_a)}): {group_a[:5]}...")
    print(f"  Group B (n={len(group_b)}): {group_b[:5]}...")

    print(f"\nH0: μA = μB (no difference between groups)")
    print(f"H1: μA ≠ μB (groups have different means)")

    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(group_a, group_b)

    print(f"\nGroup statistics:")
    print(f"  Group A mean: {group_a.mean():.2f}, std: {group_a.std(ddof=1):.2f}")
    print(f"  Group B mean: {group_b.mean():.2f}, std: {group_b.std(ddof=1):.2f}")
    print(f"  Difference: {abs(group_a.mean() - group_b.mean()):.2f}")

    print(f"\nTest results:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")

    print(f"\nDecision (α={alpha}):")
    if p_value < alpha:
        print(f"  ✓ Reject H0: Groups are significantly different")
    else:
        print(f"  ✗ Fail to reject H0: No significant difference")

    # 4. Paired t-test
    print("\n4. Paired t-test")
    print("-" * 40)

    # Before and after measurements
    before = np.array([120, 125, 130, 122, 128, 135, 118, 127, 132, 124])
    after = np.array([115, 118, 125, 119, 122, 128, 115, 120, 127, 118])

    print("Paired measurements (before/after treatment):")
    print(f"  Before (n={len(before)}): {before[:5]}...")
    print(f"  After (n={len(after)}): {after[:5]}...")

    print(f"\nH0: μD = 0 (no difference before and after)")
    print(f"H1: μD ≠ 0 (there is a difference)")

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(before, after)

    differences = before - after
    print(f"\nPaired differences:")
    print(f"  Mean difference: {differences.mean():.2f}")
    print(f"  Std of differences: {differences.std(ddof=1):.2f}")

    print(f"\nTest results:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")

    print(f"\nDecision (α={alpha}):")
    if p_value < alpha:
        print(f"  ✓ Reject H0: Significant change after treatment")
        print(f"  Average reduction: {differences.mean():.2f} units")
    else:
        print(f"  ✗ Fail to reject H0: No significant change")

    # 5. z-test
    print("\n5. z-test (Large Sample)")
    print("-" * 40)

    # Large sample
    np.random.seed(42)
    large_sample = np.random.normal(100, 15, 1000)
    pop_mean = 100
    pop_std = 15

    print(f"Large sample test (n={len(large_sample)})")
    print(f"H0: μ = {pop_mean}")
    print(f"H1: μ ≠ {pop_mean}")

    # Calculate z-statistic
    sample_mean = large_sample.mean()
    z_stat = (sample_mean - pop_mean) / (pop_std / np.sqrt(len(large_sample)))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    print(f"\nSample mean: {sample_mean:.2f}")
    print(f"z-statistic: {z_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    print(f"\nDecision (α={alpha}):")
    if p_value < alpha:
        print(f"  ✓ Reject H0: Significant difference")
    else:
        print(f"  ✗ Fail to reject H0: No significant difference")

    # 6. Confidence Intervals
    print("\n6. Confidence Intervals")
    print("-" * 40)

    data = np.array([45, 47, 50, 52, 48, 51, 49, 46, 53, 50])
    confidence_level = 0.95
    alpha_ci = 1 - confidence_level

    print(f"Data: {data}")
    print(f"Confidence level: {confidence_level*100:.0f}%")

    # Calculate confidence interval
    mean = data.mean()
    sem = stats.sem(data)  # Standard error of mean
    ci = stats.t.interval(confidence_level, len(data)-1, loc=mean, scale=sem)

    print(f"\nSample statistics:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Std: {data.std(ddof=1):.2f}")
    print(f"  SEM: {sem:.2f}")

    print(f"\n{confidence_level*100:.0f}% Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]")
    print(f"Interpretation:")
    print(f"  We are {confidence_level*100:.0f}% confident the true mean lies in this interval")
    print(f"  Margin of error: ±{(ci[1] - mean):.2f}")

    # Different confidence levels
    print(f"\nConfidence intervals at different levels:")
    for conf in [0.90, 0.95, 0.99]:
        ci = stats.t.interval(conf, len(data)-1, loc=mean, scale=sem)
        width = ci[1] - ci[0]
        print(f"  {conf*100:.0f}%: [{ci[0]:.2f}, {ci[1]:.2f}], width: {width:.2f}")

    # 7. One-tailed vs Two-tailed Tests
    print("\n7. One-tailed vs Two-tailed Tests")
    print("-" * 40)

    sample = np.array([102, 105, 108, 103, 107, 104, 106, 105, 109, 103])
    mu_0 = 100

    print(f"Sample: {sample[:5]}...")
    print(f"Sample mean: {sample.mean():.2f}")

    # Two-tailed test
    t_stat, p_two = stats.ttest_1samp(sample, mu_0)
    print(f"\nTwo-tailed test (H1: μ ≠ {mu_0}):")
    print(f"  p-value: {p_two:.4f}")

    # One-tailed test (greater)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    print(f"\nOne-tailed test (H1: μ > {mu_0}):")
    print(f"  p-value: {p_one:.4f}")

    print(f"\nNote: One-tailed tests are more powerful but require")
    print(f"      a priori directional hypothesis")

    # 8. Visualization
    print("\n8. Visualizing Hypothesis Tests")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # One-sample t-test visualization
    x = np.linspace(scores.mean() - 4*scores.std(), scores.mean() + 4*scores.std(), 100)
    axes[0, 0].hist(scores, bins=8, density=True, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(scores.mean(), color='blue', linestyle='-', linewidth=2,
                      label=f'Sample mean: {scores.mean():.1f}')
    axes[0, 0].axvline(hypothesized_mean, color='red', linestyle='--', linewidth=2,
                      label=f'H0 mean: {hypothesized_mean}')
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('One-Sample t-test')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Two-sample comparison
    positions = [1, 2]
    axes[0, 1].boxplot([group_a, group_b], labels=['Group A', 'Group B'])
    axes[0, 1].scatter([1]*len(group_a), group_a, alpha=0.5, color='blue')
    axes[0, 1].scatter([2]*len(group_b), group_b, alpha=0.5, color='orange')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Two-Sample Independent t-test')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Paired t-test
    axes[1, 0].plot([1, 2], [before, after], 'o-', alpha=0.5, color='gray')
    axes[1, 0].plot([1, 2], [before.mean(), after.mean()], 'ro-', linewidth=3,
                   markersize=10, label='Mean')
    axes[1, 0].set_xticks([1, 2])
    axes[1, 0].set_xticklabels(['Before', 'After'])
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Paired t-test (Before/After)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Confidence interval
    means = []
    cis = []
    samples_ci = [np.random.normal(50, 10, 30) for _ in range(20)]
    for s in samples_ci:
        m = s.mean()
        ci = stats.t.interval(0.95, len(s)-1, loc=m, scale=stats.sem(s))
        means.append(m)
        cis.append(ci)

    for i, (m, ci) in enumerate(zip(means, cis)):
        color = 'blue' if ci[0] <= 50 <= ci[1] else 'red'
        axes[1, 1].plot([ci[0], ci[1]], [i, i], color=color, linewidth=2)
        axes[1, 1].plot(m, i, 'o', color=color, markersize=4)

    axes[1, 1].axvline(50, color='green', linestyle='--', linewidth=2, label='True mean')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Sample number')
    axes[1, 1].set_title('95% Confidence Intervals (20 samples)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('/tmp/hypothesis_testing.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("Created visualizations:")
    print("  - One-sample t-test comparison")
    print("  - Two-sample independent groups")
    print("  - Paired measurements before/after")
    print("  - Confidence intervals demonstration")
    print("  Saved to: /tmp/hypothesis_testing.png")

    print("\n" + "=" * 60)
    print("Summary:")
    print("Hypothesis testing helps make data-driven decisions")
    print("- Set null (H0) and alternative (H1) hypotheses")
    print("- Calculate test statistic (t, z, etc.)")
    print("- Compare p-value to significance level (α)")
    print("- Reject H0 if p-value < α")
    print("- Use confidence intervals to estimate parameters")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Perform one-sample t-test: data=[52,55,58,54,56], μ0=50")
    print("2. Compare two groups: A=[20,22,25,21,23], B=[28,30,32,29,31]")
    print("3. Test before=[100,105,110] vs after=[95,98,105]")
    print("4. Calculate 99% confidence interval for [15,18,20,17,19]")
    print("5. Interpret: p-value=0.03 with α=0.05")
    print("=" * 60)

if __name__ == "__main__":
    main()
