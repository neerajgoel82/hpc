"""
Analysis of Variance (ANOVA)
============================
Comparing means across multiple groups.

Topics:
- One-way ANOVA
- ANOVA assumptions
- F-statistic and p-value
- Post-hoc tests (Tukey HSD)
- Between-group vs within-group variance
- Effect size (eta-squared)

Run: python 06_anova.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 60)
    print("Analysis of Variance (ANOVA)")
    print("=" * 60)

    # 1. ANOVA Fundamentals
    print("\n1. ANOVA Fundamentals")
    print("-" * 40)

    print("Purpose: Compare means of 3 or more groups")
    print("Why not multiple t-tests?")
    print("  - Multiple t-tests inflate Type I error rate")
    print("  - With 3 groups: 3 comparisons needed")
    print("  - With 5 groups: 10 comparisons needed")
    print("  - ANOVA controls overall error rate at α")

    print("\nHypotheses:")
    print("  H0: μ1 = μ2 = μ3 = ... (all group means are equal)")
    print("  H1: At least one mean is different")

    print("\nKey concepts:")
    print("  - Between-group variance: Differences among group means")
    print("  - Within-group variance: Variability within each group")
    print("  - F-statistic: Ratio of between/within variance")
    print("  - Large F → groups differ significantly")

    # 2. One-Way ANOVA Example
    print("\n2. One-Way ANOVA")
    print("-" * 40)

    # Example: Test scores from three different teaching methods
    np.random.seed(42)
    method_a = np.array([85, 88, 92, 79, 95, 87, 91, 83, 89, 94])
    method_b = np.array([78, 82, 80, 85, 79, 83, 81, 84, 80, 82])
    method_c = np.array([92, 95, 98, 93, 97, 94, 96, 95, 99, 93])

    print("Example: Test Scores by Teaching Method")
    print(f"\nMethod A (n={len(method_a)}): {method_a[:5]}...")
    print(f"Method B (n={len(method_b)}): {method_b[:5]}...")
    print(f"Method C (n={len(method_c)}): {method_c[:5]}...")

    print("\nGroup statistics:")
    print(f"  Method A: mean={method_a.mean():.2f}, std={method_a.std(ddof=1):.2f}")
    print(f"  Method B: mean={method_b.mean():.2f}, std={method_b.std(ddof=1):.2f}")
    print(f"  Method C: mean={method_c.mean():.2f}, std={method_c.std(ddof=1):.2f}")

    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(method_a, method_b, method_c)

    print(f"\nOne-way ANOVA results:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    # Degrees of freedom
    k = 3  # Number of groups
    n_total = len(method_a) + len(method_b) + len(method_c)
    df_between = k - 1
    df_within = n_total - k

    print(f"\nDegrees of freedom:")
    print(f"  Between groups (k-1): {df_between}")
    print(f"  Within groups (N-k): {df_within}")

    # Decision
    alpha = 0.05
    print(f"\nDecision (α={alpha}):")
    if p_value < alpha:
        print(f"  p-value ({p_value:.6f}) < α ({alpha})")
        print(f"  ✓ Reject H0: At least one method produces different scores")
    else:
        print(f"  p-value ({p_value:.6f}) >= α ({alpha})")
        print(f"  ✗ Fail to reject H0: No significant difference between methods")

    # 3. ANOVA Table Breakdown
    print("\n3. ANOVA Table (Manual Calculation)")
    print("-" * 40)

    # Combine all data
    all_data = np.concatenate([method_a, method_b, method_c])
    grand_mean = all_data.mean()

    # Sum of squares
    ss_between = (len(method_a) * (method_a.mean() - grand_mean)**2 +
                  len(method_b) * (method_b.mean() - grand_mean)**2 +
                  len(method_c) * (method_c.mean() - grand_mean)**2)

    ss_within = (np.sum((method_a - method_a.mean())**2) +
                 np.sum((method_b - method_b.mean())**2) +
                 np.sum((method_c - method_c.mean())**2))

    ss_total = np.sum((all_data - grand_mean)**2)

    # Mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # F-statistic
    f_manual = ms_between / ms_within

    print("ANOVA Table:")
    print(f"{'Source':<15} {'SS':>12} {'df':>6} {'MS':>12} {'F':>10} {'p-value':>10}")
    print("-" * 70)
    print(f"{'Between':<15} {ss_between:>12.2f} {df_between:>6} {ms_between:>12.2f} {f_manual:>10.4f} {p_value:>10.6f}")
    print(f"{'Within':<15} {ss_within:>12.2f} {df_within:>6} {ms_within:>12.2f}")
    print(f"{'Total':<15} {ss_total:>12.2f} {n_total-1:>6}")

    print(f"\nInterpretation:")
    print(f"  SS_Between: Variance explained by group differences")
    print(f"  SS_Within: Variance due to individual differences")
    print(f"  F = MS_Between / MS_Within = {ms_between:.2f} / {ms_within:.2f} = {f_manual:.4f}")

    # Effect size (eta-squared)
    eta_squared = ss_between / ss_total
    print(f"\nEffect size (η²): {eta_squared:.4f}")
    print(f"  {eta_squared*100:.1f}% of variance explained by teaching method")
    if eta_squared < 0.01:
        effect = "small"
    elif eta_squared < 0.06:
        effect = "medium"
    else:
        effect = "large"
    print(f"  Effect size interpretation: {effect}")

    # 4. Post-hoc Tests
    print("\n4. Post-hoc Tests (Pairwise Comparisons)")
    print("-" * 40)

    print("If ANOVA is significant, perform post-hoc tests")
    print("to identify which specific groups differ")

    # Tukey HSD test
    from scipy.stats import tukey_hsd

    print("\nTukey HSD (Honestly Significant Difference):")

    # Pairwise comparisons
    groups = [method_a, method_b, method_c]
    group_names = ['Method_A', 'Method_B', 'Method_C']

    print("\nPairwise comparisons:")
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            t_stat, p_pairwise = stats.ttest_ind(groups[i], groups[j])
            mean_diff = groups[i].mean() - groups[j].mean()
            print(f"  {group_names[i]} vs {group_names[j]}:")
            print(f"    Mean difference: {mean_diff:.2f}")
            print(f"    p-value: {p_pairwise:.4f}", end="")
            if p_pairwise < alpha:
                print(" *significant*")
            else:
                print()

    # Bonferroni correction
    n_comparisons = 3
    alpha_bonferroni = alpha / n_comparisons
    print(f"\nBonferroni corrected α: {alpha_bonferroni:.4f}")
    print(f"  (α / number of comparisons = {alpha} / {n_comparisons})")

    # 5. ANOVA Assumptions
    print("\n5. ANOVA Assumptions")
    print("-" * 40)

    print("1. Independence: Observations are independent")
    print("   - Check: Study design ensures independence")

    print("\n2. Normality: Data in each group is normally distributed")
    print("   - Test: Shapiro-Wilk test for each group")

    for i, (group, name) in enumerate(zip([method_a, method_b, method_c], group_names)):
        stat, p_norm = stats.shapiro(group)
        print(f"   {name}: p={p_norm:.4f}", end="")
        if p_norm > 0.05:
            print(" (normal)")
        else:
            print(" (not normal)")

    print("\n3. Homogeneity of variance: Groups have equal variances")
    print("   - Test: Levene's test")

    stat_levene, p_levene = stats.levene(method_a, method_b, method_c)
    print(f"   Levene's test: p={p_levene:.4f}", end="")
    if p_levene > 0.05:
        print(" (equal variances)")
    else:
        print(" (variances differ)")

    print("\nIf assumptions violated:")
    print("  - Non-normality: Use Kruskal-Wallis test (non-parametric)")
    print("  - Unequal variances: Use Welch's ANOVA")

    # 6. Kruskal-Wallis Test (Non-parametric Alternative)
    print("\n6. Kruskal-Wallis Test (Non-parametric)")
    print("-" * 40)

    print("Alternative when normality assumption is violated")
    print("Tests if groups come from same distribution")

    h_stat, p_kruskal = stats.kruskal(method_a, method_b, method_c)

    print(f"\nKruskal-Wallis H-test:")
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value: {p_kruskal:.6f}")

    print(f"\nComparison:")
    print(f"  ANOVA F-test: p={p_value:.6f}")
    print(f"  Kruskal-Wallis: p={p_kruskal:.6f}")

    # 7. More Groups Example
    print("\n7. ANOVA with More Groups")
    print("-" * 40)

    # Example: Plant growth under different fertilizers
    print("Example: Plant Growth by Fertilizer Type")

    np.random.seed(42)
    fertilizer_1 = np.random.normal(20, 3, 15)
    fertilizer_2 = np.random.normal(22, 3, 15)
    fertilizer_3 = np.random.normal(25, 3, 15)
    fertilizer_4 = np.random.normal(23, 3, 15)
    control = np.random.normal(18, 3, 15)

    groups_fert = [control, fertilizer_1, fertilizer_2, fertilizer_3, fertilizer_4]
    group_names_fert = ['Control', 'Fert_1', 'Fert_2', 'Fert_3', 'Fert_4']

    print(f"\nGroup means (plant height in cm):")
    for name, group in zip(group_names_fert, groups_fert):
        print(f"  {name}: {group.mean():.2f} ± {group.std(ddof=1):.2f}")

    # Perform ANOVA
    f_fert, p_fert = stats.f_oneway(*groups_fert)

    print(f"\nOne-way ANOVA:")
    print(f"  F-statistic: {f_fert:.4f}")
    print(f"  p-value: {p_fert:.6f}")

    if p_fert < alpha:
        print(f"  ✓ Significant difference among fertilizer types")
    else:
        print(f"  ✗ No significant difference")

    # 8. Visualization
    print("\n8. Visualizing ANOVA Results")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Box plot comparison
    ax1 = axes[0, 0]
    data_combined = [method_a, method_b, method_c]
    bp = ax1.boxplot(data_combined, labels=['Method A', 'Method B', 'Method C'])
    ax1.set_ylabel('Test Score')
    ax1.set_title(f'Teaching Methods Comparison\n(p={p_value:.4f})')
    ax1.grid(True, alpha=0.3, axis='y')

    # Means with error bars
    ax2 = axes[0, 1]
    means = [method_a.mean(), method_b.mean(), method_c.mean()]
    sems = [stats.sem(method_a), stats.sem(method_b), stats.sem(method_c)]
    x_pos = np.arange(len(means))
    ax2.bar(x_pos, means, yerr=sems, capsize=10, alpha=0.7, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Method A', 'Method B', 'Method C'])
    ax2.set_ylabel('Mean Test Score')
    ax2.set_title('Group Means with Standard Error')
    ax2.grid(True, alpha=0.3, axis='y')

    # Individual points
    ax3 = axes[0, 2]
    for i, (data, label) in enumerate(zip(data_combined, ['A', 'B', 'C'])):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax3.scatter(x, data, alpha=0.6, s=50)
    ax3.boxplot(data_combined, positions=[1, 2, 3], widths=0.3,
               showfliers=False, showcaps=False)
    ax3.set_xticks([1, 2, 3])
    ax3.set_xticklabels(['Method A', 'Method B', 'Method C'])
    ax3.set_ylabel('Test Score')
    ax3.set_title('Distribution with Individual Points')
    ax3.grid(True, alpha=0.3, axis='y')

    # Variance components
    ax4 = axes[1, 0]
    variance_data = [ss_between, ss_within]
    colors = ['#ff9999', '#66b3ff']
    ax4.bar(['Between\nGroups', 'Within\nGroups'], variance_data, color=colors,
           edgecolor='black', alpha=0.7)
    ax4.set_ylabel('Sum of Squares')
    ax4.set_title('Variance Components')
    ax4.grid(True, alpha=0.3, axis='y')

    # Multiple groups (fertilizer)
    ax5 = axes[1, 1]
    data_fert = [control, fertilizer_1, fertilizer_2, fertilizer_3, fertilizer_4]
    bp2 = ax5.boxplot(data_fert, labels=['Control', 'F1', 'F2', 'F3', 'F4'])
    ax5.set_ylabel('Plant Height (cm)')
    ax5.set_title(f'Fertilizer Comparison\n(p={p_fert:.6f})')
    ax5.grid(True, alpha=0.3, axis='y')

    # Effect size visualization
    ax6 = axes[1, 2]
    effect_components = [eta_squared, 1 - eta_squared]
    labels_effect = [f'Explained by\ngroups\n({eta_squared*100:.1f}%)',
                     f'Unexplained\n({(1-eta_squared)*100:.1f}%)']
    ax6.pie(effect_components, labels=labels_effect, autopct='',
           colors=['#ff9999', '#lightgray'], startangle=90)
    ax6.set_title(f'Effect Size (η² = {eta_squared:.4f})')

    plt.tight_layout()
    plt.savefig('/tmp/anova.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("Created visualizations:")
    print("  - Box plots comparing groups")
    print("  - Bar chart with error bars")
    print("  - Distribution with individual data points")
    print("  - Variance components breakdown")
    print("  - Multiple groups comparison")
    print("  - Effect size pie chart")
    print("  Saved to: /tmp/anova.png")

    print("\n" + "=" * 60)
    print("Summary:")
    print("ANOVA compares means of 3+ groups simultaneously")
    print("- Tests if at least one group mean differs")
    print("- F-statistic = Between-group variance / Within-group variance")
    print("- If significant, use post-hoc tests for pairwise comparisons")
    print("- Check assumptions: independence, normality, equal variances")
    print("- Alternative: Kruskal-Wallis (non-parametric)")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Perform ANOVA on A=[10,12,14], B=[15,17,19], C=[20,22,24]")
    print("2. Calculate eta-squared for the above data")
    print("3. Interpret F=15.7, p=0.001 with 3 groups, n=30 each")
    print("4. Which post-hoc test would you use and why?")
    print("5. Check ANOVA assumptions for your own dataset")
    print("=" * 60)

if __name__ == "__main__":
    main()
