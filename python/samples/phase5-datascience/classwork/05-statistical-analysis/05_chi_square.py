"""
Chi-Square Tests
================
Statistical tests for categorical data.

Topics:
- Chi-square test for independence
- Chi-square goodness of fit test
- Contingency tables
- Expected vs observed frequencies
- Degrees of freedom
- Cramér's V (effect size)

Run: python 05_chi_square.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 60)
    print("Chi-Square Tests")
    print("=" * 60)

    # 1. Chi-Square Test for Independence
    print("\n1. Chi-Square Test for Independence")
    print("-" * 40)

    print("Purpose: Test if two categorical variables are independent")
    print("H0: Variables are independent (no association)")
    print("H1: Variables are dependent (there is association)")

    # Example: Gender vs Product Preference
    print("\nExample: Gender vs Product Preference")

    # Create contingency table
    data = {
        'Product_A': [30, 20],
        'Product_B': [25, 35],
        'Product_C': [15, 25]
    }

    contingency_table = pd.DataFrame(data, index=['Male', 'Female'])
    print("\nObserved frequencies (contingency table):")
    print(contingency_table)

    # Add totals
    contingency_with_totals = contingency_table.copy()
    contingency_with_totals['Total'] = contingency_with_totals.sum(axis=1)
    contingency_with_totals.loc['Total'] = contingency_with_totals.sum()

    print("\nWith row and column totals:")
    print(contingency_with_totals)

    # Perform chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"\nChi-square test results:")
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"    df = (rows - 1) × (columns - 1) = (2-1) × (3-1) = {dof}")

    # Expected frequencies
    print("\nExpected frequencies (if independent):")
    expected_df = pd.DataFrame(expected,
                               index=contingency_table.index,
                               columns=contingency_table.columns)
    print(expected_df.round(2))

    # Decision
    alpha = 0.05
    print(f"\nDecision (α={alpha}):")
    if p_value < alpha:
        print(f"  p-value ({p_value:.4f}) < α ({alpha})")
        print(f"  ✓ Reject H0: Gender and product preference are dependent")
        print(f"    There IS a significant association")
    else:
        print(f"  p-value ({p_value:.4f}) >= α ({alpha})")
        print(f"  ✗ Fail to reject H0: No significant association")

    # Effect size (Cramér's V)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    print(f"\nCramér's V (effect size): {cramers_v:.4f}")
    if cramers_v < 0.1:
        effect = "negligible"
    elif cramers_v < 0.3:
        effect = "small"
    elif cramers_v < 0.5:
        effect = "medium"
    else:
        effect = "large"
    print(f"  Effect size interpretation: {effect}")

    # 2. Chi-Square Goodness of Fit Test
    print("\n2. Chi-Square Goodness of Fit Test")
    print("-" * 40)

    print("Purpose: Test if observed frequencies match expected distribution")
    print("H0: Data follows the expected distribution")
    print("H1: Data does not follow the expected distribution")

    # Example: Die roll fairness
    print("\nExample: Testing if a die is fair")

    observed = np.array([45, 52, 48, 50, 47, 58])  # Observed frequencies
    expected_prop = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])  # Expected proportions
    n_rolls = observed.sum()

    print(f"\nDie rolled {n_rolls} times")
    print("Observed frequencies:")
    for i, freq in enumerate(observed, 1):
        print(f"  Face {i}: {freq} ({freq/n_rolls*100:.1f}%)")

    print("\nExpected frequencies (if fair):")
    expected_freq = n_rolls * expected_prop
    for i, freq in enumerate(expected_freq, 1):
        print(f"  Face {i}: {freq:.1f} ({1/6*100:.1f}%)")

    # Perform goodness of fit test
    chi2_gof, p_value_gof = stats.chisquare(observed, expected_freq)

    print(f"\nChi-square goodness of fit test:")
    print(f"  Chi-square statistic: {chi2_gof:.4f}")
    print(f"  p-value: {p_value_gof:.4f}")
    print(f"  Degrees of freedom: {len(observed) - 1}")

    print(f"\nDecision (α={alpha}):")
    if p_value_gof < alpha:
        print(f"  ✓ Reject H0: Die is NOT fair (biased)")
    else:
        print(f"  ✗ Fail to reject H0: Die appears fair")

    # 3. Larger Contingency Table Example
    print("\n3. Larger Contingency Table")
    print("-" * 40)

    # Example: Education level vs Income bracket
    print("Example: Education Level vs Income Bracket")

    education_income = pd.DataFrame({
        'Low': [45, 30, 15, 10],
        'Medium': [30, 50, 40, 20],
        'High': [10, 25, 45, 60]
    }, index=['High_School', 'Bachelor', 'Master', 'PhD'])

    print("\nObserved frequencies:")
    print(education_income)

    # Perform test
    chi2_ed, p_value_ed, dof_ed, expected_ed = stats.chi2_contingency(education_income)

    print(f"\nChi-square test results:")
    print(f"  Chi-square statistic: {chi2_ed:.4f}")
    print(f"  p-value: {p_value_ed:.6f}")
    print(f"  Degrees of freedom: {dof_ed}")

    print(f"\nDecision (α={alpha}):")
    if p_value_ed < alpha:
        print(f"  ✓ Reject H0: Education and income are dependent")
    else:
        print(f"  ✗ Fail to reject H0: No significant association")

    # Calculate standardized residuals
    observed_array = education_income.values
    expected_array = expected_ed
    std_residuals = (observed_array - expected_array) / np.sqrt(expected_array)

    print("\nStandardized residuals (shows which cells contribute most):")
    residuals_df = pd.DataFrame(std_residuals,
                                index=education_income.index,
                                columns=education_income.columns)
    print(residuals_df.round(2))
    print("\n  |residual| > 2 indicates significant deviation")

    # 4. 2x2 Contingency Table
    print("\n4. Special Case: 2×2 Contingency Table")
    print("-" * 40)

    # Example: Treatment vs Outcome
    print("Example: Treatment Effect on Recovery")

    treatment_outcome = pd.DataFrame({
        'Recovered': [40, 30],
        'Not_Recovered': [10, 20]
    }, index=['Treatment', 'Control'])

    print("\nObserved frequencies:")
    print(treatment_outcome)

    # Chi-square test
    chi2_2x2, p_value_2x2, dof_2x2, expected_2x2 = stats.chi2_contingency(treatment_outcome)

    print(f"\nChi-square test:")
    print(f"  Chi-square statistic: {chi2_2x2:.4f}")
    print(f"  p-value: {p_value_2x2:.4f}")

    # Fisher's exact test (alternative for small samples)
    oddsratio, p_fisher = stats.fisher_exact(treatment_outcome)
    print(f"\nFisher's exact test (recommended for small samples):")
    print(f"  Odds ratio: {oddsratio:.4f}")
    print(f"  p-value: {p_fisher:.4f}")

    print(f"\nDecision (α={alpha}):")
    if p_value_2x2 < alpha:
        print(f"  ✓ Reject H0: Treatment has significant effect")
    else:
        print(f"  ✗ Fail to reject H0: No significant treatment effect")

    # 5. Assumptions and Requirements
    print("\n5. Chi-Square Test Assumptions")
    print("-" * 40)

    print("Requirements:")
    print("  1. Data are counts/frequencies (not percentages or means)")
    print("  2. Observations are independent")
    print("  3. Categories are mutually exclusive")
    print("  4. Expected frequency in each cell ≥ 5")

    print("\nChecking expected frequencies:")
    min_expected = expected_array.min()
    cells_below_5 = (expected_array < 5).sum()

    print(f"  Minimum expected frequency: {min_expected:.2f}")
    print(f"  Cells with expected < 5: {cells_below_5}")

    if min_expected >= 5:
        print(f"  ✓ Assumption satisfied")
    else:
        print(f"  ✗ Warning: Some expected frequencies < 5")
        print(f"    Consider: Fisher's exact test or combining categories")

    # 6. Real-world Example
    print("\n6. Real-world Example: Survey Analysis")
    print("-" * 40)

    # Customer satisfaction vs service type
    print("Example: Customer Satisfaction by Service Type")

    satisfaction = pd.DataFrame({
        'Very_Satisfied': [120, 85, 95],
        'Satisfied': [80, 90, 85],
        'Neutral': [40, 50, 55],
        'Unsatisfied': [30, 45, 40]
    }, index=['Service_A', 'Service_B', 'Service_C'])

    print("\nSurvey results (n={}):".format(satisfaction.sum().sum()))
    print(satisfaction)

    chi2_survey, p_survey, dof_survey, exp_survey = stats.chi2_contingency(satisfaction)

    print(f"\nChi-square test results:")
    print(f"  Chi-square: {chi2_survey:.4f}")
    print(f"  p-value: {p_survey:.4f}")
    print(f"  df: {dof_survey}")

    # Cramér's V
    n_survey = satisfaction.sum().sum()
    min_dim_survey = min(satisfaction.shape[0] - 1, satisfaction.shape[1] - 1)
    cramers_v_survey = np.sqrt(chi2_survey / (n_survey * min_dim_survey))

    print(f"\nEffect size (Cramér's V): {cramers_v_survey:.4f}")

    if p_survey < alpha:
        print(f"\nConclusion: Customer satisfaction differs significantly")
        print(f"across service types (p={p_survey:.4f})")
    else:
        print(f"\nConclusion: No significant difference in satisfaction")

    # 7. Visualization
    print("\n7. Visualizing Chi-Square Tests")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Observed vs Expected for first example
    ax1 = axes[0, 0]
    x = np.arange(len(contingency_table.columns))
    width = 0.35
    for i, gender in enumerate(contingency_table.index):
        ax1.bar(x + i*width, contingency_table.iloc[i],
               width, label=gender, alpha=0.8)
    ax1.set_xlabel('Product')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Gender vs Product Preference\n(Observed)')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(contingency_table.columns)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Heatmap of contingency table
    ax2 = axes[0, 1]
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',
                ax=ax2, cbar_kws={'label': 'Frequency'})
    ax2.set_title('Contingency Table Heatmap')

    # Standardized residuals
    ax3 = axes[0, 2]
    sns.heatmap(residuals_df, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax3, cbar_kws={'label': 'Std Residual'})
    ax3.set_title('Standardized Residuals\n(Education vs Income)')

    # Goodness of fit
    ax4 = axes[1, 0]
    faces = np.arange(1, 7)
    ax4.bar(faces - 0.2, observed, 0.4, label='Observed', alpha=0.8)
    ax4.bar(faces + 0.2, expected_freq, 0.4, label='Expected', alpha=0.8)
    ax4.set_xlabel('Die Face')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Goodness of Fit Test\n(p={p_value_gof:.4f})')
    ax4.set_xticks(faces)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 2x2 table
    ax5 = axes[1, 1]
    x_treat = np.arange(len(treatment_outcome.columns))
    width = 0.35
    for i, treat in enumerate(treatment_outcome.index):
        ax5.bar(x_treat + i*width, treatment_outcome.iloc[i],
               width, label=treat, alpha=0.8)
    ax5.set_xlabel('Outcome')
    ax5.set_ylabel('Count')
    ax5.set_title(f'Treatment vs Control\n(p={p_value_2x2:.4f})')
    ax5.set_xticks(x_treat + width/2)
    ax5.set_xticklabels(treatment_outcome.columns)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # Satisfaction survey
    ax6 = axes[1, 2]
    satisfaction_pct = satisfaction.div(satisfaction.sum(axis=1), axis=0) * 100
    satisfaction_pct.T.plot(kind='bar', ax=ax6, width=0.8)
    ax6.set_xlabel('Satisfaction Level')
    ax6.set_ylabel('Percentage')
    ax6.set_title('Satisfaction by Service Type (%)')
    ax6.legend(title='Service')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('/tmp/chi_square.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("Created visualizations:")
    print("  - Observed frequencies in contingency table")
    print("  - Heatmap of categorical associations")
    print("  - Standardized residuals showing deviations")
    print("  - Goodness of fit comparison")
    print("  - 2×2 table treatment effect")
    print("  - Survey satisfaction distribution")
    print("  Saved to: /tmp/chi_square.png")

    print("\n" + "=" * 60)
    print("Summary:")
    print("Chi-square tests analyze relationships in categorical data")
    print("- Independence: Test if two variables are associated")
    print("- Goodness of fit: Test if data matches expected distribution")
    print("- Requires: Independent observations, adequate sample size")
    print("- Effect size: Cramér's V measures strength of association")
    print("- For 2×2 with small n: Use Fisher's exact test")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Test independence: [[10,20],[30,40]] at α=0.05")
    print("2. Goodness of fit: observed=[15,25,20,40], expected=[25,25,25,25]")
    print("3. Create 3×3 contingency table and perform chi-square test")
    print("4. Calculate Cramér's V for a contingency table")
    print("5. Interpret standardized residuals > 2")
    print("=" * 60)

if __name__ == "__main__":
    main()
