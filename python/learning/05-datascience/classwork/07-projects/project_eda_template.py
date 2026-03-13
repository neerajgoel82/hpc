"""
Project: Exploratory Data Analysis (EDA) Template
=================================================
Comprehensive EDA workflow template for any dataset.

This template provides a complete, reusable workflow for exploratory data analysis
that can be applied to any dataset. It demonstrates best practices and a systematic
approach to understanding data.

Steps:
1. Load and understand data structure
2. Data quality assessment
3. Data cleaning and preparation
4. Univariate analysis (individual variables)
5. Bivariate analysis (relationships between pairs)
6. Multivariate analysis (multiple variables together)
7. Statistical testing
8. Insights and conclusions

Skills: Complete data science workflow, pandas, visualization, statistics
Run: python project_eda_template.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def generate_sample_dataset():
    """Generate a sample dataset for demonstration."""
    print("Generating sample e-commerce dataset...")

    np.random.seed(42)
    n_samples = 1000

    # Generate customer data
    customer_ids = [f'CUST{str(i).zfill(4)}' for i in range(n_samples)]
    ages = np.random.gamma(shape=5, scale=10, size=n_samples).clip(18, 80).astype(int)
    genders = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04])
    countries = np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'France'],
                                 n_samples, p=[0.4, 0.25, 0.15, 0.1, 0.1])

    # Purchase behavior
    num_purchases = np.random.poisson(5, n_samples).clip(0, 30)
    avg_order_value = np.random.gamma(shape=3, scale=30, size=n_samples).clip(10, 500)
    total_spend = num_purchases * avg_order_value + np.random.normal(0, 50, n_samples)
    total_spend = total_spend.clip(0, None)

    # Time-related
    days_since_last_purchase = np.random.exponential(30, n_samples).clip(0, 365).astype(int)
    member_since_months = np.random.randint(1, 60, n_samples)

    # Engagement metrics
    email_opens = (num_purchases * np.random.uniform(0.3, 0.8, n_samples)).astype(int)
    website_visits = (num_purchases * np.random.uniform(2, 5, n_samples)).astype(int)

    # Customer segments (based on spend)
    segments = []
    for spend in total_spend:
        if spend < 200:
            segments.append('Low Value')
        elif spend < 600:
            segments.append('Medium Value')
        else:
            segments.append('High Value')

    # Product categories
    favorite_categories = np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'],
                                          n_samples, p=[0.25, 0.3, 0.2, 0.15, 0.1])

    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'country': countries,
        'num_purchases': num_purchases,
        'avg_order_value': avg_order_value,
        'total_spend': total_spend,
        'days_since_last_purchase': days_since_last_purchase,
        'member_since_months': member_since_months,
        'email_opens': email_opens,
        'website_visits': website_visits,
        'segment': segments,
        'favorite_category': favorite_categories
    })

    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'avg_order_value'] = np.nan

    missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_indices, 'days_since_last_purchase'] = np.nan

    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_indices, 'total_spend'] = df['total_spend'].max() * np.random.uniform(1.5, 2.5, 5)

    print(f"Generated dataset with {len(df)} customers")

    return df


def step1_load_and_understand(df):
    """Step 1: Load and understand data structure."""
    print("\n" + "=" * 80)
    print("STEP 1: LOAD AND UNDERSTAND DATA")
    print("=" * 80)

    print("\n1.1 Dataset Shape and Size")
    print("-" * 40)
    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    print(f"Memory:  {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    print("\n1.2 First and Last Rows")
    print("-" * 40)
    print("First 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())

    print("\n1.3 Column Information")
    print("-" * 40)
    print(df.info())

    print("\n1.4 Data Types Summary")
    print("-" * 40)
    print(df.dtypes.value_counts())

    print("\n1.5 Unique Values per Column")
    print("-" * 40)
    for col in df.columns:
        n_unique = df[col].nunique()
        print(f"{col:30} {n_unique:6d} unique values")


def step2_data_quality_assessment(df):
    """Step 2: Assess data quality."""
    print("\n" + "=" * 80)
    print("STEP 2: DATA QUALITY ASSESSMENT")
    print("=" * 80)

    print("\n2.1 Missing Values")
    print("-" * 40)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100)
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    }).sort_values('Missing_Count', ascending=False)

    print(missing_df.to_string(index=False))

    total_missing = missing.sum()
    total_cells = df.shape[0] * df.shape[1]
    print(f"\nTotal missing values: {total_missing} ({total_missing/total_cells*100:.2f}% of all cells)")

    print("\n2.2 Duplicate Rows")
    print("-" * 40)
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        print("Duplicate rows found!")
        print(df[df.duplicated(keep=False)].head())

    print("\n2.3 Data Type Issues")
    print("-" * 40)
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == 'object':
            # Check if should be numeric
            try:
                pd.to_numeric(df[col].dropna(), errors='raise')
                print(f"WARNING: '{col}' is object but could be numeric")
            except:
                pass

    print("\n2.4 Constant Columns")
    print("-" * 40)
    for col in df.columns:
        if df[col].nunique() == 1:
            print(f"WARNING: '{col}' has only one unique value")


def step3_data_cleaning(df):
    """Step 3: Clean and prepare data."""
    print("\n" + "=" * 80)
    print("STEP 3: DATA CLEANING")
    print("=" * 80)

    df = df.copy()

    print("\n3.1 Handling Missing Values")
    print("-" * 40)
    print(f"Missing values before: {df.isnull().sum().sum()}")

    # Strategy: Fill numeric with median, categorical with mode
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled '{col}' with median: {median_val:.2f}")

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  Filled '{col}' with mode: {mode_val}")

    print(f"Missing values after: {df.isnull().sum().sum()}")

    print("\n3.2 Handling Outliers")
    print("-" * 40)

    for col in numeric_cols:
        if col != 'customer_id':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Using 3*IQR for extreme outliers
            upper_bound = Q3 + 3 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"  '{col}': {outliers} extreme outliers detected")

    return df


def step4_univariate_analysis(df):
    """Step 4: Analyze individual variables."""
    print("\n" + "=" * 80)
    print("STEP 4: UNIVARIATE ANALYSIS")
    print("=" * 80)

    print("\n4.1 Descriptive Statistics (Numeric)")
    print("-" * 40)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe().T.round(2))

    print("\n4.2 Distribution Statistics")
    print("-" * 40)
    for col in numeric_cols:
        if col != 'customer_id':
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()
            print(f"{col:30} Skew: {skewness:6.2f}  Kurtosis: {kurtosis:6.2f}")

    print("\n4.3 Categorical Variables Summary")
    print("-" * 40)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'customer_id':
            print(f"\n{col}:")
            value_counts = df[col].value_counts()
            value_pcts = (value_counts / len(df) * 100).round(2)
            summary = pd.DataFrame({'Count': value_counts, 'Percentage': value_pcts})
            print(summary.to_string())


def step5_bivariate_analysis(df):
    """Step 5: Analyze relationships between variable pairs."""
    print("\n" + "=" * 80)
    print("STEP 5: BIVARIATE ANALYSIS")
    print("=" * 80)

    print("\n5.1 Correlation Matrix (Numeric Variables)")
    print("-" * 40)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'customer_id']
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix.round(3))

    print("\n5.2 Strong Correlations (|r| > 0.5)")
    print("-" * 40)
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                print(f"  {corr_matrix.columns[i]:30} <-> {corr_matrix.columns[j]:30}  r={corr_val:.3f}")

    print("\n5.3 Numeric vs Categorical Relationships")
    print("-" * 40)

    # Example: Total spend by segment
    if 'segment' in df.columns and 'total_spend' in df.columns:
        print("\nTotal Spend by Segment:")
        segment_stats = df.groupby('segment')['total_spend'].agg(['mean', 'median', 'std', 'count'])
        print(segment_stats.round(2))

    # Example: Age by gender
    if 'gender' in df.columns and 'age' in df.columns:
        print("\nAge by Gender:")
        gender_stats = df.groupby('gender')['age'].agg(['mean', 'median', 'std', 'count'])
        print(gender_stats.round(2))


def step6_multivariate_analysis(df):
    """Step 6: Analyze multiple variables together."""
    print("\n" + "=" * 80)
    print("STEP 6: MULTIVARIATE ANALYSIS")
    print("=" * 80)

    print("\n6.1 Group Analysis")
    print("-" * 40)

    # Analyze by multiple dimensions
    if 'segment' in df.columns and 'country' in df.columns:
        print("\nAverage Total Spend by Segment and Country:")
        pivot = df.pivot_table(values='total_spend', index='segment',
                               columns='country', aggfunc='mean')
        print(pivot.round(2))

    if 'favorite_category' in df.columns and 'gender' in df.columns:
        print("\nCustomer Count by Category and Gender:")
        crosstab = pd.crosstab(df['favorite_category'], df['gender'])
        print(crosstab)
        print("\nPercentages:")
        print((crosstab / crosstab.sum() * 100).round(2))


def step7_statistical_testing(df):
    """Step 7: Statistical hypothesis testing."""
    print("\n" + "=" * 80)
    print("STEP 7: STATISTICAL TESTING")
    print("=" * 80)

    print("\n7.1 T-Test: Comparing Two Groups")
    print("-" * 40)

    if 'gender' in df.columns and 'total_spend' in df.columns:
        # Compare spending between male and female
        male_spend = df[df['gender'] == 'Male']['total_spend']
        female_spend = df[df['gender'] == 'Female']['total_spend']

        if len(male_spend) > 0 and len(female_spend) > 0:
            t_stat, p_value = stats.ttest_ind(male_spend, female_spend)
            print(f"Comparing total_spend: Male vs Female")
            print(f"  Male mean:   ${male_spend.mean():.2f}")
            print(f"  Female mean: ${female_spend.mean():.2f}")
            print(f"  T-statistic: {t_stat:.4f}")
            print(f"  P-value:     {p_value:.4f}")
            if p_value < 0.05:
                print("  Result: Statistically significant difference (p < 0.05)")
            else:
                print("  Result: No significant difference (p >= 0.05)")

    print("\n7.2 ANOVA: Comparing Multiple Groups")
    print("-" * 40)

    if 'segment' in df.columns and 'num_purchases' in df.columns:
        # Compare purchases across segments
        groups = [df[df['segment'] == seg]['num_purchases'].values
                 for seg in df['segment'].unique()]

        f_stat, p_value = stats.f_oneway(*groups)
        print(f"Comparing num_purchases across segments")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  P-value:     {p_value:.4f}")
        if p_value < 0.05:
            print("  Result: Statistically significant difference (p < 0.05)")
        else:
            print("  Result: No significant difference (p >= 0.05)")

    print("\n7.3 Chi-Square: Categorical Independence")
    print("-" * 40)

    if 'gender' in df.columns and 'favorite_category' in df.columns:
        contingency = pd.crosstab(df['gender'], df['favorite_category'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        print(f"Testing independence: gender vs favorite_category")
        print(f"  Chi-square:  {chi2:.4f}")
        print(f"  P-value:     {p_value:.4f}")
        print(f"  DOF:         {dof}")
        if p_value < 0.05:
            print("  Result: Variables are dependent (p < 0.05)")
        else:
            print("  Result: Variables are independent (p >= 0.05)")


def step8_visualizations(df):
    """Step 8: Create comprehensive visualizations."""
    print("\n" + "=" * 80)
    print("STEP 8: VISUALIZATIONS")
    print("=" * 80)

    fig = plt.figure(figsize=(16, 12))

    # 1. Age distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(df['age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(df['age'].mean(), color='red', linestyle='--', label=f"Mean: {df['age'].mean():.1f}")
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Age Distribution', fontweight='bold')
    ax1.legend()

    # 2. Total spend distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(df['total_spend'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Total Spend ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Total Spend Distribution', fontweight='bold')

    # 3. Customers by country
    ax3 = plt.subplot(3, 3, 3)
    country_counts = df['country'].value_counts()
    ax3.bar(country_counts.index, country_counts.values, color='coral')
    ax3.set_xlabel('Country')
    ax3.set_ylabel('Number of Customers')
    ax3.set_title('Customers by Country', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Scatter: Age vs Total Spend
    ax4 = plt.subplot(3, 3, 4)
    scatter = ax4.scatter(df['age'], df['total_spend'], alpha=0.5,
                         c=df['num_purchases'], cmap='viridis')
    ax4.set_xlabel('Age')
    ax4.set_ylabel('Total Spend ($)')
    ax4.set_title('Age vs Total Spend (color=purchases)', fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Num Purchases')

    # 5. Box plot: Spend by segment
    ax5 = plt.subplot(3, 3, 5)
    df.boxplot(column='total_spend', by='segment', ax=ax5)
    ax5.set_xlabel('Segment')
    ax5.set_ylabel('Total Spend ($)')
    ax5.set_title('Total Spend by Segment', fontweight='bold')
    plt.suptitle('')

    # 6. Gender distribution
    ax6 = plt.subplot(3, 3, 6)
    gender_counts = df['gender'].value_counts()
    ax6.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Gender Distribution', fontweight='bold')

    # 7. Correlation heatmap
    ax7 = plt.subplot(3, 3, 7)
    numeric_cols = ['age', 'num_purchases', 'avg_order_value', 'total_spend',
                   'days_since_last_purchase', 'member_since_months']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax7, center=0)
    ax7.set_title('Correlation Matrix', fontweight='bold')

    # 8. Category distribution
    ax8 = plt.subplot(3, 3, 8)
    category_counts = df['favorite_category'].value_counts()
    category_counts.plot(kind='barh', ax=ax8, color='steelblue')
    ax8.set_xlabel('Number of Customers')
    ax8.set_title('Favorite Category Distribution', fontweight='bold')

    # 9. Purchases vs Website Visits
    ax9 = plt.subplot(3, 3, 9)
    ax9.scatter(df['website_visits'], df['num_purchases'], alpha=0.5, color='purple')
    ax9.set_xlabel('Website Visits')
    ax9.set_ylabel('Number of Purchases')
    ax9.set_title('Website Visits vs Purchases', fontweight='bold')

    # Add trend line
    z = np.polyfit(df['website_visits'], df['num_purchases'], 1)
    p = np.poly1d(z)
    ax9.plot(df['website_visits'].sort_values(),
            p(df['website_visits'].sort_values()), "r--", alpha=0.8)

    plt.tight_layout()
    print("Visualizations created successfully!")
    print("Close the plot window to continue...")
    plt.show()


def step9_insights_and_conclusions(df):
    """Step 9: Generate insights and conclusions."""
    print("\n" + "=" * 80)
    print("STEP 9: INSIGHTS AND CONCLUSIONS")
    print("=" * 80)

    print("\n9.1 Key Findings")
    print("-" * 40)

    # Customer base
    print(f"1. Customer Base:")
    print(f"   - Total customers: {len(df):,}")
    print(f"   - Average age: {df['age'].mean():.1f} years")
    print(f"   - Top country: {df['country'].mode()[0]}")

    # Spending patterns
    print(f"\n2. Spending Patterns:")
    print(f"   - Average total spend: ${df['total_spend'].mean():.2f}")
    print(f"   - Average order value: ${df['avg_order_value'].mean():.2f}")
    print(f"   - Average purchases: {df['num_purchases'].mean():.1f}")

    # Segment analysis
    if 'segment' in df.columns:
        print(f"\n3. Customer Segments:")
        segment_dist = df['segment'].value_counts()
        for segment, count in segment_dist.items():
            pct = (count / len(df)) * 100
            print(f"   - {segment}: {count} ({pct:.1f}%)")

    # Engagement
    print(f"\n4. Engagement Metrics:")
    print(f"   - Average email opens: {df['email_opens'].mean():.1f}")
    print(f"   - Average website visits: {df['website_visits'].mean():.1f}")
    print(f"   - Average days since last purchase: {df['days_since_last_purchase'].mean():.1f}")

    print("\n9.2 Recommendations")
    print("-" * 40)
    print("1. Focus marketing efforts on high-value segments")
    print("2. Re-engage customers with >90 days since last purchase")
    print("3. Increase email campaign effectiveness (low open rates)")
    print("4. Analyze top-performing categories for expansion")
    print("5. Implement personalized recommendations based on favorite category")

    print("\n9.3 Next Steps")
    print("-" * 40)
    print("1. Build predictive model for customer lifetime value")
    print("2. Develop churn prediction model")
    print("3. Create customer segmentation with clustering")
    print("4. Analyze seasonal patterns in purchasing")
    print("5. A/B test different marketing strategies")


def main():
    """Main execution function - Complete EDA workflow."""
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA) TEMPLATE")
    print("=" * 80)
    print("\nThis template demonstrates a complete, systematic approach to EDA.")
    print("You can adapt this workflow for any dataset!\n")

    # Generate sample data
    df = generate_sample_dataset()

    # Execute all EDA steps
    step1_load_and_understand(df)
    step2_data_quality_assessment(df)
    df_clean = step3_data_cleaning(df)
    step4_univariate_analysis(df_clean)
    step5_bivariate_analysis(df_clean)
    step6_multivariate_analysis(df_clean)
    step7_statistical_testing(df_clean)
    step8_visualizations(df_clean)
    step9_insights_and_conclusions(df_clean)

    print("\n" + "=" * 80)
    print("EDA COMPLETE!")
    print("=" * 80)
    print("\nThis template covered:")
    print("  - Data loading and structure understanding")
    print("  - Data quality assessment")
    print("  - Data cleaning and preparation")
    print("  - Univariate analysis")
    print("  - Bivariate analysis")
    print("  - Multivariate analysis")
    print("  - Statistical hypothesis testing")
    print("  - Comprehensive visualizations")
    print("  - Insights and recommendations")
    print("\nAdapt this workflow for your own datasets!")
    print("=" * 80)


if __name__ == "__main__":
    main()
