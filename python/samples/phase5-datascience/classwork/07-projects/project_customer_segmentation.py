"""
Project: Customer Segmentation
==============================
K-means clustering on customer data with comprehensive analysis and visualization.

Dataset: Customer purchase history and demographics
Goals:
- Load and preprocess customer data
- Feature engineering and scaling
- Determine optimal number of clusters
- Apply K-Means clustering
- Analyze and profile segments
- Visualize segments in multiple dimensions
- Generate actionable insights

Skills: Pandas, scikit-learn, Matplotlib, Seaborn
Run: python project_customer_segmentation.py
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def generate_customer_data(n_customers=500):
    """Generate synthetic customer data with distinct segments."""
    print("Generating customer data...")

    np.random.seed(42)

    # Generate 4 natural customer segments
    segments = []

    # Segment 1: Budget Shoppers (30%)
    n1 = int(n_customers * 0.30)
    seg1 = {
        'age': np.random.normal(35, 10, n1),
        'income': np.random.normal(45000, 10000, n1),
        'spending_score': np.random.normal(30, 8, n1),
        'frequency': np.random.randint(2, 6, n1),
        'avg_transaction': np.random.normal(25, 8, n1),
        'tenure_months': np.random.randint(3, 24, n1)
    }
    segments.append(pd.DataFrame(seg1))

    # Segment 2: Premium Customers (20%)
    n2 = int(n_customers * 0.20)
    seg2 = {
        'age': np.random.normal(45, 8, n2),
        'income': np.random.normal(95000, 15000, n2),
        'spending_score': np.random.normal(85, 8, n2),
        'frequency': np.random.randint(15, 30, n2),
        'avg_transaction': np.random.normal(150, 30, n2),
        'tenure_months': np.random.randint(24, 60, n2)
    }
    segments.append(pd.DataFrame(seg2))

    # Segment 3: Young Professionals (25%)
    n3 = int(n_customers * 0.25)
    seg3 = {
        'age': np.random.normal(28, 5, n3),
        'income': np.random.normal(65000, 12000, n3),
        'spending_score': np.random.normal(60, 10, n3),
        'frequency': np.random.randint(8, 15, n3),
        'avg_transaction': np.random.normal(75, 20, n3),
        'tenure_months': np.random.randint(6, 30, n3)
    }
    segments.append(pd.DataFrame(seg3))

    # Segment 4: Occasional Shoppers (25%)
    n4 = n_customers - n1 - n2 - n3
    seg4 = {
        'age': np.random.normal(50, 12, n4),
        'income': np.random.normal(55000, 15000, n4),
        'spending_score': np.random.normal(45, 12, n4),
        'frequency': np.random.randint(1, 4, n4),
        'avg_transaction': np.random.normal(45, 15, n4),
        'tenure_months': np.random.randint(1, 36, n4)
    }
    segments.append(pd.DataFrame(seg4))

    # Combine all segments
    df = pd.concat(segments, ignore_index=True)

    # Add customer IDs
    df['customer_id'] = [f'C{str(i).zfill(4)}' for i in range(len(df))]

    # Clip values to reasonable ranges
    df['age'] = df['age'].clip(18, 80).astype(int)
    df['income'] = df['income'].clip(20000, 150000)
    df['spending_score'] = df['spending_score'].clip(0, 100).astype(int)
    df['avg_transaction'] = df['avg_transaction'].clip(10, 300)

    # Calculate total spending
    df['total_spending'] = df['frequency'] * df['avg_transaction']

    # Add some missing values
    missing_mask = np.random.random(len(df)) < 0.02
    df.loc[missing_mask, 'income'] = np.nan

    print(f"Generated {len(df)} customer records")

    return df


def explore_data(df):
    """Exploratory data analysis."""
    print("\n" + "=" * 60)
    print("Exploratory Data Analysis")
    print("=" * 60)

    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDescriptive statistics:")
    print(df.describe().round(2))

    # Correlation analysis
    print("\nCorrelation matrix:")
    numeric_cols = ['age', 'income', 'spending_score', 'frequency', 'avg_transaction', 'total_spending']
    corr = df[numeric_cols].corr()
    print(corr.round(2))


def preprocess_data(df):
    """Preprocess and engineer features."""
    print("\n" + "=" * 60)
    print("Data Preprocessing")
    print("=" * 60)

    # Make a copy
    df = df.copy()

    # Handle missing values
    print(f"Missing values before: {df.isnull().sum().sum()}")
    df['income'].fillna(df['income'].median(), inplace=True)
    print(f"Missing values after: {df.isnull().sum().sum()}")

    # Feature engineering
    print("\nEngineering new features...")

    # Customer lifetime value
    df['customer_ltv'] = df['total_spending'] * (df['tenure_months'] / 12)

    # Spending per month
    df['spending_per_month'] = df['total_spending'] / df['tenure_months'].replace(0, 1)

    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100],
                              labels=['Young', 'Middle', 'Senior', 'Elder'])

    # Income brackets
    df['income_bracket'] = pd.cut(df['income'],
                                   bins=[0, 40000, 70000, 100000, 200000],
                                   labels=['Low', 'Medium', 'High', 'Very High'])

    print("New features created:")
    print("  - customer_ltv")
    print("  - spending_per_month")
    print("  - age_group")
    print("  - income_bracket")

    return df


def prepare_features_for_clustering(df):
    """Select and scale features for clustering."""
    print("\n" + "=" * 60)
    print("Feature Preparation for Clustering")
    print("=" * 60)

    # Select features for clustering
    feature_cols = ['age', 'income', 'spending_score', 'frequency',
                    'avg_transaction', 'tenure_months', 'total_spending']

    X = df[feature_cols].copy()

    print(f"Features for clustering: {feature_cols}")
    print(f"Feature matrix shape: {X.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nFeatures standardized (mean=0, std=1)")
    print("Sample scaled values:")
    print(pd.DataFrame(X_scaled, columns=feature_cols).head())

    return X_scaled, feature_cols, scaler


def find_optimal_clusters(X_scaled, max_k=10):
    """Use elbow method and silhouette analysis to find optimal k."""
    print("\n" + "=" * 60)
    print("Finding Optimal Number of Clusters")
    print("=" * 60)

    inertias = []
    k_range = range(2, max_k + 1)

    print("Computing inertia for different k values...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        print(f"  k={k}: inertia={kmeans.inertia_:.2f}")

    # Calculate elbow using rate of change
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    elbow_idx = np.argmax(second_diffs) + 2

    print(f"\nSuggested optimal k (elbow method): {elbow_idx}")

    return inertias, k_range, elbow_idx


def perform_clustering(X_scaled, n_clusters=4):
    """Apply K-Means clustering."""
    print("\n" + "=" * 60)
    print(f"Performing K-Means Clustering (k={n_clusters})")
    print("=" * 60)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    print(f"Clustering complete!")
    print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
    print(f"Inertia: {kmeans.inertia_:.2f}")

    # Distribution of customers across clusters
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster distribution:")
    for cluster, count in zip(unique, counts):
        pct = (count / len(labels)) * 100
        print(f"  Cluster {cluster}: {count} customers ({pct:.1f}%)")

    return labels, kmeans


def analyze_segments(df, labels):
    """Analyze characteristics of each segment."""
    print("\n" + "=" * 60)
    print("Customer Segment Analysis")
    print("=" * 60)

    # Add cluster labels to dataframe
    df = df.copy()
    df['segment'] = labels

    # Analyze each segment
    numeric_cols = ['age', 'income', 'spending_score', 'frequency',
                    'avg_transaction', 'tenure_months', 'total_spending', 'customer_ltv']

    for segment in sorted(df['segment'].unique()):
        print(f"\n{'=' * 60}")
        print(f"SEGMENT {segment}")
        print('=' * 60)

        segment_df = df[df['segment'] == segment]
        n_customers = len(segment_df)
        pct = (n_customers / len(df)) * 100

        print(f"Size: {n_customers} customers ({pct:.1f}%)")
        print("\nKey Statistics:")

        for col in numeric_cols:
            mean_val = segment_df[col].mean()
            median_val = segment_df[col].median()
            print(f"  {col:20} Mean: {mean_val:10.2f}  Median: {median_val:10.2f}")

        # Most common demographics
        if 'age_group' in segment_df.columns:
            print(f"\nMost common age group: {segment_df['age_group'].mode()[0]}")
        if 'income_bracket' in segment_df.columns:
            print(f"Most common income bracket: {segment_df['income_bracket'].mode()[0]}")

    return df


def name_segments(df):
    """Assign meaningful names to segments based on characteristics."""
    print("\n" + "=" * 60)
    print("Segment Naming")
    print("=" * 60)

    # Analyze each segment to name them
    segment_profiles = {}

    for segment in sorted(df['segment'].unique()):
        segment_df = df[df['segment'] == segment]

        avg_spending = segment_df['total_spending'].mean()
        avg_frequency = segment_df['frequency'].mean()
        avg_ltv = segment_df['customer_ltv'].mean()

        # Simple naming logic based on key metrics
        if avg_spending > 3000 and avg_frequency > 15:
            name = "Premium Customers"
        elif avg_spending < 500 and avg_frequency < 5:
            name = "Budget Shoppers"
        elif segment_df['age'].mean() < 35 and avg_spending > 1000:
            name = "Young Professionals"
        else:
            name = "Occasional Shoppers"

        segment_profiles[segment] = name
        print(f"Segment {segment}: {name}")

    df['segment_name'] = df['segment'].map(segment_profiles)

    return df, segment_profiles


def visualize_segments(df, X_scaled, inertias, k_range):
    """Create comprehensive visualizations."""
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 12))

    # 1. Elbow plot
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. PCA visualization (2D)
    ax2 = plt.subplot(3, 3, 2)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=df['segment'],
                          cmap='viridis', alpha=0.6, s=50)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_title('Customer Segments (PCA)', fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Segment')

    # 3. Age vs Income
    ax3 = plt.subplot(3, 3, 3)
    for segment in sorted(df['segment'].unique()):
        segment_df = df[df['segment'] == segment]
        ax3.scatter(segment_df['age'], segment_df['income'], label=f'Segment {segment}', alpha=0.6)
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Income ($)')
    ax3.set_title('Age vs Income by Segment', fontweight='bold')
    ax3.legend()

    # 4. Spending Score vs Frequency
    ax4 = plt.subplot(3, 3, 4)
    for segment in sorted(df['segment'].unique()):
        segment_df = df[df['segment'] == segment]
        ax4.scatter(segment_df['spending_score'], segment_df['frequency'],
                   label=f'Segment {segment}', alpha=0.6, s=50)
    ax4.set_xlabel('Spending Score')
    ax4.set_ylabel('Purchase Frequency')
    ax4.set_title('Spending Score vs Frequency', fontweight='bold')
    ax4.legend()

    # 5. Average transaction by segment
    ax5 = plt.subplot(3, 3, 5)
    avg_by_segment = df.groupby('segment')['avg_transaction'].mean().sort_values()
    avg_by_segment.plot(kind='barh', ax=ax5, color='coral')
    ax5.set_xlabel('Average Transaction ($)')
    ax5.set_title('Avg Transaction by Segment', fontweight='bold')

    # 6. Customer LTV by segment
    ax6 = plt.subplot(3, 3, 6)
    ltv_by_segment = df.groupby('segment')['customer_ltv'].mean().sort_values()
    ltv_by_segment.plot(kind='bar', ax=ax6, color='steelblue')
    ax6.set_xlabel('Segment')
    ax6.set_ylabel('Customer LTV ($)')
    ax6.set_title('Customer Lifetime Value by Segment', fontweight='bold')
    ax6.tick_params(axis='x', rotation=0)

    # 7. Segment size distribution
    ax7 = plt.subplot(3, 3, 7)
    segment_counts = df['segment'].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, len(segment_counts)))
    ax7.pie(segment_counts, labels=[f'Seg {i}' for i in segment_counts.index],
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax7.set_title('Segment Distribution', fontweight='bold')

    # 8. Tenure vs Total Spending
    ax8 = plt.subplot(3, 3, 8)
    for segment in sorted(df['segment'].unique()):
        segment_df = df[df['segment'] == segment]
        ax8.scatter(segment_df['tenure_months'], segment_df['total_spending'],
                   label=f'Segment {segment}', alpha=0.6, s=50)
    ax8.set_xlabel('Tenure (months)')
    ax8.set_ylabel('Total Spending ($)')
    ax8.set_title('Tenure vs Total Spending', fontweight='bold')
    ax8.legend()

    # 9. Box plot of spending by segment
    ax9 = plt.subplot(3, 3, 9)
    df.boxplot(column='total_spending', by='segment', ax=ax9)
    ax9.set_xlabel('Segment')
    ax9.set_ylabel('Total Spending ($)')
    ax9.set_title('Spending Distribution by Segment', fontweight='bold')
    plt.suptitle('')  # Remove default title

    plt.tight_layout()
    print("Visualizations created successfully!")
    print("Close the plot window to continue...")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PROJECT")
    print("=" * 60)

    # 1. Generate customer data
    df = generate_customer_data(n_customers=500)

    # 2. Explore data
    explore_data(df)

    # 3. Preprocess and engineer features
    df = preprocess_data(df)

    # 4. Prepare features for clustering
    X_scaled, feature_cols, scaler = prepare_features_for_clustering(df)

    # 5. Find optimal number of clusters
    inertias, k_range, optimal_k = find_optimal_clusters(X_scaled, max_k=8)

    # 6. Perform clustering with optimal k
    labels, kmeans = perform_clustering(X_scaled, n_clusters=optimal_k)

    # 7. Analyze segments
    df = analyze_segments(df, labels)

    # 8. Name segments
    df, segment_profiles = name_segments(df)

    # 9. Visualize segments
    visualize_segments(df, X_scaled, inertias, k_range)

    # 10. Generate insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("=" * 60)

    for segment, name in segment_profiles.items():
        segment_df = df[df['segment'] == segment]
        print(f"\n{name} (Segment {segment}):")
        print(f"  Size: {len(segment_df)} customers ({len(segment_df)/len(df)*100:.1f}%)")
        print(f"  Avg Spending: ${segment_df['total_spending'].mean():.2f}")
        print(f"  Avg LTV: ${segment_df['customer_ltv'].mean():.2f}")
        print(f"  Avg Frequency: {segment_df['frequency'].mean():.1f} purchases")

        # Recommendation based on segment
        if "Premium" in name:
            print("  Strategy: VIP treatment, loyalty rewards, exclusive offers")
        elif "Budget" in name:
            print("  Strategy: Discount promotions, value bundles, referral incentives")
        elif "Young" in name:
            print("  Strategy: Digital marketing, trendy products, social media engagement")
        else:
            print("  Strategy: Re-engagement campaigns, personalized offers, feedback surveys")

    print("\n" + "=" * 60)
    print("Segmentation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
