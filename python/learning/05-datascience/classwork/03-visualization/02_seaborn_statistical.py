"""
Seaborn Statistical Plots
==========================
Creating statistical visualizations with Seaborn.

Topics:
- Distribution plots (histograms, KDE)
- Box plots
- Violin plots
- Pair plots
- Heatmaps and correlation matrices

Run: python 02_seaborn_statistical.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Seaborn Statistical Plots")
    print("=" * 60)

    # Try to import seaborn
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        print("Seaborn version:", sns.__version__)
    except ImportError:
        print("\nWARNING: Seaborn not installed")
        print("Install with: pip install seaborn")
        print("\nContinuing with matplotlib-only examples...")
        sns = None

    # 1. Distribution plot (histogram + KDE)
    print("\n1. Distribution Plot")
    print("-" * 40)

    np.random.seed(42)
    data = np.random.randn(1000)

    if sns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=True, bins=30, color='skyblue')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Distribution Plot with KDE')
        print("Created: Seaborn distribution plot with KDE")
    else:
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, color='skyblue', edgecolor='navy', alpha=0.7, density=True)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Distribution Plot (matplotlib)')
        print("Created: Matplotlib histogram (seaborn unavailable)")

    print(f"  Data points: {len(data)}")
    print(f"  Mean: {data.mean():.3f}")
    print(f"  Std: {data.std():.3f}")

    plt.savefig('/tmp/seaborn_01_dist.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/seaborn_01_dist.png")

    # 2. Box plot
    print("\n2. Box Plot")
    print("-" * 40)

    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'Group': np.repeat(['A', 'B', 'C', 'D'], 50),
        'Value': np.concatenate([
            np.random.normal(100, 10, 50),
            np.random.normal(110, 15, 50),
            np.random.normal(105, 12, 50),
            np.random.normal(115, 10, 50)
        ])
    })

    plt.figure(figsize=(10, 6))
    if sns:
        sns.boxplot(data=df, x='Group', y='Value', palette='Set2')
        plt.title('Box Plot by Group (Seaborn)')
        print("Created: Seaborn box plot")
    else:
        df.boxplot(column='Value', by='Group', figsize=(10, 6))
        plt.title('Box Plot by Group (Pandas)')
        plt.suptitle('')
        print("Created: Pandas box plot (seaborn unavailable)")

    print(f"  Groups: {df['Group'].nunique()}")
    print(f"  Total observations: {len(df)}")
    print("\n  Group statistics:")
    print(df.groupby('Group')['Value'].describe()[['mean', 'std', 'min', 'max']].round(2))

    plt.savefig('/tmp/seaborn_02_boxplot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/seaborn_02_boxplot.png")

    # 3. Violin plot
    print("\n3. Violin Plot")
    print("-" * 40)

    if sns:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x='Group', y='Value', palette='muted')
        plt.title('Violin Plot by Group')
        plt.ylabel('Value')

        print("Created: Seaborn violin plot")
        print("  Shows distribution shape for each group")
        print("  Wider sections = more data points at that value")

        plt.savefig('/tmp/seaborn_03_violin.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  Saved to: /tmp/seaborn_03_violin.png")
    else:
        print("Skipped: Violin plots require seaborn")

    # 4. Swarm plot (categorical scatter)
    print("\n4. Swarm Plot")
    print("-" * 40)

    if sns:
        # Use smaller dataset for swarm plot
        df_small = df.sample(100, random_state=42)

        plt.figure(figsize=(10, 6))
        sns.swarmplot(data=df_small, x='Group', y='Value', palette='Set1', size=5)
        plt.title('Swarm Plot (All Points Visible)')

        print("Created: Seaborn swarm plot")
        print(f"  Points plotted: {len(df_small)}")
        print("  Each point represents one observation")

        plt.savefig('/tmp/seaborn_04_swarm.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  Saved to: /tmp/seaborn_04_swarm.png")
    else:
        print("Skipped: Swarm plots require seaborn")

    # 5. Pair plot
    print("\n5. Pair Plot (Scatterplot Matrix)")
    print("-" * 40)

    if sns:
        # Create multi-variable dataset
        np.random.seed(42)
        iris_like = pd.DataFrame({
            'Feature1': np.random.randn(100),
            'Feature2': np.random.randn(100) + 0.5,
            'Feature3': np.random.randn(100) * 1.5,
            'Species': np.random.choice(['A', 'B', 'C'], 100)
        })

        pair_plot = sns.pairplot(iris_like, hue='Species', palette='husl', height=2)
        pair_plot.fig.suptitle('Pair Plot with Species Coloring', y=1.02)

        print("Created: Seaborn pair plot")
        print(f"  Features: {iris_like.shape[1] - 1}")
        print(f"  Samples: {len(iris_like)}")
        print("  Shows all pairwise relationships")

        plt.savefig('/tmp/seaborn_05_pairplot.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  Saved to: /tmp/seaborn_05_pairplot.png")
    else:
        print("Skipped: Pair plots require seaborn")

    # 6. Heatmap
    print("\n6. Heatmap")
    print("-" * 40)

    # Create correlation matrix
    np.random.seed(42)
    data_matrix = np.random.randn(10, 12)
    df_heat = pd.DataFrame(data_matrix,
                          columns=[f'Col{i}' for i in range(12)],
                          index=[f'Row{i}' for i in range(10)])

    plt.figure(figsize=(12, 8))
    if sns:
        sns.heatmap(df_heat, annot=False, cmap='coolwarm', center=0,
                   cbar_kws={'label': 'Value'})
        plt.title('Heatmap Example')
        print("Created: Seaborn heatmap")
    else:
        plt.imshow(df_heat, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Value')
        plt.title('Heatmap (matplotlib)')
        print("Created: Matplotlib heatmap (seaborn unavailable)")

    print(f"  Shape: {df_heat.shape}")
    print(f"  Value range: [{df_heat.values.min():.2f}, {df_heat.values.max():.2f}]")

    plt.savefig('/tmp/seaborn_06_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/seaborn_06_heatmap.png")

    # 7. Correlation heatmap
    print("\n7. Correlation Heatmap")
    print("-" * 40)

    # Create correlated data
    np.random.seed(42)
    df_corr = pd.DataFrame({
        'Height': np.random.normal(170, 10, 100),
        'Weight': np.random.normal(70, 15, 100),
        'Age': np.random.randint(18, 65, 100),
        'Income': np.random.normal(50000, 20000, 100)
    })

    # Add some correlations
    df_corr['Weight'] = df_corr['Height'] * 0.5 + np.random.normal(0, 5, 100)
    df_corr['Income'] = df_corr['Age'] * 500 + np.random.normal(0, 10000, 100)

    correlation = df_corr.corr()

    plt.figure(figsize=(10, 8))
    if sns:
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Matrix Heatmap')
        print("Created: Seaborn correlation heatmap")
    else:
        plt.imshow(correlation, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Correlation')
        plt.title('Correlation Matrix')
        print("Created: Matplotlib correlation heatmap (seaborn unavailable)")

    print("\nCorrelation matrix:")
    print(correlation.round(2))

    plt.savefig('/tmp/seaborn_07_correlation.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/seaborn_07_correlation.png")

    # 8. Count plot
    print("\n8. Count Plot (Categorical)")
    print("-" * 40)

    if sns:
        # Create categorical data
        categories = pd.DataFrame({
            'Category': np.random.choice(['A', 'B', 'C', 'D'], 200),
            'Type': np.random.choice(['X', 'Y'], 200)
        })

        plt.figure(figsize=(10, 6))
        sns.countplot(data=categories, x='Category', hue='Type', palette='pastel')
        plt.title('Count Plot by Category and Type')
        plt.ylabel('Count')

        print("Created: Seaborn count plot")
        print("\nCounts:")
        print(categories.groupby(['Category', 'Type']).size().unstack(fill_value=0))

        plt.savefig('/tmp/seaborn_08_countplot.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  Saved to: /tmp/seaborn_08_countplot.png")
    else:
        print("Skipped: Count plots require seaborn")

    # 9. Joint plot
    print("\n9. Joint Plot (Scatter + Distributions)")
    print("-" * 40)

    if sns:
        np.random.seed(42)
        x = np.random.randn(200)
        y = x * 2 + np.random.randn(200)

        joint_plot = sns.jointplot(x=x, y=y, kind='scatter', height=8)
        joint_plot.fig.suptitle('Joint Plot with Marginal Distributions', y=1.02)

        print("Created: Seaborn joint plot")
        print(f"  Points: {len(x)}")
        print(f"  Correlation: {np.corrcoef(x, y)[0, 1]:.3f}")
        print("  Shows scatter + marginal distributions")

        plt.savefig('/tmp/seaborn_09_jointplot.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  Saved to: /tmp/seaborn_09_jointplot.png")
    else:
        print("Skipped: Joint plots require seaborn")

    print("\n" + "=" * 60)
    print("Summary:")
    if sns:
        print("Created multiple Seaborn statistical plots")
        print("Seaborn provides beautiful statistical visualizations")
    else:
        print("Created basic plots using matplotlib")
        print("Install seaborn for enhanced statistical visualizations")
    print("\nAll plots saved to /tmp/ directory")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create box plot comparing 3 groups of random data")
    print("2. Make violin plot showing distribution differences")
    print("3. Create correlation heatmap for 5 variables")
    print("4. Generate pair plot for iris-like dataset with 4 features")
    print("=" * 60)

if __name__ == "__main__":
    main()
