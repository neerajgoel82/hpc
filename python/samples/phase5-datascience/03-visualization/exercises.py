"""
Visualization Exercises
=======================
Practice problems covering all visualization concepts.

Topics covered:
- Matplotlib basics
- Statistical plots
- Customization
- Subplots and layouts
- Best practices

Run: python exercises.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def exercise_1():
    """Basic line and scatter plots"""
    print("\nExercise 1: Basic Line and Scatter Plots")
    print("-" * 40)

    x = np.linspace(0, 10, 100)
    y1 = x ** 2
    y2 = x ** 1.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Line plot
    ax1.plot(x, y1, 'b-', linewidth=2, label='y = x²')
    ax1.plot(x, y2, 'r--', linewidth=2, label='y = x^1.5')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Polynomial Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot
    np.random.seed(42)
    x_scatter = np.random.rand(100) * 10
    y_scatter = x_scatter ** 2 + np.random.randn(100) * 10
    ax2.scatter(x_scatter, y_scatter, alpha=0.6, s=50)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Scatter Plot with Noise')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/ex_01_basic.png')
    plt.close()
    print("Created: Basic line and scatter plots")
    print("  Saved: /tmp/ex_01_basic.png")

def exercise_2():
    """Bar charts with comparisons"""
    print("\nExercise 2: Monthly Sales Bar Chart")
    print("-" * 40)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sales_2023 = [45, 52, 48, 58, 62, 65, 68, 72, 70, 75, 78, 82]
    sales_2024 = [48, 55, 52, 62, 68, 72, 75, 80, 78, 83, 85, 90]

    x = np.arange(len(months))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, sales_2023, width, label='2023', color='steelblue')
    ax.bar(x + width/2, sales_2024, width, label='2024', color='coral')

    ax.set_xlabel('Month')
    ax.set_ylabel('Sales ($K)')
    ax.set_title('Monthly Sales Comparison: 2023 vs 2024')
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/tmp/ex_02_bars.png')
    plt.close()

    print(f"Created: Monthly sales comparison")
    print(f"  2023 total: ${sum(sales_2023)}K")
    print(f"  2024 total: ${sum(sales_2024)}K")
    print(f"  Growth: {((sum(sales_2024) / sum(sales_2023)) - 1) * 100:.1f}%")
    print("  Saved: /tmp/ex_02_bars.png")

def exercise_3():
    """Histogram with distributions"""
    print("\nExercise 3: Distribution Analysis")
    print("-" * 40)

    np.random.seed(42)
    normal_data = np.random.randn(10000)
    uniform_data = np.random.uniform(-3, 3, 10000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Normal distribution
    ax1.hist(normal_data, bins=50, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.axvline(normal_data.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {normal_data.mean():.2f}')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Normal Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Uniform distribution
    ax2.hist(uniform_data, bins=50, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax2.axvline(uniform_data.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {uniform_data.mean():.2f}')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Uniform Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/tmp/ex_03_histograms.png')
    plt.close()

    print("Created: Distribution comparison")
    print(f"  Normal - Mean: {normal_data.mean():.3f}, Std: {normal_data.std():.3f}")
    print(f"  Uniform - Mean: {uniform_data.mean():.3f}, Std: {uniform_data.std():.3f}")
    print("  Saved: /tmp/ex_03_histograms.png")

def exercise_4():
    """Box plots for group comparison"""
    print("\nExercise 4: Box Plot Comparison")
    print("-" * 40)

    np.random.seed(42)
    data_groups = {
        'Group A': np.random.normal(100, 15, 100),
        'Group B': np.random.normal(110, 20, 100),
        'Group C': np.random.normal(105, 10, 100),
        'Group D': np.random.normal(115, 18, 100)
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data_groups.values(), labels=data_groups.keys())
    ax.set_ylabel('Value')
    ax.set_title('Distribution Comparison Across Groups')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/tmp/ex_04_boxplot.png')
    plt.close()

    print("Created: Box plot comparison")
    for name, data in data_groups.items():
        print(f"  {name}: Mean={data.mean():.1f}, Std={data.std():.1f}")
    print("  Saved: /tmp/ex_04_boxplot.png")

def exercise_5():
    """Subplot grid with different plot types"""
    print("\nExercise 5: 2x3 Subplot Grid")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    np.random.seed(42)
    x = np.linspace(0, 10, 100)

    # Plot 1: Sine
    axes[0, 0].plot(x, np.sin(x), 'b-')
    axes[0, 0].set_title('Sine Wave')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Cosine
    axes[0, 1].plot(x, np.cos(x), 'r-')
    axes[0, 1].set_title('Cosine Wave')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Exponential
    axes[0, 2].plot(x, np.exp(x/5), 'g-')
    axes[0, 2].set_title('Exponential')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Scatter
    axes[1, 0].scatter(np.random.randn(50), np.random.randn(50), alpha=0.6)
    axes[1, 0].set_title('Random Scatter')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Bar
    axes[1, 1].bar(['A', 'B', 'C', 'D'], [3, 7, 5, 9])
    axes[1, 1].set_title('Bar Chart')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Plot 6: Histogram
    axes[1, 2].hist(np.random.randn(1000), bins=30, color='purple', alpha=0.7)
    axes[1, 2].set_title('Histogram')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Multiple Plot Types in 2x3 Grid', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/ex_05_subplots.png')
    plt.close()

    print("Created: 2x3 subplot grid with 6 different plot types")
    print("  Saved: /tmp/ex_05_subplots.png")

def exercise_6():
    """Customized plot with annotations"""
    print("\nExercise 6: Customized Plot with Annotations")
    print("-" * 40)

    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x/10)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, 'b-', linewidth=2.5, label='Damped Sine Wave')

    # Find and annotate maximum
    max_idx = np.argmax(y)
    ax.plot(x[max_idx], y[max_idx], 'ro', markersize=10)
    ax.annotate('Maximum Point',
               xy=(x[max_idx], y[max_idx]),
               xytext=(x[max_idx] + 2, y[max_idx] + 0.2),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=12,
               color='red',
               fontweight='bold')

    # Add zero crossing annotation
    zero_cross = np.where(np.diff(np.sign(y)))[0]
    if len(zero_cross) > 0:
        idx = zero_cross[1]
        ax.axvline(x[idx], color='gray', linestyle='--', alpha=0.5)
        ax.text(x[idx], -0.3, 'Zero Crossing', ha='center', fontsize=10)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Damped Sine Wave with Annotations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('/tmp/ex_06_annotated.png')
    plt.close()

    print("Created: Annotated plot with custom styling")
    print(f"  Maximum at x={x[max_idx]:.2f}, y={y[max_idx]:.3f}")
    print("  Saved: /tmp/ex_06_annotated.png")

def exercise_7():
    """Time series visualization"""
    print("\nExercise 7: Time Series Visualization")
    print("-" * 40)

    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    np.random.seed(42)

    # Generate time series with trend and seasonality
    trend = np.linspace(100, 150, 365)
    seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, 365))
    noise = np.random.randn(365) * 5
    values = trend + seasonal + noise

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, values, 'b-', alpha=0.7, linewidth=1, label='Actual')

    # Add moving average
    window = 30
    ma = pd.Series(values).rolling(window=window).mean()
    ax.plot(dates, ma, 'r-', linewidth=2, label=f'{window}-Day Moving Average')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Time Series with Trend and Seasonality', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('/tmp/ex_07_timeseries.png')
    plt.close()

    print("Created: Time series visualization")
    print(f"  Period: {dates[0].date()} to {dates[-1].date()}")
    print(f"  Mean value: {values.mean():.2f}")
    print("  Saved: /tmp/ex_07_timeseries.png")

def exercise_8():
    """Correlation heatmap"""
    print("\nExercise 8: Correlation Heatmap")
    print("-" * 40)

    np.random.seed(42)

    # Create correlated data
    n = 100
    df = pd.DataFrame({
        'Variable A': np.random.randn(n),
        'Variable B': np.random.randn(n),
        'Variable C': np.random.randn(n),
        'Variable D': np.random.randn(n)
    })

    # Add correlations
    df['Variable B'] = df['Variable A'] * 0.8 + np.random.randn(n) * 0.2
    df['Variable C'] = df['Variable A'] * -0.6 + np.random.randn(n) * 0.4
    df['Variable D'] = df['Variable B'] * 0.5 + np.random.randn(n) * 0.5

    # Calculate correlation
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=11)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)

    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10)

    ax.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/ex_08_heatmap.png')
    plt.close()

    print("Created: Correlation heatmap")
    print("\nCorrelation matrix:")
    print(corr.round(2))
    print("  Saved: /tmp/ex_08_heatmap.png")

def main():
    print("=" * 60)
    print("Visualization Exercises")
    print("=" * 60)

    exercises = [
        exercise_1,
        exercise_2,
        exercise_3,
        exercise_4,
        exercise_5,
        exercise_6,
        exercise_7,
        exercise_8,
    ]

    for exercise in exercises:
        try:
            exercise()
        except Exception as e:
            print(f"\nError in {exercise.__name__}: {e}")

    print("\n" + "=" * 60)
    print("All exercises completed!")
    print("All plots saved to /tmp/ directory")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Additional Practice:")
    print("1. Create a pie chart showing category distribution")
    print("2. Make a stacked bar chart showing composition over time")
    print("3. Create a 3D surface plot of z = sin(x) * cos(y)")
    print("4. Design a dashboard with 4-6 plots in custom layout")
    print("5. Build an interactive plot using Plotly")
    print("=" * 60)

if __name__ == "__main__":
    main()
