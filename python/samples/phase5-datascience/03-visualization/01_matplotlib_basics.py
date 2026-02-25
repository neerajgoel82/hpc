"""
Matplotlib Basics
=================
Creating line plots, scatter plots, and bar charts with Matplotlib.

Topics:
- Line plots
- Scatter plots
- Bar charts
- Multiple plots
- Basic customization

Run: python 01_matplotlib_basics.py
Note: Plots will be displayed or can be saved to files
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Matplotlib Basics")
    print("=" * 60)

    # Set style for better-looking plots
    plt.style.use('default')

    # 1. Line plots
    print("\n1. Line Plots")
    print("-" * 40)

    # Create data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Simple Sine Wave')
    plt.grid(True)

    print("Created: Simple sine wave line plot")
    print(f"  Data points: {len(x)}")
    print(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")

    # Uncomment to display: plt.show()
    plt.savefig('/tmp/01_line_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/01_line_plot.png")

    # 2. Multiple lines
    print("\n2. Multiple Lines on Same Plot")
    print("-" * 40)

    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.cos(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label='sin(x)', linewidth=2)
    plt.plot(x, y2, label='cos(x)', linewidth=2)
    plt.plot(x, y3, label='sin(x)*cos(x)', linewidth=2, linestyle='--')

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Multiple Trigonometric Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    print("Created: Multiple line plot")
    print("  Lines: sin(x), cos(x), sin(x)*cos(x)")
    print("  Legend: Enabled")

    plt.savefig('/tmp/02_multiple_lines.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/02_multiple_lines.png")

    # 3. Scatter plots
    print("\n3. Scatter Plots")
    print("-" * 40)

    np.random.seed(42)
    n = 50
    x = np.random.randn(n)
    y = 2 * x + np.random.randn(n) * 0.5

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, s=100)
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Scatter Plot with Correlation')
    plt.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r--', linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    plt.legend()

    print("Created: Scatter plot with trend line")
    print(f"  Points: {n}")
    print(f"  Correlation: {np.corrcoef(x, y)[0, 1]:.3f}")
    print(f"  Trend line: y={z[0]:.2f}x+{z[1]:.2f}")

    plt.savefig('/tmp/03_scatter_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/03_scatter_plot.png")

    # 4. Colored scatter plot
    print("\n4. Colored Scatter Plot")
    print("-" * 40)

    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y = np.random.randn(n)
    colors = np.random.rand(n)
    sizes = 1000 * np.random.rand(n)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')
    plt.colorbar(scatter, label='Color value')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Scatter Plot with Color and Size Variations')
    plt.grid(True, alpha=0.3)

    print("Created: Colored scatter plot")
    print(f"  Points: {n}")
    print(f"  Color map: viridis")
    print(f"  Size range: [{sizes.min():.0f}, {sizes.max():.0f}]")

    plt.savefig('/tmp/04_colored_scatter.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/04_colored_scatter.png")

    # 5. Bar charts
    print("\n5. Bar Charts")
    print("-" * 40)

    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color='skyblue', edgecolor='navy')
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title('Simple Bar Chart')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + 2, str(v), ha='center', va='bottom')

    print("Created: Bar chart")
    print(f"  Categories: {len(categories)}")
    print(f"  Values: {values}")
    print(f"  Max value: {max(values)}")

    plt.savefig('/tmp/05_bar_chart.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/05_bar_chart.png")

    # 6. Grouped bar chart
    print("\n6. Grouped Bar Chart")
    print("-" * 40)

    categories = ['Q1', 'Q2', 'Q3', 'Q4']
    sales_2023 = [150, 180, 200, 190]
    sales_2024 = [160, 195, 215, 205]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, sales_2023, width, label='2023', color='steelblue')
    plt.bar(x + width/2, sales_2024, width, label='2024', color='coral')

    plt.xlabel('Quarter')
    plt.ylabel('Sales (in thousands)')
    plt.title('Quarterly Sales Comparison')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    print("Created: Grouped bar chart")
    print(f"  Quarters: {len(categories)}")
    print(f"  2023 total: {sum(sales_2023)}")
    print(f"  2024 total: {sum(sales_2024)}")
    print(f"  Growth: {((sum(sales_2024) / sum(sales_2023)) - 1) * 100:.1f}%")

    plt.savefig('/tmp/06_grouped_bar.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/06_grouped_bar.png")

    # 7. Horizontal bar chart
    print("\n7. Horizontal Bar Chart")
    print("-" * 40)

    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    revenue = [45000, 38000, 52000, 41000, 48000]

    plt.figure(figsize=(10, 6))
    plt.barh(products, revenue, color='lightgreen', edgecolor='darkgreen')
    plt.xlabel('Revenue ($)')
    plt.ylabel('Product')
    plt.title('Product Revenue Comparison')
    plt.grid(True, alpha=0.3, axis='x')

    print("Created: Horizontal bar chart")
    print(f"  Products: {len(products)}")
    print(f"  Total revenue: ${sum(revenue):,}")
    print(f"  Top product: {products[revenue.index(max(revenue))]} (${max(revenue):,})")

    plt.savefig('/tmp/07_horizontal_bar.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/07_horizontal_bar.png")

    # 8. Pie chart
    print("\n8. Pie Chart")
    print("-" * 40)

    labels = ['Python', 'Java', 'JavaScript', 'C++', 'Others']
    sizes = [35, 25, 20, 12, 8]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    explode = (0.1, 0, 0, 0, 0)  # Explode first slice

    plt.figure(figsize=(10, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Programming Language Usage')
    plt.axis('equal')

    print("Created: Pie chart")
    print(f"  Categories: {len(labels)}")
    print(f"  Sizes: {sizes}")
    print(f"  Largest slice: {labels[sizes.index(max(sizes))]} ({max(sizes)}%)")

    plt.savefig('/tmp/08_pie_chart.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/08_pie_chart.png")

    # 9. Histogram
    print("\n9. Histogram")
    print("-" * 40)

    np.random.seed(42)
    data = np.random.randn(1000)

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Normal Distribution')
    plt.grid(True, alpha=0.3, axis='y')

    # Add statistics
    plt.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
    plt.axvline(data.mean() + data.std(), color='orange', linestyle='--', linewidth=1, label=f'±1 Std')
    plt.axvline(data.mean() - data.std(), color='orange', linestyle='--', linewidth=1)
    plt.legend()

    print("Created: Histogram")
    print(f"  Data points: {len(data)}")
    print(f"  Mean: {data.mean():.3f}")
    print(f"  Std: {data.std():.3f}")
    print(f"  Bins: 30")

    plt.savefig('/tmp/09_histogram.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/09_histogram.png")

    print("\n" + "=" * 60)
    print("Summary:")
    print("Created 9 different plot types demonstrating Matplotlib basics")
    print("All plots saved to /tmp/ directory")
    print("\nTo display plots interactively, uncomment plt.show() calls")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create line plot with y = x^2 for x in [0, 10]")
    print("2. Make scatter plot comparing two random variables with 100 points")
    print("3. Create bar chart showing monthly sales data (12 months)")
    print("4. Plot histogram of 10000 random numbers from uniform distribution")
    print("=" * 60)

if __name__ == "__main__":
    main()
