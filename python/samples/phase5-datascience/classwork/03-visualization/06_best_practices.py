"""
Visualization Best Practices
=============================
Effective data visualization principles and common mistakes to avoid.

Topics:
- Choosing the right chart type
- Color selection and accessibility
- Clear labeling and titles
- Avoiding misleading visualizations
- Design principles

Run: python 06_best_practices.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Visualization Best Practices")
    print("=" * 60)

    # 1. Choosing the right chart type
    print("\n1. Choosing the Right Chart Type")
    print("-" * 40)

    print("\nGuidelines:")
    print("  COMPARISON: Bar chart, grouped bar chart")
    print("  DISTRIBUTION: Histogram, box plot, violin plot")
    print("  COMPOSITION: Pie chart, stacked bar chart")
    print("  RELATIONSHIP: Scatter plot, line plot")
    print("  TREND OVER TIME: Line chart, area chart")

    # Example: Good vs bad chart choice
    categories = ['Q1', 'Q2', 'Q3', 'Q4']
    values = [45, 52, 48, 58]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Good: Bar chart for comparison
    ax1.bar(categories, values, color='steelblue')
    ax1.set_title('GOOD: Bar Chart for Comparison', fontweight='bold', color='green')
    ax1.set_ylabel('Sales ($M)')
    ax1.grid(True, alpha=0.3, axis='y')

    # Bad: Line chart for categories
    ax2.plot(categories, values, 'o-', linewidth=2, markersize=8)
    ax2.set_title('BAD: Line Implies Trend Between Categories', fontweight='bold', color='red')
    ax2.set_ylabel('Sales ($M)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/best_01_chart_choice.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved comparison: /tmp/best_01_chart_choice.png")

    # 2. Start y-axis at zero
    print("\n2. Y-Axis Starting Point")
    print("-" * 40)

    data = [100, 102, 101, 103, 105]
    x = ['Jan', 'Feb', 'Mar', 'Apr', 'May']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Misleading: Y-axis doesn't start at zero
    ax1.plot(x, data, 'o-', linewidth=2, markersize=8)
    ax1.set_ylim(99, 106)
    ax1.set_title('MISLEADING: Exaggerates Change', fontweight='bold', color='red')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)

    # Correct: Y-axis starts at zero
    ax2.plot(x, data, 'o-', linewidth=2, markersize=8)
    ax2.set_ylim(0, 120)
    ax2.set_title('CORRECT: Shows True Scale', fontweight='bold', color='green')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/best_02_yaxis.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Rule: Start bar charts and area charts at zero")
    print("  Exception: Line charts can use appropriate range")
    print("  Saved: /tmp/best_02_yaxis.png")

    # 3. Color accessibility
    print("\n3. Color Accessibility")
    print("-" * 40)

    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 32, 48]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bad: Similar colors hard to distinguish
    bad_colors = ['#FF0000', '#FF3333', '#FF6666', '#FF9999', '#FFCCCC']
    ax1.bar(categories, values, color=bad_colors)
    ax1.set_title('BAD: Similar Colors', fontweight='bold', color='red')
    ax1.set_ylabel('Value')

    # Good: Distinct colors
    good_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ax2.bar(categories, values, color=good_colors)
    ax2.set_title('GOOD: Distinct Colors', fontweight='bold', color='green')
    ax2.set_ylabel('Value')

    plt.tight_layout()
    plt.savefig('/tmp/best_03_colors.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Guidelines:")
    print("    - Use colorblind-friendly palettes")
    print("    - Ensure sufficient contrast")
    print("    - Don't rely solely on color to convey information")
    print("  Saved: /tmp/best_03_colors.png")

    # 4. Clear labeling
    print("\n4. Clear Labeling and Titles")
    print("-" * 40)

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bad: No labels
    ax1.plot(x, y)
    ax1.set_title('BAD: Missing Information', fontweight='bold', color='red')

    # Good: Complete labels
    ax2.plot(x, y, linewidth=2)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Amplitude (volts)', fontsize=11)
    ax2.set_title('GOOD: Signal Amplitude Over Time\nMeasured on 2024-01-15',
                 fontweight='bold', color='green', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/best_04_labels.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Always include:")
    print("    - Descriptive title (what + when/where if relevant)")
    print("    - Axis labels with units")
    print("    - Legend if multiple series")
    print("    - Data source if applicable")
    print("  Saved: /tmp/best_04_labels.png")

    # 5. Avoid 3D when 2D suffices
    print("\n5. Avoid Unnecessary 3D")
    print("-" * 40)

    sizes = [30, 25, 20, 15, 10]
    labels = ['A', 'B', 'C', 'D', 'E']

    fig = plt.figure(figsize=(14, 5))

    # Bad: 3D pie chart (distortion)
    ax1 = fig.add_subplot(121, projection='3d' if False else None)  # 3D disabled for simplicity
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
           shadow=True, explode=[0.1, 0, 0, 0, 0])
    ax1.set_title('BAD: 3D/Shadow Distorts Perception', fontweight='bold', color='red')

    # Good: Simple 2D
    ax2 = fig.add_subplot(122)
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('GOOD: Clear 2D Representation', fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig('/tmp/best_05_3d.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Rule: Use 3D only when you have 3 dimensions of data")
    print("  Problem: 3D adds visual clutter and distorts perception")
    print("  Saved: /tmp/best_05_3d.png")

    # 6. Data-ink ratio
    print("\n6. Maximize Data-Ink Ratio")
    print("-" * 40)

    data = [45, 52, 48, 58, 55]
    x = ['2020', '2021', '2022', '2023', '2024']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bad: Too much decoration
    ax1.bar(x, data, color='red', edgecolor='black', linewidth=2)
    ax1.set_facecolor('#EEEEEE')
    ax1.grid(True, linewidth=2)
    ax1.set_title('BAD: Too Much Decoration', fontweight='bold', color='red',
                 bbox=dict(boxstyle='round', facecolor='yellow'))
    ax1.set_ylabel('Sales', fontsize=14, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='lightblue'))

    # Good: Clean and simple
    ax2.bar(x, data, color='steelblue', alpha=0.8)
    ax2.set_title('GOOD: Clean and Focused', fontweight='bold', color='green')
    ax2.set_ylabel('Sales ($M)')
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('/tmp/best_06_ink_ratio.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Principle: Remove non-essential elements")
    print("    - Reduce chartjunk (unnecessary decorations)")
    print("    - Simplify gridlines")
    print("    - Remove unnecessary borders")
    print("  Saved: /tmp/best_06_ink_ratio.png")

    # 7. Sorting data meaningfully
    print("\n7. Sort Data Meaningfully")
    print("-" * 40)

    categories = ['Product E', 'Product A', 'Product C', 'Product B', 'Product D']
    values = [35, 65, 45, 55, 40]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bad: Random order
    ax1.barh(categories, values, color='coral')
    ax1.set_title('BAD: Random Order', fontweight='bold', color='red')
    ax1.set_xlabel('Sales')

    # Good: Sorted by value
    sorted_data = sorted(zip(categories, values), key=lambda x: x[1], reverse=True)
    sorted_cats, sorted_vals = zip(*sorted_data)
    ax2.barh(sorted_cats, sorted_vals, color='steelblue')
    ax2.set_title('GOOD: Sorted by Value', fontweight='bold', color='green')
    ax2.set_xlabel('Sales')

    plt.tight_layout()
    plt.savefig('/tmp/best_07_sorting.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Rule: Sort bars by value (unless natural order exists)")
    print("  Exception: Time-based data should maintain chronological order")
    print("  Saved: /tmp/best_07_sorting.png")

    # 8. Summary of principles
    print("\n8. Summary of Best Practices")
    print("-" * 40)

    principles = """
Key Principles:
1. Choose the right chart type for your data
2. Start y-axis at zero for bar/area charts
3. Use colorblind-friendly, distinct colors
4. Label everything clearly (title, axes, units)
5. Avoid 3D when 2D suffices
6. Maximize data-ink ratio (remove clutter)
7. Sort data meaningfully
8. Keep it simple and focused
9. Be honest with data representation
10. Consider your audience

Common Mistakes to Avoid:
- Truncated y-axis on bar charts
- Too many categories in pie charts
- Dual y-axes with different scales
- Cherry-picking data ranges
- Using area to represent 1D data
- Overusing colors
- Missing units or context
"""

    print(principles)

    print("\n" + "=" * 60)
    print("Summary:")
    print("Demonstrated visualization best practices and common pitfalls")
    print("All comparison plots saved to /tmp/ directory")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Find a misleading chart online and explain why")
    print("2. Create same data using good vs bad practices")
    print("3. Redesign a cluttered chart to follow best practices")
    print("4. Test your charts with a colorblind simulator")
    print("=" * 60)

if __name__ == "__main__":
    main()
