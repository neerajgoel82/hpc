"""
Subplots and Layouts
====================
Creating multiple plots in a single figure with different layouts.

Topics:
- Basic subplots
- Grid layouts
- Nested subplots
- Shared axes
- Figure-level arrangements

Run: python 04_subplots_layouts.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Subplots and Layouts")
    print("=" * 60)

    # 1. Basic subplots (grid layout)
    print("\n1. Basic Subplots - 2x2 Grid")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x = np.linspace(0, 10, 100)

    # Plot 1: Sine
    axes[0, 0].plot(x, np.sin(x), 'b-')
    axes[0, 0].set_title('Sine Wave')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Cosine
    axes[0, 1].plot(x, np.cos(x), 'r-')
    axes[0, 1].set_title('Cosine Wave')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Exponential
    axes[1, 0].plot(x, np.exp(x/5), 'g-')
    axes[1, 0].set_title('Exponential')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Logarithm
    axes[1, 1].plot(x[1:], np.log(x[1:]), 'm-')
    axes[1, 1].set_title('Logarithm')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('2x2 Subplot Grid', fontsize=14, fontweight='bold')
    plt.tight_layout()

    print("Created: 2x2 grid with 4 different functions")
    print("  Each subplot has its own title and labels")

    plt.savefig('/tmp/subplots_01_grid.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/subplots_01_grid.png")

    # 2. Subplots with different sizes
    print("\n2. Subplots with Different Sizes")
    print("-" * 40)

    fig = plt.figure(figsize=(12, 8))

    # Large subplot on left
    ax1 = plt.subplot(1, 2, 1)
    x = np.linspace(0, 10, 100)
    ax1.plot(x, np.sin(x), 'b-', linewidth=2)
    ax1.set_title('Main Plot (Large)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Two smaller subplots on right
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(x, np.cos(x), 'r-')
    ax2.set_title('Secondary Plot 1')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 4)
    ax3.plot(x, np.tan(x), 'g-')
    ax3.set_title('Secondary Plot 2')
    ax3.set_ylim(-5, 5)
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Mixed Subplot Sizes', fontsize=14, fontweight='bold')
    plt.tight_layout()

    print("Created: Mixed-size subplots")
    print("  1 large plot + 2 smaller plots")

    plt.savefig('/tmp/subplots_02_mixed.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/subplots_02_mixed.png")

    # 3. Shared axes
    print("\n3. Shared Axes")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    x = np.linspace(0, 10, 100)

    # All plots share same x and y axes
    axes[0, 0].plot(x, np.sin(x))
    axes[0, 0].set_title('sin(x)')

    axes[0, 1].plot(x, np.sin(2*x))
    axes[0, 1].set_title('sin(2x)')

    axes[1, 0].plot(x, np.sin(3*x))
    axes[1, 0].set_title('sin(3x)')
    axes[1, 0].set_ylabel('Y axis')

    axes[1, 1].plot(x, np.sin(4*x))
    axes[1, 1].set_title('sin(4x)')
    axes[1, 1].set_xlabel('X axis')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.suptitle('Shared X and Y Axes', fontsize=14, fontweight='bold')
    plt.tight_layout()

    print("Created: Subplots with shared axes")
    print("  All subplots use same x and y scales")
    print("  Easier to compare across plots")

    plt.savefig('/tmp/subplots_03_shared.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/subplots_03_shared.png")

    # 4. GridSpec for complex layouts
    print("\n4. GridSpec for Complex Layouts")
    print("-" * 40)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 3, figure=fig)

    # Top spanning plot
    ax1 = fig.add_subplot(gs[0, :])
    x = np.linspace(0, 10, 100)
    ax1.plot(x, np.sin(x), 'b-', linewidth=2)
    ax1.set_title('Top Spanning Plot')
    ax1.grid(True, alpha=0.3)

    # Middle left
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x, np.cos(x), 'r-')
    ax2.set_title('Middle Left')
    ax2.grid(True, alpha=0.3)

    # Middle center + right
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.plot(x, np.sin(x) * np.cos(x), 'g-')
    ax3.set_title('Middle Spanning')
    ax3.grid(True, alpha=0.3)

    # Bottom plots
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(['A', 'B', 'C'], [3, 7, 5])
    ax4.set_title('Bottom Left')

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.scatter(np.random.randn(50), np.random.randn(50))
    ax5.set_title('Bottom Center')

    ax6 = fig.add_subplot(gs[2, 2])
    ax6.hist(np.random.randn(100), bins=20)
    ax6.set_title('Bottom Right')

    plt.suptitle('GridSpec Complex Layout', fontsize=14, fontweight='bold')
    plt.tight_layout()

    print("Created: Complex layout using GridSpec")
    print("  6 subplots with varying sizes and positions")
    print("  GridSpec allows flexible positioning")

    plt.savefig('/tmp/subplots_04_gridspec.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/subplots_04_gridspec.png")

    # 5. Nested subplots
    print("\n5. Nested Subplots (Inset)")
    print("-" * 40)

    fig, ax = plt.subplots(figsize=(10, 8))

    x = np.linspace(0, 10, 1000)
    y = np.sin(x)

    # Main plot
    ax.plot(x, y, 'b-', linewidth=2)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title('Main Plot with Inset')
    ax.grid(True, alpha=0.3)

    # Create inset (zoomed region)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="40%", height="40%", loc='upper right')

    # Plot zoomed region
    zoom_start, zoom_end = 100, 200
    axins.plot(x[zoom_start:zoom_end], y[zoom_start:zoom_end], 'r-', linewidth=2)
    axins.set_title('Zoomed Region', fontsize=10)
    axins.grid(True, alpha=0.3)

    # Mark the region
    ax.indicate_inset_zoom(axins, edgecolor="black")

    print("Created: Plot with inset (zoomed region)")
    print(f"  Zoomed region: indices {zoom_start}-{zoom_end}")

    plt.savefig('/tmp/subplots_05_inset.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/subplots_05_inset.png")

    # 6. Multiple plots with loop
    print("\n6. Creating Multiple Subplots in a Loop")
    print("-" * 40)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    x = np.linspace(0, 10, 100)

    for i, ax in enumerate(axes.flat):
        frequency = i + 1
        y = np.sin(frequency * x)
        ax.plot(x, y, linewidth=2)
        ax.set_title(f'sin({frequency}x)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)

    plt.suptitle('Multiple Sine Waves', fontsize=14, fontweight='bold')
    plt.tight_layout()

    print("Created: 3x3 grid of sine waves (frequencies 1-9)")
    print("  All created in a loop for efficiency")

    plt.savefig('/tmp/subplots_06_loop.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/subplots_06_loop.png")

    # 7. Subplots with different plot types
    print("\n7. Mixed Plot Types in Subplots")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    np.random.seed(42)

    # Line plot
    x = np.linspace(0, 10, 100)
    axes[0, 0].plot(x, np.sin(x), 'b-')
    axes[0, 0].set_title('Line Plot')

    # Scatter plot
    axes[0, 1].scatter(np.random.randn(50), np.random.randn(50))
    axes[0, 1].set_title('Scatter Plot')

    # Bar chart
    axes[0, 2].bar(['A', 'B', 'C', 'D'], [3, 7, 5, 6])
    axes[0, 2].set_title('Bar Chart')

    # Histogram
    axes[1, 0].hist(np.random.randn(1000), bins=30)
    axes[1, 0].set_title('Histogram')

    # Box plot
    axes[1, 1].boxplot([np.random.randn(100) + i for i in range(4)])
    axes[1, 1].set_title('Box Plot')

    # Heatmap
    data = np.random.rand(10, 10)
    im = axes[1, 2].imshow(data, cmap='viridis')
    axes[1, 2].set_title('Heatmap')
    plt.colorbar(im, ax=axes[1, 2])

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.suptitle('Different Plot Types', fontsize=14, fontweight='bold')
    plt.tight_layout()

    print("Created: 6 different plot types in one figure")
    print("  Line, scatter, bar, histogram, box, heatmap")

    plt.savefig('/tmp/subplots_07_mixed_types.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/subplots_07_mixed_types.png")

    print("\n" + "=" * 60)
    print("Summary:")
    print("Demonstrated various subplot layouts and arrangements")
    print("All plots saved to /tmp/ directory")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create 2x3 grid showing 6 different mathematical functions")
    print("2. Make layout with 1 large plot and 4 small plots around it")
    print("3. Create plot with 2 insets showing different zoomed regions")
    print("4. Use GridSpec to create custom asymmetric layout")
    print("=" * 60)

if __name__ == "__main__":
    main()
