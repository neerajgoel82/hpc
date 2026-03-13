"""
Plot Customization
==================
Customizing colors, styles, annotations, and legends in Matplotlib.

Topics:
- Colors and colormaps
- Line styles and markers
- Annotations and text
- Legends and labels
- Styles and themes

Run: python 03_customization.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Plot Customization")
    print("=" * 60)

    # 1. Colors
    print("\n1. Colors and Color Specifications")
    print("-" * 40)

    x = np.linspace(0, 10, 100)

    plt.figure(figsize=(12, 6))

    # Different color specifications
    plt.plot(x, np.sin(x), 'r', label='Red (letter)', linewidth=2)
    plt.plot(x, np.sin(x) + 1, color='blue', label='Blue (name)', linewidth=2)
    plt.plot(x, np.sin(x) + 2, color='#FF5733', label='Hex color', linewidth=2)
    plt.plot(x, np.sin(x) + 3, color=(0.2, 0.8, 0.2), label='RGB tuple', linewidth=2)

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Different Color Specifications')
    plt.legend()
    plt.grid(True, alpha=0.3)

    print("Color specification methods:")
    print("  - Single letter: 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'")
    print("  - Color names: 'red', 'blue', 'green', etc.")
    print("  - Hex codes: '#FF5733'")
    print("  - RGB tuples: (0.2, 0.8, 0.2)")

    plt.savefig('/tmp/custom_01_colors.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/custom_01_colors.png")

    # 2. Line styles and markers
    print("\n2. Line Styles and Markers")
    print("-" * 40)

    plt.figure(figsize=(12, 8))

    x = np.linspace(0, 10, 20)

    # Line styles
    plt.plot(x, x + 0, '-', label='Solid', linewidth=2)
    plt.plot(x, x + 1, '--', label='Dashed', linewidth=2)
    plt.plot(x, x + 2, '-.', label='Dash-dot', linewidth=2)
    plt.plot(x, x + 3, ':', label='Dotted', linewidth=2)

    # Markers
    plt.plot(x, x + 5, 'o-', label='Circle markers', markersize=8)
    plt.plot(x, x + 6, 's-', label='Square markers', markersize=8)
    plt.plot(x, x + 7, '^-', label='Triangle markers', markersize=8)
    plt.plot(x, x + 8, 'D-', label='Diamond markers', markersize=8)

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Line Styles and Markers')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    print("Line styles: '-', '--', '-.', ':'")
    print("Markers: 'o', 's', '^', 'v', 'D', '*', '+', 'x'")

    plt.savefig('/tmp/custom_02_styles.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/custom_02_styles.png")

    # 3. Annotations
    print("\n3. Annotations and Text")
    print("-" * 40)

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'b-', linewidth=2)

    # Annotate maximum
    max_idx = np.argmax(y)
    plt.annotate('Maximum',
                xy=(x[max_idx], y[max_idx]),
                xytext=(x[max_idx] + 1, y[max_idx] + 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12,
                color='red')

    # Annotate minimum
    min_idx = np.argmin(y)
    plt.annotate('Minimum',
                xy=(x[min_idx], y[min_idx]),
                xytext=(x[min_idx] + 1, y[min_idx] - 0.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12,
                color='green')

    # Add text box
    plt.text(7, 0, 'Sine Wave\ny = sin(x)',
            fontsize=14,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel('X axis', fontsize=12)
    plt.ylabel('Y axis', fontsize=12)
    plt.title('Annotations and Text', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    print("Added annotations:")
    print(f"  Maximum at x={x[max_idx]:.2f}, y={y[max_idx]:.2f}")
    print(f"  Minimum at x={x[min_idx]:.2f}, y={y[min_idx]:.2f}")

    plt.savefig('/tmp/custom_03_annotations.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/custom_03_annotations.png")

    # 4. Legend customization
    print("\n4. Legend Customization")
    print("-" * 40)

    x = np.linspace(0, 10, 100)

    plt.figure(figsize=(12, 6))
    plt.plot(x, np.sin(x), 'b-', label='sin(x)', linewidth=2)
    plt.plot(x, np.cos(x), 'r--', label='cos(x)', linewidth=2)
    plt.plot(x, np.sin(x) * np.cos(x), 'g-.', label='sin(x)*cos(x)', linewidth=2)

    # Customized legend
    plt.legend(loc='upper right',
              frameon=True,
              shadow=True,
              fancybox=True,
              fontsize=11,
              title='Functions',
              title_fontsize=12)

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Customized Legend')
    plt.grid(True, alpha=0.3)

    print("Legend customization options:")
    print("  - Location: 'upper right', 'lower left', 'center', etc.")
    print("  - Frame, shadow, fancy box options")
    print("  - Font sizes and title")

    plt.savefig('/tmp/custom_04_legend.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/custom_04_legend.png")

    # 5. Colormaps
    print("\n5. Colormaps")
    print("-" * 40)

    np.random.seed(42)
    data = np.random.rand(10, 10)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colormaps = ['viridis', 'plasma', 'coolwarm', 'RdYlBu', 'hot', 'seismic']

    for ax, cmap in zip(axes.flat, colormaps):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(f'Colormap: {cmap}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('Different Colormaps', fontsize=14, fontweight='bold')
    plt.tight_layout()

    print("Popular colormaps:")
    print("  - Sequential: 'viridis', 'plasma', 'inferno', 'magma'")
    print("  - Diverging: 'coolwarm', 'RdBu', 'seismic'")
    print("  - Qualitative: 'Set1', 'Set2', 'tab10'")

    plt.savefig('/tmp/custom_05_colormaps.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/custom_05_colormaps.png")

    # 6. Plot styles
    print("\n6. Plot Styles")
    print("-" * 40)

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    styles = ['default', 'ggplot', 'seaborn-v0_8', 'fivethirtyeight']
    available_styles = [s for s in styles if s in plt.style.available or s == 'default']

    if len(available_styles) < len(styles):
        print(f"Note: Some styles not available. Using: {available_styles}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, style in zip(axes.flat, available_styles):
        with plt.style.context(style if style != 'default' else 'default'):
            ax.plot(x, y, linewidth=2)
            ax.set_title(f'Style: {style}')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')

    plt.suptitle('Different Plot Styles', fontsize=14, fontweight='bold')
    plt.tight_layout()

    print(f"Demonstrated {len(available_styles)} plot styles")
    print(f"Available styles on this system: {len(plt.style.available)}")

    plt.savefig('/tmp/custom_06_styles.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/custom_06_styles.png")

    # 7. Axis customization
    print("\n7. Axis Customization")
    print("-" * 40)

    x = np.linspace(0, 10, 100)
    y = np.exp(x/5)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'b-', linewidth=2)

    # Axis limits
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    # Axis labels with formatting
    plt.xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    plt.ylabel('Value', fontsize=14, fontweight='bold')
    plt.title('Exponential Growth', fontsize=16, fontweight='bold', pad=20)

    # Tick customization
    plt.xticks(np.arange(0, 11, 2), fontsize=12)
    plt.yticks(np.arange(0, 11, 2), fontsize=12)

    # Grid customization
    plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.3)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
    plt.minorticks_on()

    print("Axis customizations applied:")
    print("  - Custom axis limits")
    print("  - Formatted labels and title")
    print("  - Custom tick locations and sizes")
    print("  - Major and minor grids")

    plt.savefig('/tmp/custom_07_axes.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved to: /tmp/custom_07_axes.png")

    # 8. Figure size and DPI
    print("\n8. Figure Size and Resolution")
    print("-" * 40)

    x = np.linspace(0, 10, 100)

    # Small figure
    plt.figure(figsize=(6, 4), dpi=80)
    plt.plot(x, np.sin(x))
    plt.title('Small Figure (6x4, 80 DPI)')
    plt.savefig('/tmp/custom_08_small.png', dpi=80, bbox_inches='tight')
    plt.close()
    print("Created: Small figure (6x4 inches, 80 DPI)")

    # Large figure
    plt.figure(figsize=(12, 8), dpi=150)
    plt.plot(x, np.sin(x))
    plt.title('Large Figure (12x8, 150 DPI)')
    plt.savefig('/tmp/custom_08_large.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: Large figure (12x8 inches, 150 DPI)")

    print("\nFigure size affects:")
    print("  - Display dimensions")
    print("  - Text and element sizes")
    print("  - Saved file resolution")

    print("\n" + "=" * 60)
    print("Summary:")
    print("Demonstrated comprehensive plot customization options")
    print("All plots saved to /tmp/ directory")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create plot with custom colors and line styles")
    print("2. Add annotations to mark important points on a curve")
    print("3. Make plot with custom legend outside the plot area")
    print("4. Create heatmap with different colormaps and compare")
    print("=" * 60)

if __name__ == "__main__":
    main()
