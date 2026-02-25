"""
Interactive Plots with Plotly
==============================
Creating interactive visualizations that can be explored in a browser.

Topics:
- Interactive line and scatter plots
- Interactive bar and histogram charts
- 3D plots
- Hover information
- Saving to HTML

Run: python 05_plotly_interactive.py
Note: Creates HTML files that can be opened in a browser
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 60)
    print("Interactive Plots with Plotly")
    print("=" * 60)

    # Try to import plotly
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        plotly_available = True
        print("Plotly is available")
    except ImportError:
        print("\nWARNING: Plotly not installed")
        print("Install with: pip install plotly")
        print("\nContinuing with matplotlib fallbacks...")
        plotly_available = False
        import matplotlib.pyplot as plt

    # 1. Interactive line plot
    print("\n1. Interactive Line Plot")
    print("-" * 40)

    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    if plotly_available:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='sin(x)'))
        fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='cos(x)'))
        fig.update_layout(
            title='Interactive Line Plot',
            xaxis_title='X axis',
            yaxis_title='Y axis',
            hovermode='x unified'
        )
        fig.write_html('/tmp/plotly_01_line.html')
        print("Created: Interactive line plot")
        print("  Features: Hover, zoom, pan, legend toggle")
        print("  Saved to: /tmp/plotly_01_line.html")
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(x, y1, label='sin(x)')
        plt.plot(x, y2, label='cos(x)')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Line Plot (static - plotly not available)')
        plt.legend()
        plt.savefig('/tmp/plotly_01_line.png')
        plt.close()
        print("Created: Static line plot (plotly unavailable)")
        print("  Saved to: /tmp/plotly_01_line.png")

    # 2. Interactive scatter plot
    print("\n2. Interactive Scatter Plot")
    print("-" * 40)

    np.random.seed(42)
    n = 100
    scatter_data = pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n),
        'size': np.random.randint(10, 100, n),
        'category': np.random.choice(['A', 'B', 'C'], n)
    })

    if plotly_available:
        fig = px.scatter(scatter_data, x='x', y='y', size='size', color='category',
                        hover_data=['size'],
                        title='Interactive Scatter Plot')
        fig.write_html('/tmp/plotly_02_scatter.html')
        print("Created: Interactive scatter plot")
        print(f"  Points: {n}")
        print("  Features: Hover shows all data, color by category, size variation")
        print("  Saved to: /tmp/plotly_02_scatter.html")
    else:
        plt.figure(figsize=(10, 6))
        for cat in scatter_data['category'].unique():
            mask = scatter_data['category'] == cat
            plt.scatter(scatter_data[mask]['x'], scatter_data[mask]['y'],
                       s=scatter_data[mask]['size'], alpha=0.6, label=cat)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Scatter Plot (static)')
        plt.legend()
        plt.savefig('/tmp/plotly_02_scatter.png')
        plt.close()
        print("Created: Static scatter plot")
        print("  Saved to: /tmp/plotly_02_scatter.png")

    # 3. Interactive bar chart
    print("\n3. Interactive Bar Chart")
    print("-" * 40)

    categories = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    values = [45, 38, 52, 41, 48]

    if plotly_available:
        fig = go.Figure(data=[go.Bar(x=categories, y=values, text=values,
                                     textposition='auto')])
        fig.update_layout(
            title='Interactive Bar Chart',
            xaxis_title='Product',
            yaxis_title='Sales',
            hovermode='x'
        )
        fig.write_html('/tmp/plotly_03_bar.html')
        print("Created: Interactive bar chart")
        print("  Features: Hover shows exact values")
        print("  Saved to: /tmp/plotly_03_bar.html")
    else:
        plt.figure(figsize=(10, 6))
        plt.bar(categories, values)
        plt.xlabel('Product')
        plt.ylabel('Sales')
        plt.title('Bar Chart (static)')
        plt.savefig('/tmp/plotly_03_bar.png')
        plt.close()
        print("Created: Static bar chart")
        print("  Saved to: /tmp/plotly_03_bar.png")

    # 4. Interactive histogram
    print("\n4. Interactive Histogram")
    print("-" * 40)

    np.random.seed(42)
    data = np.random.randn(1000)

    if plotly_available:
        fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30)])
        fig.update_layout(
            title='Interactive Histogram',
            xaxis_title='Value',
            yaxis_title='Frequency'
        )
        fig.write_html('/tmp/plotly_04_hist.html')
        print("Created: Interactive histogram")
        print(f"  Data points: {len(data)}")
        print("  Features: Hover shows bin ranges and counts")
        print("  Saved to: /tmp/plotly_04_hist.html")
    else:
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram (static)')
        plt.savefig('/tmp/plotly_04_hist.png')
        plt.close()
        print("Created: Static histogram")
        print("  Saved to: /tmp/plotly_04_hist.png")

    # 5. 3D scatter plot
    print("\n5. 3D Scatter Plot")
    print("-" * 40)

    np.random.seed(42)
    n = 100
    x3d = np.random.randn(n)
    y3d = np.random.randn(n)
    z3d = np.random.randn(n)

    if plotly_available:
        fig = go.Figure(data=[go.Scatter3d(
            x=x3d, y=y3d, z=z3d,
            mode='markers',
            marker=dict(
                size=8,
                color=z3d,
                colorscale='Viridis',
                showscale=True
            )
        )])
        fig.update_layout(
            title='3D Interactive Scatter Plot',
            scene=dict(
                xaxis_title='X axis',
                yaxis_title='Y axis',
                zaxis_title='Z axis'
            )
        )
        fig.write_html('/tmp/plotly_05_3d.html')
        print("Created: Interactive 3D scatter plot")
        print(f"  Points: {n}")
        print("  Features: Rotate, zoom, pan in 3D space")
        print("  Saved to: /tmp/plotly_05_3d.html")
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x3d, y3d, z3d, c=z3d, cmap='viridis')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Scatter Plot (static)')
        plt.savefig('/tmp/plotly_05_3d.png')
        plt.close()
        print("Created: Static 3D scatter plot")
        print("  Saved to: /tmp/plotly_05_3d.png")

    # 6. Time series with range selector
    print("\n6. Interactive Time Series")
    print("-" * 40)

    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(365) * 2)

    if plotly_available:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='Value'))
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        fig.update_layout(title='Time Series with Range Selector')
        fig.write_html('/tmp/plotly_06_timeseries.html')
        print("Created: Interactive time series")
        print("  Features: Range selector, range slider, zoom to time periods")
        print("  Saved to: /tmp/plotly_06_timeseries.html")
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Time Series (static)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('/tmp/plotly_06_timeseries.png')
        plt.close()
        print("Created: Static time series")
        print("  Saved to: /tmp/plotly_06_timeseries.png")

    # 7. Multiple subplots
    print("\n7. Interactive Subplots")
    print("-" * 40)

    if plotly_available:
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sine', 'Cosine', 'Tangent', 'Bar Chart')
        )

        x = np.linspace(0, 10, 100)

        fig.add_trace(go.Scatter(x=x, y=np.sin(x), name='sin'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=np.cos(x), name='cos'), row=1, col=2)
        fig.add_trace(go.Scatter(x=x, y=np.tan(x), name='tan'), row=2, col=1)
        fig.add_trace(go.Bar(x=['A', 'B', 'C'], y=[3, 7, 5], name='bar'), row=2, col=2)

        fig.update_layout(height=800, title_text="Interactive Subplots")
        fig.write_html('/tmp/plotly_07_subplots.html')
        print("Created: Interactive subplots")
        print("  4 different plots in 2x2 grid")
        print("  Saved to: /tmp/plotly_07_subplots.html")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        x = np.linspace(0, 10, 100)
        axes[0, 0].plot(x, np.sin(x))
        axes[0, 0].set_title('Sine')
        axes[0, 1].plot(x, np.cos(x))
        axes[0, 1].set_title('Cosine')
        axes[1, 0].plot(x, np.tan(x))
        axes[1, 0].set_title('Tangent')
        axes[1, 0].set_ylim(-5, 5)
        axes[1, 1].bar(['A', 'B', 'C'], [3, 7, 5])
        axes[1, 1].set_title('Bar Chart')
        plt.tight_layout()
        plt.savefig('/tmp/plotly_07_subplots.png')
        plt.close()
        print("Created: Static subplots")
        print("  Saved to: /tmp/plotly_07_subplots.png")

    print("\n" + "=" * 60)
    print("Summary:")
    if plotly_available:
        print("Created interactive Plotly visualizations")
        print("Open HTML files in browser to interact with plots")
        print("\nInteractive features:")
        print("  - Hover to see data details")
        print("  - Click legend to toggle series")
        print("  - Zoom by dragging, double-click to reset")
        print("  - Pan by dragging in pan mode")
        print("  - Download plot as PNG")
    else:
        print("Created static matplotlib plots")
        print("Install plotly for interactive features: pip install plotly")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create interactive scatter plot with 3 categories")
    print("2. Make interactive bar chart with grouped bars")
    print("3. Create 3D surface plot (z = sin(x) * cos(y))")
    print("4. Build time series with multiple traces and range selector")
    print("=" * 60)

if __name__ == "__main__":
    main()
