"""
GroupBy and Aggregation
=======================
Group data and perform aggregate calculations.

Topics:
- GroupBy basics
- Aggregation functions
- Multiple aggregations
- Grouping by multiple columns
- Transform and apply

Run: python 04_groupby_aggregation.py
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("GroupBy and Aggregation")
    print("=" * 60)

    # Create sample sales data
    np.random.seed(42)
    df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=20),
        'Product': np.random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor'], 20),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 20),
        'Sales': np.random.randint(100, 1000, 20),
        'Quantity': np.random.randint(1, 10, 20)
    })

    print("\nSales Data:")
    print(df.head(10))

    # 1. Basic GroupBy
    print("\n1. Basic GroupBy Operations")
    print("-" * 40)

    # Group by single column
    grouped = df.groupby('Product')
    print("GroupBy object:")
    print(grouped)

    # Calculate mean per group
    mean_sales = df.groupby('Product')['Sales'].mean()
    print("\nAverage sales by product:")
    print(mean_sales)

    # Count per group
    count = df.groupby('Product').size()
    print("\nCount by product:")
    print(count)

    # 2. Common aggregation functions
    print("\n2. Common Aggregation Functions")
    print("-" * 40)

    # Sum
    total_sales = df.groupby('Product')['Sales'].sum()
    print("Total sales by product:")
    print(total_sales)

    # Mean
    avg_quantity = df.groupby('Product')['Quantity'].mean()
    print("\nAverage quantity by product:")
    print(avg_quantity)

    # Min and Max
    print("\nMin sales by product:")
    print(df.groupby('Product')['Sales'].min())

    print("\nMax sales by product:")
    print(df.groupby('Product')['Sales'].max())

    # Standard deviation
    print("\nSales std by product:")
    print(df.groupby('Product')['Sales'].std())

    # 3. Multiple aggregations
    print("\n3. Multiple Aggregations")
    print("-" * 40)

    # Aggregate multiple functions
    stats = df.groupby('Product')['Sales'].agg(['count', 'sum', 'mean', 'std', 'min', 'max'])
    print("Multiple statistics by product:")
    print(stats)

    # Different aggregations for different columns
    multi_agg = df.groupby('Product').agg({
        'Sales': ['sum', 'mean'],
        'Quantity': ['sum', 'mean']
    })
    print("\nMultiple columns with different aggregations:")
    print(multi_agg)

    # 4. GroupBy with multiple columns
    print("\n4. GroupBy with Multiple Columns")
    print("-" * 40)

    # Group by Product and Region
    multi_group = df.groupby(['Product', 'Region'])['Sales'].sum()
    print("Total sales by Product and Region:")
    print(multi_group)

    # Unstack for better visualization
    print("\nUnstacked (Products as rows, Regions as columns):")
    print(multi_group.unstack(fill_value=0))

    # Statistics by multiple groups
    multi_stats = df.groupby(['Product', 'Region']).agg({
        'Sales': ['count', 'sum', 'mean'],
        'Quantity': 'sum'
    })
    print("\nDetailed stats by Product and Region:")
    print(multi_stats.head(10))

    # 5. Custom aggregation functions
    print("\n5. Custom Aggregation Functions")
    print("-" * 40)

    # Define custom function
    def range_func(x):
        return x.max() - x.min()

    custom_agg = df.groupby('Product')['Sales'].agg([
        ('Total', 'sum'),
        ('Average', 'mean'),
        ('Range', range_func),
        ('Count', 'count')
    ])
    print("Custom aggregations:")
    print(custom_agg)

    # Lambda functions
    lambda_agg = df.groupby('Product')['Sales'].agg([
        ('Total', 'sum'),
        ('Top10%', lambda x: x.quantile(0.9)),
        ('Bottom10%', lambda x: x.quantile(0.1))
    ])
    print("\nUsing lambda functions:")
    print(lambda_agg)

    # 6. Transform method
    print("\n6. Transform Method")
    print("-" * 40)

    # Add group mean to original DataFrame
    df['Product_Avg_Sales'] = df.groupby('Product')['Sales'].transform('mean')
    print("DataFrame with product average:")
    print(df[['Product', 'Sales', 'Product_Avg_Sales']].head(10))

    # Normalize by group
    df['Sales_Normalized'] = df.groupby('Product')['Sales'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    print("\nNormalized sales by product:")
    print(df[['Product', 'Sales', 'Sales_Normalized']].head(10))

    # 7. Filter groups
    print("\n7. Filter Groups")
    print("-" * 40)

    # Keep groups with more than 4 records
    filtered = df.groupby('Product').filter(lambda x: len(x) > 4)
    print("Products with more than 4 sales:")
    print(filtered.groupby('Product').size())

    # Filter based on aggregate condition
    high_sales = df.groupby('Product').filter(lambda x: x['Sales'].sum() > 2000)
    print("\nProducts with total sales > $2000:")
    print(high_sales.groupby('Product')['Sales'].sum())

    # 8. Apply method
    print("\n8. Apply Method (More Flexible)")
    print("-" * 40)

    # Apply custom function to each group
    def top_n_sales(group, n=3):
        return group.nlargest(n, 'Sales')

    top_sales = df.groupby('Product').apply(top_n_sales, n=2)
    print("Top 2 sales for each product:")
    print(top_sales[['Product', 'Sales', 'Quantity']])

    # 9. Grouping by time periods
    print("\n9. Grouping by Time Periods")
    print("-" * 40)

    # Set Date as index
    df_time = df.set_index('Date')

    # Group by week
    weekly = df_time.groupby(pd.Grouper(freq='W'))['Sales'].sum()
    print("Weekly total sales:")
    print(weekly)

    # Group by product and week
    product_weekly = df_time.groupby(['Product', pd.Grouper(freq='W')])['Sales'].sum()
    print("\nWeekly sales by product (first 10):")
    print(product_weekly.head(10))

    # 10. Practical example: Summary statistics
    print("\n10. Practical Example: Sales Summary")
    print("-" * 40)

    # Comprehensive summary
    summary = df.groupby('Product').agg({
        'Sales': ['count', 'sum', 'mean', 'min', 'max'],
        'Quantity': ['sum', 'mean']
    }).round(2)

    # Rename columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    print("Comprehensive sales summary:")
    print(summary)

    # Add total row
    print("\nProduct performance ranking:")
    ranking = df.groupby('Product').agg({
        'Sales': 'sum',
        'Quantity': 'sum'
    }).sort_values('Sales', ascending=False)
    print(ranking)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Find the region with highest total sales")
    print("2. Calculate average quantity per product per region")
    print("3. Find products where max sale > 3x mean sale")
    print("4. Create new column with percentage of product's total sales")
    print("=" * 60)

if __name__ == "__main__":
    main()
