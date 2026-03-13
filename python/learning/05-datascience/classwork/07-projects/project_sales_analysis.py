"""
Project: Sales Data Analysis Dashboard
======================================
Complete time series sales analysis with trends, seasonality, and forecasting.

Dataset: Sales data with date, product, quantity, price, region
Goals:
- Load and clean sales data
- Calculate key metrics (total sales, average order value)
- Analyze trends over time
- Detect seasonality patterns
- Compare performance across regions/products
- Forecast future sales
- Create dashboard visualizations

Skills: Pandas, Matplotlib, Seaborn, Time Series Analysis
Run: python project_sales_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def generate_sales_data():
    """Generate realistic sample sales data with seasonality and trends."""
    print("Generating synthetic sales data...")

    np.random.seed(42)

    # Generate date range for 2 years
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    n_days = len(dates)

    # Products and regions
    products = ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard']
    regions = ['North', 'South', 'East', 'West']

    # Generate multiple transactions per day
    records = []

    for day_idx, date in enumerate(dates):
        # Base transactions per day with weekly seasonality
        day_of_week = date.dayofweek
        weekend_factor = 0.7 if day_of_week >= 5 else 1.0

        # Seasonal factor (higher in Q4 for holidays)
        month = date.month
        if month in [11, 12]:
            seasonal_factor = 1.4
        elif month in [1, 2]:
            seasonal_factor = 0.8
        else:
            seasonal_factor = 1.0

        # Trend (growing sales over time)
        trend_factor = 1.0 + (day_idx / n_days) * 0.3

        # Number of transactions
        base_transactions = 15
        n_transactions = int(base_transactions * weekend_factor * seasonal_factor *
                           trend_factor * np.random.uniform(0.8, 1.2))

        for _ in range(n_transactions):
            product = np.random.choice(products)
            region = np.random.choice(regions)

            # Product-specific pricing
            base_prices = {
                'Laptop': 1200,
                'Phone': 800,
                'Tablet': 500,
                'Monitor': 400,
                'Keyboard': 80
            }

            base_price = base_prices[product]
            price = base_price * np.random.uniform(0.9, 1.1)
            quantity = np.random.randint(1, 4) if product != 'Keyboard' else np.random.randint(1, 6)

            # Add some missing values and outliers
            if np.random.random() < 0.02:
                price = np.nan
            if np.random.random() < 0.01:
                quantity = np.nan

            records.append({
                'date': date,
                'product': product,
                'region': region,
                'quantity': quantity,
                'price': price,
                'customer_id': f'C{np.random.randint(1000, 9999)}'
            })

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} sales records")
    return df


def clean_data(df):
    """Clean and preprocess sales data."""
    print("\n" + "=" * 60)
    print("Data Cleaning")
    print("=" * 60)

    # Make a copy
    df = df.copy()

    print(f"Initial records: {len(df)}")
    print(f"Initial missing values:\n{df.isnull().sum()}")

    # Handle missing values
    # Fill missing prices with product median
    for product in df['product'].unique():
        median_price = df[df['product'] == product]['price'].median()
        df.loc[df['product'] == product, 'price'] = df.loc[
            df['product'] == product, 'price'
        ].fillna(median_price)

    # Fill missing quantities with 1
    df['quantity'].fillna(1, inplace=True)

    # Calculate revenue
    df['revenue'] = df['quantity'] * df['price']

    # Remove any outliers (revenue > 99th percentile)
    revenue_99 = df['revenue'].quantile(0.99)
    outliers = df[df['revenue'] > revenue_99]
    print(f"\nRemoved {len(outliers)} outliers (revenue > ${revenue_99:.2f})")
    df = df[df['revenue'] <= revenue_99]

    # Ensure data types
    df['date'] = pd.to_datetime(df['date'])
    df['quantity'] = df['quantity'].astype(int)

    print(f"Final records: {len(df)}")
    print(f"Final missing values: {df.isnull().sum().sum()}")

    return df


def calculate_key_metrics(df):
    """Calculate and display key business metrics."""
    print("\n" + "=" * 60)
    print("Key Business Metrics")
    print("=" * 60)

    total_revenue = df['revenue'].sum()
    total_transactions = len(df)
    total_items = df['quantity'].sum()
    avg_order_value = df['revenue'].mean()
    avg_items_per_order = df['quantity'].mean()
    unique_customers = df['customer_id'].nunique()

    print(f"Total Revenue:           ${total_revenue:,.2f}")
    print(f"Total Transactions:      {total_transactions:,}")
    print(f"Total Items Sold:        {total_items:,}")
    print(f"Average Order Value:     ${avg_order_value:.2f}")
    print(f"Average Items/Order:     {avg_items_per_order:.2f}")
    print(f"Unique Customers:        {unique_customers:,}")

    # Revenue per product
    print("\nRevenue by Product:")
    product_revenue = df.groupby('product')['revenue'].sum().sort_values(ascending=False)
    for product, revenue in product_revenue.items():
        pct = (revenue / total_revenue) * 100
        print(f"  {product:15} ${revenue:12,.2f}  ({pct:5.1f}%)")

    # Revenue per region
    print("\nRevenue by Region:")
    region_revenue = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
    for region, revenue in region_revenue.items():
        pct = (revenue / total_revenue) * 100
        print(f"  {region:15} ${revenue:12,.2f}  ({pct:5.1f}%)")

    return {
        'total_revenue': total_revenue,
        'total_transactions': total_transactions,
        'avg_order_value': avg_order_value
    }


def analyze_time_trends(df):
    """Analyze sales trends over time."""
    print("\n" + "=" * 60)
    print("Time Trends Analysis")
    print("=" * 60)

    # Daily aggregation
    daily = df.groupby('date').agg({
        'revenue': 'sum',
        'quantity': 'sum',
        'customer_id': 'count'
    }).rename(columns={'customer_id': 'transactions'})

    # Calculate moving averages
    daily['revenue_ma7'] = daily['revenue'].rolling(7).mean()
    daily['revenue_ma30'] = daily['revenue'].rolling(30).mean()

    # Monthly aggregation
    monthly = df.set_index('date').resample('ME').agg({
        'revenue': 'sum',
        'quantity': 'sum'
    })

    # Calculate month-over-month growth
    monthly['revenue_growth'] = monthly['revenue'].pct_change() * 100

    print("Monthly Revenue Summary:")
    print(monthly[['revenue', 'revenue_growth']].round(2))

    # Identify best and worst months
    best_month = monthly['revenue'].idxmax()
    worst_month = monthly['revenue'].idxmin()

    print(f"\nBest month:  {best_month.strftime('%B %Y')} (${monthly['revenue'].max():,.2f})")
    print(f"Worst month: {worst_month.strftime('%B %Y')} (${monthly['revenue'].min():,.2f})")

    return daily, monthly


def analyze_seasonality(df):
    """Analyze seasonal patterns."""
    print("\n" + "=" * 60)
    print("Seasonality Analysis")
    print("=" * 60)

    # Add time features
    df_time = df.copy()
    df_time['day_of_week'] = df_time['date'].dt.day_name()
    df_time['month'] = df_time['date'].dt.month
    df_time['month_name'] = df_time['date'].dt.month_name()
    df_time['quarter'] = df_time['date'].dt.quarter

    # Day of week patterns
    dow_sales = df_time.groupby('day_of_week')['revenue'].sum().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])

    print("Average Revenue by Day of Week:")
    for day, revenue in dow_sales.items():
        print(f"  {day:10} ${revenue:10,.2f}")

    # Monthly patterns
    print("\nAverage Revenue by Month:")
    monthly_avg = df_time.groupby('month_name')['revenue'].mean().reindex([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])

    for month, revenue in monthly_avg.items():
        print(f"  {month:10} ${revenue:10,.2f}")

    # Quarterly patterns
    print("\nRevenue by Quarter:")
    quarterly = df_time.groupby('quarter')['revenue'].sum()
    for q, revenue in quarterly.items():
        print(f"  Q{q}        ${revenue:10,.2f}")

    return df_time


def forecast_sales(daily_data, periods=30):
    """Simple sales forecasting using moving average."""
    print("\n" + "=" * 60)
    print(f"Sales Forecast (Next {periods} Days)")
    print("=" * 60)

    # Use last 30 days average as forecast
    last_30_avg = daily_data['revenue'].tail(30).mean()

    # Create forecast dates
    last_date = daily_data.index[-1]
    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=periods)

    # Simple forecast: last 30 days average with slight trend
    trend = daily_data['revenue'].tail(60).mean() - daily_data['revenue'].head(60).mean()
    daily_trend = trend / len(daily_data)

    forecast = []
    for i in range(periods):
        forecast_value = last_30_avg + (daily_trend * i) * np.random.uniform(0.8, 1.2)
        forecast.append(forecast_value)

    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast_revenue': forecast
    })

    total_forecast = sum(forecast)
    print(f"Forecasted total revenue for next {periods} days: ${total_forecast:,.2f}")
    print(f"Average daily revenue forecast: ${last_30_avg:,.2f}")

    return forecast_df


def create_visualizations(df, daily_data, monthly_data, df_time):
    """Create comprehensive dashboard visualizations."""
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 12))

    # 1. Daily revenue with moving averages
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(daily_data.index, daily_data['revenue'], alpha=0.3, label='Daily')
    ax1.plot(daily_data.index, daily_data['revenue_ma7'], label='7-day MA')
    ax1.plot(daily_data.index, daily_data['revenue_ma30'], label='30-day MA', linewidth=2)
    ax1.set_title('Daily Revenue Trends', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Revenue ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Monthly revenue
    ax2 = plt.subplot(3, 3, 2)
    monthly_data['revenue'].plot(kind='bar', ax=ax2, color='steelblue')
    ax2.set_title('Monthly Revenue', fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Revenue ($)')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Revenue by product
    ax3 = plt.subplot(3, 3, 3)
    product_revenue = df.groupby('product')['revenue'].sum().sort_values()
    product_revenue.plot(kind='barh', ax=ax3, color='coral')
    ax3.set_title('Revenue by Product', fontweight='bold')
    ax3.set_xlabel('Revenue ($)')

    # 4. Revenue by region
    ax4 = plt.subplot(3, 3, 4)
    region_revenue = df.groupby('region')['revenue'].sum()
    ax4.pie(region_revenue, labels=region_revenue.index, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Revenue Distribution by Region', fontweight='bold')

    # 5. Day of week patterns
    ax5 = plt.subplot(3, 3, 5)
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_revenue = df_time.groupby('day_of_week')['revenue'].mean().reindex(dow_order)
    dow_revenue.plot(kind='bar', ax=ax5, color='lightgreen')
    ax5.set_title('Average Revenue by Day of Week', fontweight='bold')
    ax5.set_xlabel('Day')
    ax5.set_ylabel('Avg Revenue ($)')
    ax5.tick_params(axis='x', rotation=45)

    # 6. Product quantity distribution
    ax6 = plt.subplot(3, 3, 6)
    product_qty = df.groupby('product')['quantity'].sum().sort_values(ascending=False)
    product_qty.plot(kind='bar', ax=ax6, color='mediumpurple')
    ax6.set_title('Total Quantity Sold by Product', fontweight='bold')
    ax6.set_xlabel('Product')
    ax6.set_ylabel('Quantity')
    ax6.tick_params(axis='x', rotation=45)

    # 7. Revenue distribution
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(df['revenue'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax7.set_title('Revenue Distribution', fontweight='bold')
    ax7.set_xlabel('Revenue ($)')
    ax7.set_ylabel('Frequency')
    ax7.axvline(df['revenue'].mean(), color='red', linestyle='--', label=f"Mean: ${df['revenue'].mean():.2f}")
    ax7.legend()

    # 8. Quarterly comparison
    ax8 = plt.subplot(3, 3, 8)
    quarterly = df_time.groupby('quarter')['revenue'].sum()
    ax8.bar(quarterly.index, quarterly.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    ax8.set_title('Revenue by Quarter', fontweight='bold')
    ax8.set_xlabel('Quarter')
    ax8.set_ylabel('Revenue ($)')
    ax8.set_xticks(quarterly.index)
    ax8.set_xticklabels([f'Q{i}' for i in quarterly.index])

    # 9. Top products by region
    ax9 = plt.subplot(3, 3, 9)
    pivot = df.pivot_table(values='revenue', index='product', columns='region', aggfunc='sum')
    pivot.plot(kind='bar', stacked=True, ax=ax9)
    ax9.set_title('Revenue: Products by Region', fontweight='bold')
    ax9.set_xlabel('Product')
    ax9.set_ylabel('Revenue ($)')
    ax9.tick_params(axis='x', rotation=45)
    ax9.legend(title='Region')

    plt.tight_layout()
    print("Dashboard created successfully!")
    print("Close the plot window to continue...")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("SALES DATA ANALYSIS DASHBOARD")
    print("=" * 60)

    # 1. Generate and load data
    df = generate_sales_data()

    print(f"\nDataset overview:")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total records: {len(df):,}")
    print(f"  Products: {df['product'].nunique()}")
    print(f"  Regions: {df['region'].nunique()}")

    # 2. Clean data
    df = clean_data(df)

    # 3. Calculate key metrics
    metrics = calculate_key_metrics(df)

    # 4. Time trends analysis
    daily_data, monthly_data = analyze_time_trends(df)

    # 5. Seasonality analysis
    df_time = analyze_seasonality(df)

    # 6. Forecast
    forecast = forecast_sales(daily_data, periods=30)

    # 7. Create visualizations
    create_visualizations(df, daily_data, monthly_data, df_time)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("1. Sales show clear seasonal patterns with Q4 spike")
    print("2. Weekday sales outperform weekend sales")
    print("3. Laptops and Phones are top revenue generators")
    print("4. Consistent growth trend throughout the period")
    print("5. Regional performance is relatively balanced")
    print("=" * 60)


if __name__ == "__main__":
    main()
