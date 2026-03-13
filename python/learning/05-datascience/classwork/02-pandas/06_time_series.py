"""
Time Series Analysis with Pandas
=================================
Working with dates, times, and time-indexed data.

Topics:
- Creating datetime objects
- DatetimeIndex and time-based indexing
- Resampling and frequency conversion
- Rolling windows
- Time-based operations

Run: python 06_time_series.py
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("Time Series Analysis with Pandas")
    print("=" * 60)

    # 1. Creating datetime objects
    print("\n1. Creating Datetime Objects")
    print("-" * 40)

    # String to datetime
    date_str = '2024-01-15'
    date = pd.to_datetime(date_str)
    print(f"String '{date_str}' to datetime: {date}")
    print(f"Type: {type(date)}")

    # List of dates
    dates = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
    print(f"\nList of dates:\n{dates}")

    # Different date formats
    dates_fmt = pd.to_datetime(['01/15/2024', '02/20/2024'], format='%m/%d/%Y')
    print(f"\nCustom format dates:\n{dates_fmt}")

    # 2. Date ranges
    print("\n2. Creating Date Ranges")
    print("-" * 40)

    # Daily dates
    daily = pd.date_range(start='2024-01-01', end='2024-01-10')
    print("Daily dates:")
    print(daily)

    # With periods
    weekly = pd.date_range(start='2024-01-01', periods=8, freq='W')
    print("\nWeekly dates (8 weeks):")
    print(weekly)

    # Monthly dates
    monthly = pd.date_range(start='2024-01-01', periods=12, freq='ME')
    print("\nMonthly dates (12 months):")
    print(monthly)

    # Hourly dates
    hourly = pd.date_range(start='2024-01-01', periods=24, freq='h')
    print("\nHourly dates (24 hours):")
    print(hourly[:5], "...")

    # 3. DatetimeIndex
    print("\n3. Time-Indexed DataFrames")
    print("-" * 40)

    # Create time series data
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    values = np.random.randint(100, 200, size=10)

    ts = pd.Series(values, index=dates)
    print("Time Series:")
    print(ts)

    # DataFrame with DatetimeIndex
    df = pd.DataFrame({
        'Temperature': np.random.uniform(20, 30, 10),
        'Humidity': np.random.uniform(40, 70, 10)
    }, index=dates)

    print("\nTime-indexed DataFrame:")
    print(df)

    # 4. Accessing time series data
    print("\n4. Accessing Time Series Data")
    print("-" * 40)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'Sales': np.random.randint(100, 500, 30)
    }, index=dates)

    print("January 2024 sales (first 10 days):")
    print(df.head(10))

    # Access by date string
    print(f"\nSales on 2024-01-05: {df.loc['2024-01-05', 'Sales']}")

    # Slice by date range
    print("\nSales from Jan 10-15:")
    print(df.loc['2024-01-10':'2024-01-15'])

    # Access by year-month
    print("\nAll January 2024 sales:")
    print(df.loc['2024-01'])

    # 5. Time-based operations
    print("\n5. Time-Based Operations")
    print("-" * 40)

    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Value': np.random.randint(10, 100, 10)
    })

    # Extract components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.day_name()

    print("DataFrame with date components:")
    print(df[['Date', 'Year', 'Month', 'Day', 'DayOfWeek', 'Value']])

    # Day of week, quarter, etc.
    df['Quarter'] = df['Date'].dt.quarter
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    print("\nAdditional time features:")
    print(df[['Date', 'Quarter', 'WeekOfYear']].head())

    # 6. Resampling
    print("\n6. Resampling (Changing Frequency)")
    print("-" * 40)

    # Daily data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'Sales': np.random.randint(100, 500, 30)
    }, index=dates)

    print("Daily sales (first 10 days):")
    print(df.head(10))

    # Resample to weekly (sum)
    weekly = df.resample('W').sum()
    print("\nWeekly sales (sum):")
    print(weekly)

    # Resample to weekly (mean)
    weekly_avg = df.resample('W').mean()
    print("\nWeekly average sales:")
    print(weekly_avg)

    # 7. Rolling windows
    print("\n7. Rolling Windows (Moving Averages)")
    print("-" * 40)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    df = pd.DataFrame({
        'Price': np.random.uniform(90, 110, 20)
    }, index=dates)

    # 3-day moving average
    df['MA_3'] = df['Price'].rolling(window=3).mean()

    # 7-day moving average
    df['MA_7'] = df['Price'].rolling(window=7).mean()

    print("Price with moving averages:")
    print(df[['Price', 'MA_3', 'MA_7']].round(2))

    # Rolling sum, min, max
    df['Roll_Sum_5'] = df['Price'].rolling(window=5).sum()
    df['Roll_Min_5'] = df['Price'].rolling(window=5).min()
    df['Roll_Max_5'] = df['Price'].rolling(window=5).max()

    print("\nRolling statistics (window=5):")
    print(df[['Price', 'Roll_Sum_5', 'Roll_Min_5', 'Roll_Max_5']].tail(10).round(2))

    # 8. Shifting and lagging
    print("\n8. Shifting and Lagging")
    print("-" * 40)

    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'Price': [100, 102, 105, 103, 107, 110, 108, 112, 115, 113]
    }, index=dates)

    # Shift forward (lag)
    df['Price_Lag1'] = df['Price'].shift(1)
    df['Price_Lag2'] = df['Price'].shift(2)

    # Shift backward (lead)
    df['Price_Lead1'] = df['Price'].shift(-1)

    print("Price with shifts:")
    print(df)

    # Calculate daily change
    df['Daily_Change'] = df['Price'] - df['Price_Lag1']
    df['Percent_Change'] = ((df['Price'] - df['Price_Lag1']) / df['Price_Lag1'] * 100).round(2)

    print("\nWith calculated changes:")
    print(df[['Price', 'Daily_Change', 'Percent_Change']])

    # 9. Time deltas
    print("\n9. Time Deltas (Time Differences)")
    print("-" * 40)

    # Create events with timestamps
    events = pd.DataFrame({
        'Event': ['Start', 'Checkpoint1', 'Checkpoint2', 'End'],
        'Time': pd.to_datetime([
            '2024-01-01 10:00:00',
            '2024-01-01 10:15:30',
            '2024-01-01 10:35:45',
            '2024-01-01 11:00:00'
        ])
    })

    print("Events:")
    print(events)

    # Calculate time differences
    events['Duration'] = events['Time'].diff()
    print("\nWith duration:")
    print(events)

    # Total duration
    total_duration = events['Time'].iloc[-1] - events['Time'].iloc[0]
    print(f"\nTotal duration: {total_duration}")

    # 10. Practical example: Stock analysis
    print("\n10. Practical Example: Stock Price Analysis")
    print("-" * 40)

    # Simulate stock data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=60, freq='D')

    # Random walk for stock price
    returns = np.random.randn(60) * 2
    price = 100 + np.cumsum(returns)

    stock = pd.DataFrame({
        'Price': price,
        'Volume': np.random.randint(1000, 5000, 60)
    }, index=dates)

    # Calculate indicators
    stock['MA_7'] = stock['Price'].rolling(7).mean()
    stock['MA_30'] = stock['Price'].rolling(30).mean()
    stock['Daily_Return'] = stock['Price'].pct_change() * 100

    print("Stock data (last 10 days):")
    print(stock[['Price', 'MA_7', 'MA_30', 'Daily_Return']].tail(10).round(2))

    # Monthly statistics
    monthly_stats = stock.resample('ME').agg({
        'Price': ['first', 'last', 'min', 'max', 'mean'],
        'Volume': 'sum'
    })

    print("\nMonthly statistics:")
    print(monthly_stats.round(2))

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create hourly temperature data for a week, resample to daily")
    print("2. Calculate 5-day and 10-day moving averages for price data")
    print("3. Find the day of week with highest average sales")
    print("4. Calculate month-over-month growth rate for a time series")
    print("=" * 60)

if __name__ == "__main__":
    main()
