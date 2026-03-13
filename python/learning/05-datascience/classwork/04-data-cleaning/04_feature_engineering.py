"""
Feature Engineering
===================
Creating new features from existing data to improve model performance.

Topics:
- Creating derived features
- Binning (discretization)
- Feature interactions
- Polynomial features
- DateTime features
- Feature encoding

Run: python 04_feature_engineering.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def main():
    print("=" * 60)
    print("Feature Engineering")
    print("=" * 60)

    # 1. Derived features
    print("\n1. Creating Derived Features")
    print("-" * 40)

    # Create sample data
    df = pd.DataFrame({
        'Price': [100, 150, 200, 120, 180],
        'Quantity': [2, 3, 1, 4, 2],
        'Discount': [10, 20, 0, 15, 10],
        'Shipping': [5, 10, 8, 5, 10]
    })

    print("Original data:")
    print(df)

    # Calculate total revenue
    df['Revenue'] = df['Price'] * df['Quantity']
    print("\nWith Revenue (Price * Quantity):")
    print(df)

    # Calculate final cost
    df['Final_Price'] = df['Price'] - df['Discount'] + df['Shipping']
    df['Total_Cost'] = df['Final_Price'] * df['Quantity']
    print("\nWith calculated costs:")
    print(df)

    # Calculate discount percentage
    df['Discount_Pct'] = (df['Discount'] / df['Price'] * 100).round(2)
    print("\nWith discount percentage:")
    print(df[['Price', 'Discount', 'Discount_Pct']])

    # 2. Binning continuous variables
    print("\n2. Binning (Discretization)")
    print("-" * 40)

    # Age binning
    ages = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'],
        'Age': [22, 35, 42, 28, 55, 19, 67, 31]
    })

    print("Age data:")
    print(ages)

    # Equal-width bins
    ages['Age_Group'] = pd.cut(ages['Age'],
                                bins=3,
                                labels=['Young', 'Middle', 'Senior'])
    print("\nWith age groups (equal-width bins):")
    print(ages)

    # Custom bins
    bins = [0, 25, 40, 60, 100]
    labels = ['Youth', 'Adult', 'Middle-aged', 'Senior']
    ages['Age_Category'] = pd.cut(ages['Age'], bins=bins, labels=labels)
    print("\nWith custom age categories:")
    print(ages)

    # Quantile-based bins
    ages['Age_Quartile'] = pd.qcut(ages['Age'],
                                    q=4,
                                    labels=['Q1', 'Q2', 'Q3', 'Q4'])
    print("\nWith age quartiles (equal frequency):")
    print(ages)

    # 3. Feature interactions
    print("\n3. Feature Interactions")
    print("-" * 40)

    # Create dataset
    df_interact = pd.DataFrame({
        'Feature_A': [1, 2, 3, 4, 5],
        'Feature_B': [10, 20, 30, 40, 50],
        'Feature_C': [100, 200, 300, 400, 500]
    })

    print("Original features:")
    print(df_interact)

    # Multiply features
    df_interact['A_times_B'] = df_interact['Feature_A'] * df_interact['Feature_B']
    print("\nWith multiplication interaction:")
    print(df_interact)

    # Ratio features
    df_interact['A_over_B'] = df_interact['Feature_A'] / df_interact['Feature_B']
    df_interact['B_over_C'] = df_interact['Feature_B'] / df_interact['Feature_C']
    print("\nWith ratio features:")
    print(df_interact[['Feature_A', 'Feature_B', 'Feature_C', 'A_over_B', 'B_over_C']])

    # Sum and difference
    df_interact['A_plus_B'] = df_interact['Feature_A'] + df_interact['Feature_B']
    df_interact['B_minus_A'] = df_interact['Feature_B'] - df_interact['Feature_A']
    print("\nWith sum and difference:")
    print(df_interact[['Feature_A', 'Feature_B', 'A_plus_B', 'B_minus_A']])

    # 4. Polynomial features
    print("\n4. Polynomial Features")
    print("-" * 40)

    df_poly = pd.DataFrame({
        'X': [1, 2, 3, 4, 5]
    })

    print("Original feature:")
    print(df_poly)

    # Create polynomial features
    df_poly['X_squared'] = df_poly['X'] ** 2
    df_poly['X_cubed'] = df_poly['X'] ** 3
    df_poly['X_sqrt'] = np.sqrt(df_poly['X'])
    df_poly['X_log'] = np.log(df_poly['X'])

    print("\nWith polynomial features:")
    print(df_poly.round(3))

    # Multiple features - cross terms
    df_cross = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [2, 4, 6, 8]
    })

    print("\nOriginal features A and B:")
    print(df_cross)

    # Polynomial degree 2
    df_cross['A_squared'] = df_cross['A'] ** 2
    df_cross['B_squared'] = df_cross['B'] ** 2
    df_cross['A_times_B'] = df_cross['A'] * df_cross['B']

    print("\nWith degree-2 polynomial features:")
    print(df_cross)

    # 5. DateTime features
    print("\n5. DateTime Feature Extraction")
    print("-" * 40)

    # Create datetime data
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    df_date = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.randint(100, 500, 10)
    })

    print("Original datetime data:")
    print(df_date)

    # Extract datetime components
    df_date['Year'] = df_date['Date'].dt.year
    df_date['Month'] = df_date['Date'].dt.month
    df_date['Day'] = df_date['Date'].dt.day
    df_date['DayOfWeek'] = df_date['Date'].dt.dayofweek
    df_date['DayName'] = df_date['Date'].dt.day_name()
    df_date['Quarter'] = df_date['Date'].dt.quarter
    df_date['WeekOfYear'] = df_date['Date'].dt.isocalendar().week

    print("\nWith extracted datetime features:")
    print(df_date)

    # Is weekend feature
    df_date['Is_Weekend'] = df_date['DayOfWeek'].isin([5, 6]).astype(int)
    print("\nWith weekend indicator:")
    print(df_date[['Date', 'DayName', 'Is_Weekend']])

    # Time since reference
    reference_date = pd.Timestamp('2024-01-01')
    df_date['Days_Since_Start'] = (df_date['Date'] - reference_date).dt.days
    print("\nWith days since start:")
    print(df_date[['Date', 'Days_Since_Start']])

    # 6. Aggregated features
    print("\n6. Aggregated Features")
    print("-" * 40)

    # Create customer transaction data
    transactions = pd.DataFrame({
        'CustomerID': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'Amount': [100, 150, 200, 300, 250, 50, 75, 100, 125],
        'Date': pd.date_range('2024-01-01', periods=9, freq='D')
    })

    print("Transaction data:")
    print(transactions)

    # Aggregate by customer
    customer_features = transactions.groupby('CustomerID').agg({
        'Amount': ['count', 'sum', 'mean', 'min', 'max', 'std']
    }).round(2)

    customer_features.columns = ['Transaction_Count', 'Total_Spent',
                                  'Avg_Transaction', 'Min_Transaction',
                                  'Max_Transaction', 'Std_Transaction']

    print("\nAggregated customer features:")
    print(customer_features)

    # Add range feature
    customer_features['Amount_Range'] = (customer_features['Max_Transaction'] -
                                          customer_features['Min_Transaction'])

    print("\nWith additional derived features:")
    print(customer_features)

    # 7. Text-based features
    print("\n7. Text-Based Features")
    print("-" * 40)

    df_text = pd.DataFrame({
        'Product': ['Laptop Pro 15', 'Phone X', 'Tablet Mini',
                    'Laptop Air 13', 'Phone Pro Max'],
        'Description': ['High-end laptop', 'Smartphone', 'Small tablet',
                       'Lightweight laptop', 'Premium phone']
    })

    print("Product data:")
    print(df_text)

    # Extract from product name
    df_text['Product_Length'] = df_text['Product'].str.len()
    df_text['Word_Count'] = df_text['Product'].str.split().str.len()
    df_text['Has_Pro'] = df_text['Product'].str.contains('Pro').astype(int)
    df_text['Has_Number'] = df_text['Product'].str.contains(r'\d').astype(int)

    print("\nWith text features:")
    print(df_text)

    # Extract numbers from text
    df_text['Screen_Size'] = df_text['Product'].str.extract(r'(\d+)').astype(float)
    print("\nWith extracted numbers:")
    print(df_text[['Product', 'Screen_Size']])

    # 8. Flag/Indicator features
    print("\n8. Flag and Indicator Features")
    print("-" * 40)

    df_flags = pd.DataFrame({
        'Price': [100, 50, 200, 30, 150, 80],
        'Stock': [10, 0, 5, 0, 20, 3],
        'Rating': [4.5, 3.2, 4.8, 2.9, 4.0, 3.8]
    })

    print("Product data:")
    print(df_flags)

    # Create indicator features
    df_flags['Is_Premium'] = (df_flags['Price'] > 100).astype(int)
    df_flags['Out_of_Stock'] = (df_flags['Stock'] == 0).astype(int)
    df_flags['Low_Stock'] = (df_flags['Stock'] < 5).astype(int)
    df_flags['High_Rating'] = (df_flags['Rating'] >= 4.0).astype(int)
    df_flags['Excellent_Rating'] = (df_flags['Rating'] >= 4.5).astype(int)

    print("\nWith indicator features:")
    print(df_flags)

    # Combined conditions
    df_flags['Premium_High_Rating'] = ((df_flags['Price'] > 100) &
                                        (df_flags['Rating'] >= 4.0)).astype(int)
    print("\nWith combined indicators:")
    print(df_flags[['Price', 'Rating', 'Premium_High_Rating']])

    print("\n" + "=" * 60)
    print("Summary - Feature Engineering Techniques:")
    print("  1. Derived features: Mathematical combinations of features")
    print("  2. Binning: Convert continuous to categorical")
    print("  3. Interactions: Multiply, divide, add features")
    print("  4. Polynomial: Square, cube, sqrt, log transformations")
    print("  5. DateTime: Extract year, month, day, day of week, etc.")
    print("  6. Aggregations: Group-based statistics")
    print("  7. Text features: Length, word count, pattern matching")
    print("  8. Indicators: Binary flags for conditions")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create BMI feature from height and weight")
    print("2. Engineer time-based features (hour of day, part of day)")
    print("3. Create ratio features for financial data (P/E ratio)")
    print("4. Generate lag features for time series data")
    print("5. Create one-hot encoded interaction features")
    print("=" * 60)

if __name__ == "__main__":
    main()
