"""
Pandas Exercises
================
Practice problems covering all Pandas concepts from this module.

Topics covered:
- Series and DataFrames
- Reading data
- Selection and filtering
- GroupBy and aggregation
- Merging and joining
- Time series operations

Run: python exercises.py
"""

import pandas as pd
import numpy as np
import io

def exercise_1():
    """DataFrame creation and basic operations"""
    print("\nExercise 1: DataFrame Creation")
    print("-" * 40)

    # Create student DataFrame
    students = pd.DataFrame({
        'StudentID': [1, 2, 3, 4, 5],
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Grade': [85, 92, 78, 88, 95],
        'Subject': ['Math', 'Science', 'Math', 'Science', 'Math']
    })

    print("Student DataFrame:")
    print(students)

    # Calculate average grade
    print(f"\nAverage grade: {students['Grade'].mean():.2f}")

    # Find students with grade > 85
    high_performers = students[students['Grade'] > 85]
    print("\nStudents with grade > 85:")
    print(high_performers)

    # Add Pass/Fail column
    students['Status'] = students['Grade'].apply(lambda x: 'Pass' if x >= 70 else 'Fail')
    print("\nWith Pass/Fail status:")
    print(students)

def exercise_2():
    """Selection and filtering"""
    print("\nExercise 2: Selection and Filtering")
    print("-" * 40)

    # Create employee DataFrame
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
        'Department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'Finance'],
        'Age': [25, 30, 35, 28, 32, 29],
        'Salary': [60000, 75000, 80000, 72000, 65000, 85000]
    })

    print("Employee DataFrame:")
    print(df)

    # Finance department with salary > 80k
    finance_high = df[(df['Department'] == 'Finance') & (df['Salary'] > 80000)]
    print("\nFinance employees with salary > $80k:")
    print(finance_high)

    # Names and ages of employees 25-30
    age_range = df.loc[df['Age'].between(25, 30), ['Name', 'Age']]
    print("\nEmployees aged 25-30:")
    print(age_range)

    # Names containing 'e' (case-insensitive)
    names_with_e = df[df['Name'].str.contains('e', case=False)]
    print("\nNames containing 'e':")
    print(names_with_e[['Name']])

def exercise_3():
    """GroupBy and aggregation"""
    print("\nExercise 3: GroupBy and Aggregation")
    print("-" * 40)

    # Create sales data
    np.random.seed(42)
    sales = pd.DataFrame({
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 20),
        'Product': np.random.choice(['A', 'B', 'C'], 20),
        'Sales': np.random.randint(100, 1000, 20),
        'Quantity': np.random.randint(1, 10, 20)
    })

    print("Sales data (first 10 rows):")
    print(sales.head(10))

    # Find region with highest total sales
    region_sales = sales.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    print("\nTotal sales by region:")
    print(region_sales)
    print(f"Highest sales region: {region_sales.index[0]} (${region_sales.iloc[0]})")

    # Average quantity per product per region
    avg_qty = sales.groupby(['Region', 'Product'])['Quantity'].mean().round(2)
    print("\nAverage quantity by Region and Product:")
    print(avg_qty)

    # Products where max sale > 3x mean sale
    product_stats = sales.groupby('Product')['Sales'].agg(['mean', 'max'])
    product_stats['Ratio'] = product_stats['max'] / product_stats['mean']
    high_variance = product_stats[product_stats['Ratio'] > 3]
    print("\nProducts with max > 3x mean:")
    print(high_variance)

def exercise_4():
    """Merging DataFrames"""
    print("\nExercise 4: Merging DataFrames")
    print("-" * 40)

    # Create student and grades DataFrames
    students = pd.DataFrame({
        'StudentID': [1, 2, 3, 4],
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Major': ['CS', 'Math', 'CS', 'Physics']
    })

    grades = pd.DataFrame({
        'StudentID': [1, 2, 3, 5],
        'Course': ['Math', 'CS', 'Physics', 'Math'],
        'Grade': [85, 92, 78, 88]
    })

    print("Students:")
    print(students)
    print("\nGrades:")
    print(grades)

    # Inner join
    inner = pd.merge(students, grades, on='StudentID', how='inner')
    print("\nInner join:")
    print(inner)

    # Left join
    left = pd.merge(students, grades, on='StudentID', how='left')
    print("\nLeft join:")
    print(left)

    # Right join
    right = pd.merge(students, grades, on='StudentID', how='right')
    print("\nRight join:")
    print(right)

    # Outer join
    outer = pd.merge(students, grades, on='StudentID', how='outer')
    print("\nOuter join:")
    print(outer)

def exercise_5():
    """Time series operations"""
    print("\nExercise 5: Time Series Operations")
    print("-" * 40)

    # Create daily sales data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    sales_data = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.randint(100, 500, 60),
        'DayOfWeek': dates.day_name()
    })

    print("Daily sales (first 10 days):")
    print(sales_data.head(10))

    # Resample to weekly
    sales_data_indexed = sales_data.set_index('Date')
    weekly_sales = sales_data_indexed['Sales'].resample('W').sum()
    print("\nWeekly total sales:")
    print(weekly_sales)

    # Average sales by day of week
    avg_by_dow = sales_data.groupby('DayOfWeek')['Sales'].mean().sort_values(ascending=False)
    print("\nAverage sales by day of week:")
    print(avg_by_dow)
    print(f"Highest sales day: {avg_by_dow.index[0]} (${avg_by_dow.iloc[0]:.2f})")

    # 7-day moving average
    sales_data_indexed['MA_7'] = sales_data_indexed['Sales'].rolling(7).mean()
    print("\nLast 10 days with 7-day moving average:")
    print(sales_data_indexed[['Sales', 'MA_7']].tail(10).round(2))

def exercise_6():
    """Reading and cleaning data"""
    print("\nExercise 6: Reading and Cleaning Data")
    print("-" * 40)

    # CSV with missing values
    csv_data = """Name,Age,Score,City
Alice,25,85,New York
Bob,NA,92,London
Charlie,35,,Paris
David,28,78,
Eve,32,88,Berlin"""

    df = pd.read_csv(io.StringIO(csv_data), na_values=['NA', ''])
    print("Data with missing values:")
    print(df)

    print("\nNull counts:")
    print(df.isnull().sum())

    # Fill missing values
    df_filled = df.copy()
    df_filled['Age'].fillna(df_filled['Age'].mean(), inplace=True)
    df_filled['Score'].fillna(df_filled['Score'].median(), inplace=True)
    df_filled['City'].fillna('Unknown', inplace=True)

    print("\nAfter filling missing values:")
    print(df_filled)

def exercise_7():
    """Complex aggregation"""
    print("\nExercise 7: Complex Aggregation")
    print("-" * 40)

    # Create transaction data
    np.random.seed(42)
    transactions = pd.DataFrame({
        'TransactionID': range(1, 21),
        'Date': pd.date_range('2024-01-01', periods=20, freq='D'),
        'Product': np.random.choice(['Laptop', 'Phone', 'Tablet'], 20),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 20),
        'Amount': np.random.randint(100, 1000, 20),
        'Quantity': np.random.randint(1, 5, 20)
    })

    print("Transaction data (first 10):")
    print(transactions.head(10))

    # Multiple aggregations by product
    product_stats = transactions.groupby('Product').agg({
        'Amount': ['count', 'sum', 'mean', 'min', 'max'],
        'Quantity': ['sum', 'mean']
    }).round(2)

    print("\nProduct statistics:")
    print(product_stats)

    # Top 3 regions by total amount
    top_regions = transactions.groupby('Region')['Amount'].sum().nlargest(3)
    print("\nTop 3 regions by total amount:")
    print(top_regions)

def exercise_8():
    """Data transformation"""
    print("\nExercise 8: Data Transformation")
    print("-" * 40)

    # Create employee data
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Department': ['IT', 'IT', 'HR', 'HR', 'Finance'],
        'Salary': [75000, 80000, 65000, 70000, 85000]
    })

    print("Original data:")
    print(df)

    # Add department average salary
    df['Dept_Avg_Salary'] = df.groupby('Department')['Salary'].transform('mean')
    print("\nWith department average:")
    print(df)

    # Add percentage of department average
    df['Pct_of_Dept_Avg'] = (df['Salary'] / df['Dept_Avg_Salary'] * 100).round(2)
    print("\nWith percentage of department average:")
    print(df)

    # Normalize salary within department
    df['Salary_Normalized'] = df.groupby('Department')['Salary'].transform(
        lambda x: (x - x.mean()) / x.std()
    ).round(2)
    print("\nWith normalized salary:")
    print(df)

def exercise_9():
    """Multi-DataFrame joins"""
    print("\nExercise 9: Multi-DataFrame Joins")
    print("-" * 40)

    # Create three related DataFrames
    orders = pd.DataFrame({
        'OrderID': [1, 2, 3, 4],
        'CustomerID': [101, 102, 103, 101],
        'ProductID': [1001, 1002, 1001, 1003],
        'Quantity': [2, 1, 3, 1]
    })

    customers = pd.DataFrame({
        'CustomerID': [101, 102, 103],
        'CustomerName': ['Alice', 'Bob', 'Charlie'],
        'City': ['New York', 'London', 'Paris']
    })

    products = pd.DataFrame({
        'ProductID': [1001, 1002, 1003],
        'ProductName': ['Laptop', 'Mouse', 'Keyboard'],
        'Price': [1000, 25, 75]
    })

    print("Orders:")
    print(orders)
    print("\nCustomers:")
    print(customers)
    print("\nProducts:")
    print(products)

    # Merge all three
    result = orders.merge(customers, on='CustomerID')
    result = result.merge(products, on='ProductID')
    result['Total'] = result['Quantity'] * result['Price']

    print("\nComplete order information:")
    print(result[['OrderID', 'CustomerName', 'ProductName', 'Quantity', 'Price', 'Total']])

    # Summary by customer
    customer_totals = result.groupby('CustomerName')['Total'].sum().sort_values(ascending=False)
    print("\nTotal spending by customer:")
    print(customer_totals)

def exercise_10():
    """Advanced time series"""
    print("\nExercise 10: Advanced Time Series")
    print("-" * 40)

    # Create monthly revenue data
    dates = pd.date_range('2023-01-01', periods=12, freq='ME')
    revenue = pd.DataFrame({
        'Date': dates,
        'Revenue': [100, 110, 105, 120, 130, 125, 140, 145, 150, 155, 160, 170]
    })

    print("Monthly revenue:")
    print(revenue)

    # Calculate month-over-month growth
    revenue['MoM_Growth'] = revenue['Revenue'].pct_change() * 100
    print("\nWith month-over-month growth:")
    print(revenue.round(2))

    # Calculate 3-month moving average
    revenue['MA_3'] = revenue['Revenue'].rolling(3).mean()
    print("\nWith 3-month moving average:")
    print(revenue.round(2))

    # Year-over-year comparison (if we had 2 years)
    print(f"\nTotal revenue 2023: ${revenue['Revenue'].sum()}")
    print(f"Average monthly revenue: ${revenue['Revenue'].mean():.2f}")
    print(f"Growth from Jan to Dec: {((revenue['Revenue'].iloc[-1] / revenue['Revenue'].iloc[0]) - 1) * 100:.1f}%")

def main():
    print("=" * 60)
    print("Pandas Exercises")
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
        exercise_9,
        exercise_10,
    ]

    for i, exercise in enumerate(exercises, 1):
        try:
            exercise()
        except Exception as e:
            print(f"\nError in exercise {i}: {e}")

    print("\n" + "=" * 60)
    print("Additional Practice:")
    print("1. Create pivot table from sales data (Product x Region)")
    print("2. Find records in left DataFrame not in right using indicator")
    print("3. Calculate cumulative sum for time series data")
    print("4. Group by multiple columns and apply custom function")
    print("5. Resample hourly data to daily and calculate various statistics")
    print("=" * 60)

if __name__ == "__main__":
    main()
