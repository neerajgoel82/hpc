"""
Merging and Joining DataFrames
===============================
Combine multiple DataFrames using merge, join, and concat.

Topics:
- Concatenating DataFrames
- Merging on keys
- Different join types (inner, outer, left, right)
- Joining on index
- Handling duplicate columns

Run: python 05_merging_joining.py
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("Merging and Joining DataFrames")
    print("=" * 60)

    # 1. Concatenating DataFrames
    print("\n1. Concatenating DataFrames (pd.concat)")
    print("-" * 40)

    df1 = pd.DataFrame({
        'A': ['A0', 'A1', 'A2'],
        'B': ['B0', 'B1', 'B2']
    })

    df2 = pd.DataFrame({
        'A': ['A3', 'A4', 'A5'],
        'B': ['B3', 'B4', 'B5']
    })

    print("DataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)

    # Concatenate vertically (stack rows)
    result = pd.concat([df1, df2])
    print("\nConcatenated (vertical):")
    print(result)

    # Reset index
    result_reset = pd.concat([df1, df2], ignore_index=True)
    print("\nConcatenated with reset index:")
    print(result_reset)

    # Concatenate horizontally (stack columns)
    result_horiz = pd.concat([df1, df2], axis=1)
    print("\nConcatenated (horizontal):")
    print(result_horiz)

    # 2. Merging on a key column
    print("\n2. Merging on a Key Column")
    print("-" * 40)

    # Employee data
    employees = pd.DataFrame({
        'EmployeeID': [1, 2, 3, 4],
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'DepartmentID': [101, 102, 101, 103]
    })

    # Department data
    departments = pd.DataFrame({
        'DepartmentID': [101, 102, 103],
        'Department': ['HR', 'IT', 'Finance']
    })

    print("Employees:")
    print(employees)
    print("\nDepartments:")
    print(departments)

    # Inner merge (default)
    merged = pd.merge(employees, departments, on='DepartmentID')
    print("\nMerged (inner join):")
    print(merged)

    # 3. Different join types
    print("\n3. Different Join Types")
    print("-" * 40)

    # Create sample data with non-matching keys
    left = pd.DataFrame({
        'Key': ['A', 'B', 'C', 'D'],
        'Value1': [1, 2, 3, 4]
    })

    right = pd.DataFrame({
        'Key': ['B', 'C', 'D', 'E'],
        'Value2': [10, 20, 30, 40]
    })

    print("Left DataFrame:")
    print(left)
    print("\nRight DataFrame:")
    print(right)

    # Inner join (intersection)
    inner = pd.merge(left, right, on='Key', how='inner')
    print("\nInner join (intersection):")
    print(inner)

    # Outer join (union)
    outer = pd.merge(left, right, on='Key', how='outer')
    print("\nOuter join (union):")
    print(outer)

    # Left join (all from left)
    left_join = pd.merge(left, right, on='Key', how='left')
    print("\nLeft join (all from left):")
    print(left_join)

    # Right join (all from right)
    right_join = pd.merge(left, right, on='Key', how='right')
    print("\nRight join (all from right):")
    print(right_join)

    # 4. Merging on multiple keys
    print("\n4. Merging on Multiple Keys")
    print("-" * 40)

    sales = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
        'Product': ['A', 'B', 'A', 'B'],
        'Quantity': [10, 15, 12, 18]
    })

    prices = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-01', '2024-01-02'],
        'Product': ['A', 'B', 'A'],
        'Price': [100, 150, 105]
    })

    print("Sales:")
    print(sales)
    print("\nPrices:")
    print(prices)

    # Merge on multiple columns
    result = pd.merge(sales, prices, on=['Date', 'Product'], how='left')
    print("\nMerged on Date and Product:")
    print(result)

    # Calculate revenue
    result['Revenue'] = result['Quantity'] * result['Price']
    print("\nWith calculated Revenue:")
    print(result)

    # 5. Merging on different column names
    print("\n5. Merging on Different Column Names")
    print("-" * 40)

    orders = pd.DataFrame({
        'OrderID': [1, 2, 3],
        'CustomerID': [101, 102, 103],
        'Amount': [250, 150, 300]
    })

    customers = pd.DataFrame({
        'ID': [101, 102, 103],
        'CustomerName': ['Alice', 'Bob', 'Charlie']
    })

    print("Orders:")
    print(orders)
    print("\nCustomers:")
    print(customers)

    # Merge with different column names
    result = pd.merge(orders, customers, left_on='CustomerID', right_on='ID')
    print("\nMerged (left_on='CustomerID', right_on='ID'):")
    print(result)

    # Drop redundant column
    result = result.drop('ID', axis=1)
    print("\nAfter dropping redundant ID column:")
    print(result)

    # 6. Joining on index
    print("\n6. Joining on Index")
    print("-" * 40)

    df1 = pd.DataFrame({
        'A': ['A0', 'A1', 'A2'],
        'B': ['B0', 'B1', 'B2']
    }, index=['K0', 'K1', 'K2'])

    df2 = pd.DataFrame({
        'C': ['C0', 'C1', 'C2'],
        'D': ['D0', 'D1', 'D2']
    }, index=['K0', 'K1', 'K3'])

    print("DataFrame 1 (indexed):")
    print(df1)
    print("\nDataFrame 2 (indexed):")
    print(df2)

    # Join on index
    result = df1.join(df2)
    print("\nJoined on index (left join default):")
    print(result)

    # Inner join
    result = df1.join(df2, how='inner')
    print("\nJoined on index (inner):")
    print(result)

    # 7. Handling suffix for duplicate columns
    print("\n7. Handling Duplicate Column Names")
    print("-" * 40)

    df1 = pd.DataFrame({
        'Key': ['A', 'B', 'C'],
        'Value': [1, 2, 3]
    })

    df2 = pd.DataFrame({
        'Key': ['A', 'B', 'D'],
        'Value': [10, 20, 40]
    })

    print("DataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)

    # Merge with suffixes
    result = pd.merge(df1, df2, on='Key', suffixes=('_left', '_right'))
    print("\nMerged with custom suffixes:")
    print(result)

    # 8. Indicator column
    print("\n8. Merge with Indicator Column")
    print("-" * 40)

    left = pd.DataFrame({'Key': ['A', 'B', 'C'], 'Value': [1, 2, 3]})
    right = pd.DataFrame({'Key': ['B', 'C', 'D'], 'Value': [10, 20, 40]})

    # Add indicator
    result = pd.merge(left, right, on='Key', how='outer', indicator=True)
    print("Merge with indicator:")
    print(result)

    # Count by source
    print("\nCounts by source:")
    print(result['_merge'].value_counts())

    # 9. Practical example: Sales analysis
    print("\n9. Practical Example: Sales Analysis")
    print("-" * 40)

    # Create sample data
    transactions = pd.DataFrame({
        'TransactionID': range(1, 6),
        'ProductID': [101, 102, 101, 103, 102],
        'CustomerID': [1, 2, 1, 3, 2],
        'Quantity': [2, 1, 3, 1, 2],
        'Date': pd.date_range('2024-01-01', periods=5)
    })

    products = pd.DataFrame({
        'ProductID': [101, 102, 103],
        'ProductName': ['Laptop', 'Mouse', 'Keyboard'],
        'Price': [1000, 25, 75]
    })

    customers = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'CustomerName': ['Alice', 'Bob', 'Charlie'],
        'City': ['New York', 'London', 'Paris']
    })

    print("Transactions:")
    print(transactions)

    # Merge all data
    result = transactions.merge(products, on='ProductID')
    result = result.merge(customers, on='CustomerID')

    # Calculate revenue
    result['Revenue'] = result['Quantity'] * result['Price']

    print("\nComplete sales data:")
    print(result[['CustomerName', 'ProductName', 'Quantity', 'Price', 'Revenue']])

    print("\nTotal revenue by customer:")
    print(result.groupby('CustomerName')['Revenue'].sum().sort_values(ascending=False))

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create student and grades DataFrames, merge on StudentID")
    print("2. Perform all 4 join types (inner, outer, left, right) and compare")
    print("3. Merge 3 DataFrames: Orders, Customers, Products")
    print("4. Find records that exist in left but not in right DataFrame")
    print("=" * 60)

if __name__ == "__main__":
    main()
