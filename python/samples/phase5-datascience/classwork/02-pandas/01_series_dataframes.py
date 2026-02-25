"""
Pandas Series and DataFrames
============================
Introduction to Pandas data structures.

Topics:
- Series creation and operations
- DataFrame creation and manipulation
- Basic DataFrame operations
- Column and row selection
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("Pandas Series and DataFrames")
    print("=" * 60)
    
    # 1. Pandas Series
    print("\n1. Pandas Series")
    print("-" * 40)
    
    # Create from list
    s1 = pd.Series([10, 20, 30, 40, 50])
    print("Series from list:")
    print(s1)
    
    # Create with custom index
    s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
    print("\nSeries with custom index:")
    print(s2)
    
    # Create from dictionary
    s3 = pd.Series({'A': 100, 'B': 200, 'C': 300})
    print("\nSeries from dictionary:")
    print(s3)
    
    # Series operations
    print(f"\nMean: {s1.mean()}")
    print(f"Sum: {s1.sum()}")
    print(f"Max: {s1.max()}")
    
    # 2. Creating DataFrames
    print("\n2. Creating DataFrames")
    print("-" * 40)
    
    # From dictionary
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 28],
        'City': ['New York', 'London', 'Paris', 'Tokyo'],
        'Salary': [70000, 80000, 90000, 75000]
    }
    
    df = pd.DataFrame(data)
    print("DataFrame from dictionary:")
    print(df)
    
    # From list of lists
    df2 = pd.DataFrame([
        ['Eve', 32, 'Berlin', 85000],
        ['Frank', 29, 'Sydney', 78000]
    ], columns=['Name', 'Age', 'City', 'Salary'])
    
    print("\nDataFrame from list of lists:")
    print(df2)
    
    # 3. DataFrame attributes
    print("\n3. DataFrame Attributes")
    print("-" * 40)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Index: {df.index.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    
    # 4. Viewing data
    print("\n4. Viewing Data")
    print("-" * 40)
    
    print("First 3 rows:")
    print(df.head(3))
    
    print("\nLast 2 rows:")
    print(df.tail(2))
    
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nDataFrame info:")
    df.info()
    
    # 5. Column selection
    print("\n5. Column Selection")
    print("-" * 40)
    
    # Single column (returns Series)
    print("Names column (Series):")
    print(df['Name'])
    
    # Multiple columns (returns DataFrame)
    print("\nName and Age columns:")
    print(df[['Name', 'Age']])
    
    # 6. Row selection
    print("\n6. Row Selection")
    print("-" * 40)
    
    # iloc (by position)
    print("First row using iloc[0]:")
    print(df.iloc[0])
    
    print("\nFirst two rows using iloc[0:2]:")
    print(df.iloc[0:2])
    
    # loc (by label/index)
    print("\nRow with index 1 using loc[1]:")
    print(df.loc[1])
    
    # 7. Basic operations
    print("\n7. Basic Operations")
    print("-" * 40)
    
    # Add new column
    df['Bonus'] = df['Salary'] * 0.1
    print("DataFrame with Bonus column:")
    print(df)
    
    # Calculate column statistics
    print(f"\nAverage salary: ${df['Salary'].mean():.2f}")
    print(f"Max age: {df['Age'].max()}")
    print(f"Min salary: ${df['Salary'].min():.2f}")
    
    # Filter rows
    print("\nPeople with salary > 75000:")
    high_salary = df[df['Salary'] > 75000]
    print(high_salary)
    
    # Sort values
    print("\nSorted by age (descending):")
    sorted_df = df.sort_values('Age', ascending=False)
    print(sorted_df)
    
    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create DataFrame with student data (name, grade, subject)")
    print("2. Calculate average grade per subject")
    print("3. Find students with grades > 80")
    print("4. Add a 'Pass/Fail' column (Pass if grade >= 70)")
    print("=" * 60)

if __name__ == "__main__":
    main()
