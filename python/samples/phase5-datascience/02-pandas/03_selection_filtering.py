"""
Data Selection and Filtering
=============================
Query and filter DataFrames using loc, iloc, and boolean indexing.

Topics:
- loc (label-based selection)
- iloc (integer-based selection)
- Boolean filtering
- Query method
- Column and row selection

Run: python 03_selection_filtering.py
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("Data Selection and Filtering")
    print("=" * 60)

    # Create sample DataFrame
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
        'Age': [25, 30, 35, 28, 32, 29],
        'City': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin', 'Sydney'],
        'Salary': [70000, 80000, 90000, 75000, 85000, 78000],
        'Department': ['HR', 'IT', 'IT', 'HR', 'Finance', 'IT']
    })

    print("\nSample DataFrame:")
    print(df)

    # 1. Column selection
    print("\n1. Column Selection")
    print("-" * 40)

    # Single column (returns Series)
    names = df['Name']
    print("Single column (Series):")
    print(names)
    print(f"Type: {type(names)}")

    # Single column as DataFrame
    names_df = df[['Name']]
    print("\nSingle column (DataFrame):")
    print(names_df)
    print(f"Type: {type(names_df)}")

    # Multiple columns
    subset = df[['Name', 'Age', 'Salary']]
    print("\nMultiple columns:")
    print(subset)

    # 2. Row selection with loc (label-based)
    print("\n2. Row Selection with loc (Label-Based)")
    print("-" * 40)

    # Single row by index
    print("Row with index 2:")
    print(df.loc[2])

    # Multiple rows
    print("\nRows 1, 2, 3:")
    print(df.loc[1:3])

    # Specific rows and columns
    print("\nRows 0-2, columns Name and Salary:")
    print(df.loc[0:2, ['Name', 'Salary']])

    # All rows, specific columns
    print("\nAll rows, Name and Age columns:")
    print(df.loc[:, ['Name', 'Age']])

    # 3. Row selection with iloc (integer-based)
    print("\n3. Row Selection with iloc (Integer-Based)")
    print("-" * 40)

    # Single row by position
    print("First row:")
    print(df.iloc[0])

    # Multiple rows
    print("\nFirst 3 rows:")
    print(df.iloc[0:3])

    # Specific rows and columns by position
    print("\nRows 0-2, columns 0 and 3:")
    print(df.iloc[0:3, [0, 3]])

    # Last 2 rows
    print("\nLast 2 rows:")
    print(df.iloc[-2:])

    # Every other row
    print("\nEvery other row:")
    print(df.iloc[::2])

    # 4. Boolean filtering
    print("\n4. Boolean Filtering")
    print("-" * 40)

    # Single condition
    high_salary = df[df['Salary'] > 80000]
    print("Employees with salary > $80,000:")
    print(high_salary)

    # Multiple conditions with AND (&)
    it_high_salary = df[(df['Department'] == 'IT') & (df['Salary'] > 75000)]
    print("\nIT employees with salary > $75,000:")
    print(it_high_salary)

    # Multiple conditions with OR (|)
    young_or_rich = df[(df['Age'] < 30) | (df['Salary'] > 85000)]
    print("\nEmployees under 30 OR earning > $85,000:")
    print(young_or_rich)

    # NOT condition (~)
    not_it = df[~(df['Department'] == 'IT')]
    print("\nNon-IT employees:")
    print(not_it)

    # 5. isin() method
    print("\n5. Filtering with isin()")
    print("-" * 40)

    # Check if value is in a list
    selected_cities = df[df['City'].isin(['New York', 'London', 'Tokyo'])]
    print("Employees in New York, London, or Tokyo:")
    print(selected_cities)

    # Check multiple columns
    tech_or_hr = df[df['Department'].isin(['IT', 'HR'])]
    print("\nIT or HR employees:")
    print(tech_or_hr)

    # 6. String methods
    print("\n6. String Filtering")
    print("-" * 40)

    # Contains
    has_o = df[df['City'].str.contains('o')]
    print("Cities containing 'o':")
    print(has_o)

    # Starts with
    starts_with_a = df[df['Name'].str.startswith('A')]
    print("\nNames starting with 'A':")
    print(starts_with_a)

    # Case-insensitive
    contains_new = df[df['City'].str.contains('new', case=False)]
    print("\nCities containing 'new' (case-insensitive):")
    print(contains_new)

    # 7. Query method
    print("\n7. Query Method")
    print("-" * 40)

    # Simple query
    result = df.query('Age > 30')
    print("Query: Age > 30")
    print(result)

    # Multiple conditions
    result = df.query('Age > 28 and Salary < 85000')
    print("\nQuery: Age > 28 and Salary < 85000")
    print(result)

    # Using variables
    min_salary = 75000
    result = df.query('Salary >= @min_salary')
    print(f"\nQuery: Salary >= {min_salary}")
    print(result)

    # 8. Between method
    print("\n8. Between Method")
    print("-" * 40)

    # Values between range
    mid_age = df[df['Age'].between(28, 32)]
    print("Employees aged 28-32:")
    print(mid_age)

    mid_salary = df[df['Salary'].between(75000, 85000)]
    print("\nEmployees earning $75k-$85k:")
    print(mid_salary)

    # 9. Combining loc with boolean conditions
    print("\n9. Advanced Selection: loc with Conditions")
    print("-" * 40)

    # Select specific columns for filtered rows
    high_earners = df.loc[df['Salary'] > 80000, ['Name', 'Salary']]
    print("Names and salaries of high earners:")
    print(high_earners)

    # Complex condition
    result = df.loc[
        (df['Age'] < 32) & (df['Department'] == 'IT'),
        ['Name', 'Age', 'Salary']
    ]
    print("\nYoung IT employees (Name, Age, Salary):")
    print(result)

    # 10. Modifying values with selection
    print("\n10. Modifying Values with Selection")
    print("-" * 40)

    # Create a copy to modify
    df_copy = df.copy()

    # Update single value
    df_copy.loc[0, 'Salary'] = 72000
    print("After updating first salary:")
    print(df_copy[['Name', 'Salary']])

    # Update based on condition
    df_copy = df.copy()
    df_copy.loc[df_copy['Department'] == 'IT', 'Salary'] *= 1.1
    print("\nAfter 10% raise for IT department:")
    print(df_copy[['Name', 'Department', 'Salary']])

    # Add new column based on condition
    df_copy['Seniority'] = 'Junior'
    df_copy.loc[df_copy['Age'] > 30, 'Seniority'] = 'Senior'
    print("\nWith Seniority column:")
    print(df_copy[['Name', 'Age', 'Seniority']])

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Select employees from Finance department with salary > $80k")
    print("2. Get Names and Cities of employees aged 25-30")
    print("3. Find employees whose names contain 'e' (case-insensitive)")
    print("4. Use query() to find employees: Age > 28 OR Department == 'HR'")
    print("=" * 60)

if __name__ == "__main__":
    main()
