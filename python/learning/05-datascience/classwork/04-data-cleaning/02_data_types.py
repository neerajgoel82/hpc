"""
Data Types and Type Conversion
===============================
Working with data types, type conversion, and categorical data.

Topics:
- Data type inspection
- Type conversion
- Categorical data
- Type errors and handling
- Optimizing data types

Run: python 02_data_types.py
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 60)
    print("Data Types and Type Conversion")
    print("=" * 60)

    # 1. Data type inspection
    print("\n1. Data Type Inspection")
    print("-" * 40)

    df = pd.DataFrame({
        'Integer': [1, 2, 3, 4, 5],
        'Float': [1.1, 2.2, 3.3, 4.4, 5.5],
        'String': ['A', 'B', 'C', 'D', 'E'],
        'Boolean': [True, False, True, False, True],
        'Mixed': [1, '2', 3.0, '4', 5]
    })

    print("DataFrame:")
    print(df)

    print("\nData types:")
    print(df.dtypes)

    print("\nDetailed info:")
    df.info()

    print(f"\nMemory usage: {df.memory_usage(deep=True).sum()} bytes")

    # 2. Type conversion
    print("\n2. Type Conversion")
    print("-" * 40)

    # String to numeric
    df_convert = pd.DataFrame({
        'Numbers': ['1', '2', '3', '4', '5']
    })

    print("Original (strings):")
    print(df_convert)
    print(f"Type: {df_convert['Numbers'].dtype}")

    df_convert['Numbers'] = pd.to_numeric(df_convert['Numbers'])
    print("\nAfter to_numeric:")
    print(f"Type: {df_convert['Numbers'].dtype}")

    # Handling errors in conversion
    df_errors = pd.DataFrame({
        'Values': ['1', '2', 'three', '4', '5']
    })

    print("\nData with non-numeric values:")
    print(df_errors)

    # Coerce errors to NaN
    df_errors['Values_Numeric'] = pd.to_numeric(df_errors['Values'], errors='coerce')
    print("\nAfter to_numeric with errors='coerce':")
    print(df_errors)

    # 3. Datetime conversion
    print("\n3. DateTime Conversion")
    print("-" * 40)

    df_dates = pd.DataFrame({
        'Date_String': ['2024-01-01', '2024-02-15', '2024-03-30'],
        'Date_Custom': ['01/15/2024', '02/20/2024', '03/25/2024']
    })

    print("Original:")
    print(df_dates)
    print(f"\nTypes:\n{df_dates.dtypes}")

    # Convert to datetime
    df_dates['Date_String'] = pd.to_datetime(df_dates['Date_String'])
    df_dates['Date_Custom'] = pd.to_datetime(df_dates['Date_Custom'], format='%m/%d/%Y')

    print("\nAfter conversion:")
    print(df_dates)
    print(f"\nTypes:\n{df_dates.dtypes}")

    # Extract components
    df_dates['Year'] = df_dates['Date_String'].dt.year
    df_dates['Month'] = df_dates['Date_String'].dt.month
    df_dates['Day'] = df_dates['Date_String'].dt.day

    print("\nWith extracted components:")
    print(df_dates)

    # 4. Categorical data
    print("\n4. Categorical Data")
    print("-" * 40)

    df_cat = pd.DataFrame({
        'Color': ['Red', 'Blue', 'Red', 'Green', 'Blue', 'Red', 'Green', 'Blue']
    })

    print("Original (object type):")
    print(df_cat)
    print(f"Type: {df_cat['Color'].dtype}")
    print(f"Memory: {df_cat.memory_usage(deep=True)['Color']} bytes")

    # Convert to categorical
    df_cat['Color'] = df_cat['Color'].astype('category')

    print("\nAfter converting to categorical:")
    print(f"Type: {df_cat['Color'].dtype}")
    print(f"Memory: {df_cat.memory_usage(deep=True)['Color']} bytes")
    print(f"Categories: {df_cat['Color'].cat.categories.tolist()}")

    # Categorical with ordering
    df_rating = pd.DataFrame({
        'Rating': ['Good', 'Excellent', 'Poor', 'Good', 'Excellent', 'Fair']
    })

    print("\nRating data:")
    print(df_rating)

    # Create ordered categorical
    rating_order = ['Poor', 'Fair', 'Good', 'Excellent']
    df_rating['Rating'] = pd.Categorical(
        df_rating['Rating'],
        categories=rating_order,
        ordered=True
    )

    print("\nOrdered categorical:")
    print(df_rating['Rating'])
    print(f"Is ordered: {df_rating['Rating'].cat.ordered}")

    # Sort by categorical order
    df_rating_sorted = df_rating.sort_values('Rating')
    print("\nSorted by rating:")
    print(df_rating_sorted)

    # 5. Type optimization
    print("\n5. Memory Optimization")
    print("-" * 40)

    # Create large DataFrame
    df_opt = pd.DataFrame({
        'ID': range(1000),
        'Category': np.random.choice(['A', 'B', 'C'], 1000),
        'Value': np.random.randint(0, 100, 1000)
    })

    print("Original memory usage:")
    print(df_opt.memory_usage(deep=True))
    print(f"Total: {df_opt.memory_usage(deep=True).sum()} bytes")

    # Optimize types
    df_opt['ID'] = df_opt['ID'].astype('int16')  # Was int64
    df_opt['Category'] = df_opt['Category'].astype('category')  # Was object
    df_opt['Value'] = df_opt['Value'].astype('int8')  # Was int64

    print("\nAfter optimization:")
    print(df_opt.memory_usage(deep=True))
    print(f"Total: {df_opt.memory_usage(deep=True).sum()} bytes")

    # 6. Boolean conversion
    print("\n6. Boolean Conversion")
    print("-" * 40)

    df_bool = pd.DataFrame({
        'Flag': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Status': [1, 0, 1, 0, 1],
        'Active': ['True', 'False', 'True', 'False', 'True']
    })

    print("Original:")
    print(df_bool)

    # Convert to boolean
    df_bool['Flag_Bool'] = df_bool['Flag'].map({'Yes': True, 'No': False})
    df_bool['Status_Bool'] = df_bool['Status'].astype(bool)
    df_bool['Active_Bool'] = df_bool['Active'].map({'True': True, 'False': False})

    print("\nWith boolean columns:")
    print(df_bool)
    print(f"\nTypes:\n{df_bool.dtypes}")

    print("\n" + "=" * 60)
    print("Summary - Data Type Operations:")
    print("  1. .dtypes - Check data types")
    print("  2. .astype() - Convert types")
    print("  3. pd.to_numeric() - Convert to numeric with error handling")
    print("  4. pd.to_datetime() - Convert to datetime")
    print("  5. .astype('category') - Create categorical data")
    print("  6. Memory optimization - Use smaller int types")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Convert string column '1,234' to numeric (remove commas)")
    print("2. Create ordered categorical for education levels")
    print("3. Optimize memory usage of a large DataFrame")
    print("4. Handle mixed types with errors in conversion")
    print("=" * 60)

if __name__ == "__main__":
    main()
