"""
Reading Data with Pandas
=========================
Loading data from various file formats (CSV, Excel, JSON).

Topics:
- Reading CSV files
- Reading Excel files
- Reading JSON files
- Reading from strings and dictionaries
- Handling common reading issues

Run: python 02_reading_data.py
"""

import pandas as pd
import numpy as np
import io

def main():
    print("=" * 60)
    print("Reading Data with Pandas")
    print("=" * 60)

    # 1. Reading CSV from string
    print("\n1. Reading CSV Data")
    print("-" * 40)

    # CSV data as string (simulating file content)
    csv_data = """Name,Age,City,Salary
Alice,25,New York,70000
Bob,30,London,80000
Charlie,35,Paris,90000
David,28,Tokyo,75000
Eve,32,Berlin,85000"""

    # Read CSV from string
    df = pd.read_csv(io.StringIO(csv_data))
    print("Data loaded from CSV:")
    print(df)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # 2. CSV reading options
    print("\n2. CSV Reading Options")
    print("-" * 40)

    # CSV with custom delimiter
    tsv_data = """Name\tAge\tScore
Alice\t25\t85
Bob\t30\t92
Charlie\t35\t78"""

    df_tsv = pd.read_csv(io.StringIO(tsv_data), sep='\t')
    print("Tab-separated data:")
    print(df_tsv)

    # CSV with custom index column
    csv_with_index = """ID,Name,Score
1,Alice,85
2,Bob,92
3,Charlie,78"""

    df_indexed = pd.read_csv(io.StringIO(csv_with_index), index_col='ID')
    print("\nWith custom index:")
    print(df_indexed)

    # CSV with selected columns
    df_selected = pd.read_csv(io.StringIO(csv_data), usecols=['Name', 'Age', 'Salary'])
    print("\nSelected columns only:")
    print(df_selected)

    # 3. Handling missing data
    print("\n3. Handling Missing Data During Reading")
    print("-" * 40)

    csv_with_missing = """Name,Age,Score
Alice,25,85
Bob,NA,92
Charlie,35,
David,,78
Eve,32,N/A"""

    df_missing = pd.read_csv(io.StringIO(csv_with_missing), na_values=['NA', 'N/A', ''])
    print("Data with missing values:")
    print(df_missing)
    print(f"\nNull values per column:")
    print(df_missing.isnull().sum())

    # 4. Reading with data types
    print("\n4. Specifying Data Types")
    print("-" * 40)

    csv_data_types = """ID,Name,Score,Active
1,Alice,85.5,True
2,Bob,92.0,False
3,Charlie,78.5,True"""

    # Specify dtypes
    df_typed = pd.read_csv(
        io.StringIO(csv_data_types),
        dtype={'ID': int, 'Name': str, 'Score': float, 'Active': bool}
    )
    print("Data with specified types:")
    print(df_typed)
    print(f"\nData types:")
    print(df_typed.dtypes)

    # 5. Reading JSON
    print("\n5. Reading JSON Data")
    print("-" * 40)

    # JSON string (records orientation)
    json_data = """[
        {"Name": "Alice", "Age": 25, "City": "New York"},
        {"Name": "Bob", "Age": 30, "City": "London"},
        {"Name": "Charlie", "Age": 35, "City": "Paris"}
    ]"""

    df_json = pd.read_json(io.StringIO(json_data))
    print("Data from JSON (records):")
    print(df_json)

    # JSON with columns orientation
    json_columns = """{
        "Name": {"0": "Alice", "1": "Bob", "2": "Charlie"},
        "Age": {"0": 25, "1": 30, "2": 35},
        "City": {"0": "New York", "1": "London", "2": "Paris"}
    }"""

    df_json_cols = pd.read_json(io.StringIO(json_columns), orient='columns')
    print("\nData from JSON (columns orientation):")
    print(df_json_cols)

    # 6. Reading from dictionary
    print("\n6. Reading from Python Dictionary")
    print("-" * 40)

    # Dictionary to DataFrame
    data_dict = {
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
        'Price': [1200, 25, 75, 300],
        'Stock': [15, 150, 80, 25]
    }

    df_dict = pd.DataFrame(data_dict)
    print("DataFrame from dictionary:")
    print(df_dict)

    # List of dictionaries
    list_of_dicts = [
        {'Name': 'Alice', 'Age': 25, 'Score': 85},
        {'Name': 'Bob', 'Age': 30, 'Score': 92},
        {'Name': 'Charlie', 'Age': 35, 'Score': 78}
    ]

    df_list = pd.DataFrame(list_of_dicts)
    print("\nDataFrame from list of dictionaries:")
    print(df_list)

    # 7. Reading in chunks
    print("\n7. Reading Large Files in Chunks")
    print("-" * 40)

    # Simulate large CSV
    large_csv = "\n".join([f"Row{i},{i*10},{i*100}" for i in range(100)])
    large_csv = "ID,Value1,Value2\n" + large_csv

    print("Reading in chunks of 25 rows:")
    chunk_count = 0
    for chunk in pd.read_csv(io.StringIO(large_csv), chunksize=25):
        chunk_count += 1
        print(f"\nChunk {chunk_count}:")
        print(f"  Rows: {len(chunk)}")
        print(f"  First row ID: {chunk['ID'].iloc[0]}")
        print(f"  Last row ID: {chunk['ID'].iloc[-1]}")

    # 8. Common reading parameters
    print("\n8. Common Reading Parameters Summary")
    print("-" * 40)

    print("pd.read_csv() common parameters:")
    print("  filepath_or_buffer: file path or buffer")
    print("  sep: delimiter (default ',')")
    print("  header: row number for column names (default 0)")
    print("  index_col: column(s) to use as index")
    print("  usecols: columns to read")
    print("  dtype: data types for columns")
    print("  na_values: strings to recognize as NaN")
    print("  skiprows: rows to skip at start")
    print("  nrows: number of rows to read")
    print("  chunksize: read file in chunks")

    # 9. Creating sample data for practice
    print("\n9. Creating Sample Data for Practice")
    print("-" * 40)

    # Create sample DataFrame
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=10),
        'Product': np.random.choice(['A', 'B', 'C'], 10),
        'Sales': np.random.randint(100, 1000, 10),
        'Profit': np.random.uniform(0.1, 0.3, 10)
    })

    print("Sample DataFrame:")
    print(sample_df)

    # Convert to CSV string
    csv_string = sample_df.to_csv(index=False)
    print("\nAs CSV:")
    print(csv_string[:200] + "...")

    # Convert to JSON
    json_string = sample_df.to_json(orient='records', indent=2)
    print("\nAs JSON (first record):")
    print(json_string[:150] + "...")

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create CSV string with 5 students, read with custom index")
    print("2. Read CSV skipping first 2 rows using skiprows parameter")
    print("3. Create JSON data and read it with different orient options")
    print("4. Read large dataset in chunks, calculate sum per chunk")
    print("=" * 60)

if __name__ == "__main__":
    main()
