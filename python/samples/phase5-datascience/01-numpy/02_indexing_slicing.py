"""
NumPy Indexing and Slicing
===========================
Advanced array indexing and slicing techniques for accessing array elements.

Topics:
- Basic indexing
- Slicing operations
- Boolean indexing
- Fancy indexing
- Multi-dimensional indexing

Run: python 02_indexing_slicing.py
"""

import numpy as np

def main():
    print("=" * 60)
    print("NumPy Indexing and Slicing")
    print("=" * 60)

    # 1. Basic indexing
    print("\n1. Basic Indexing")
    print("-" * 40)

    arr = np.array([10, 20, 30, 40, 50])
    print(f"Array: {arr}")

    print(f"\nFirst element (index 0): {arr[0]}")
    print(f"Third element (index 2): {arr[2]}")
    print(f"Last element (index -1): {arr[-1]}")
    print(f"Second to last (index -2): {arr[-2]}")

    # 2D array indexing
    arr2d = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
    print(f"\n2D Array:\n{arr2d}")

    print(f"\nElement at row 0, col 1: {arr2d[0, 1]}")
    print(f"Element at row 1, col 3: {arr2d[1, 3]}")
    print(f"Element at row 2, col 0: {arr2d[2, 0]}")
    print(f"First row: {arr2d[0]}")
    print(f"Last row: {arr2d[-1]}")

    # 2. Slicing
    print("\n2. Slicing [start:stop:step]")
    print("-" * 40)

    arr = np.arange(10)
    print(f"Array: {arr}")

    print(f"\nFirst 5 elements [0:5]: {arr[0:5]}")
    print(f"From index 3 to end [3:]: {arr[3:]}")
    print(f"Up to index 6 [:6]: {arr[:6]}")
    print(f"Every other element [::2]: {arr[::2]}")
    print(f"Reverse array [::-1]: {arr[::-1]}")
    print(f"Indices 2 to 8, step 2 [2:8:2]: {arr[2:8:2]}")

    # 2D slicing
    arr2d = np.arange(20).reshape(4, 5)
    print(f"\n2D Array (4x5):\n{arr2d}")

    print(f"\nFirst 2 rows:\n{arr2d[:2]}")
    print(f"\nFirst 2 rows, first 3 columns:\n{arr2d[:2, :3]}")
    print(f"\nAll rows, column 2:\n{arr2d[:, 2]}")
    print(f"\nEvery other row, every other column:\n{arr2d[::2, ::2]}")

    # 3. Boolean indexing
    print("\n3. Boolean Indexing")
    print("-" * 40)

    arr = np.array([1, 5, 10, 15, 20, 25, 30])
    print(f"Array: {arr}")

    # Create boolean mask
    mask = arr > 15
    print(f"\nBoolean mask (arr > 15): {mask}")
    print(f"Elements > 15: {arr[mask]}")

    # Direct boolean indexing
    print(f"\nElements > 10: {arr[arr > 10]}")
    print(f"Elements <= 15: {arr[arr <= 15]}")
    print(f"Even elements: {arr[arr % 2 == 0]}")

    # Multiple conditions
    print(f"\nElements between 10 and 25: {arr[(arr >= 10) & (arr <= 25)]}")
    print(f"Elements < 10 or > 25: {arr[(arr < 10) | (arr > 25)]}")

    # 2D boolean indexing
    arr2d = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    print(f"\n2D Array:\n{arr2d}")
    print(f"Elements > 5: {arr2d[arr2d > 5]}")

    # 4. Fancy indexing
    print("\n4. Fancy Indexing (Array of Indices)")
    print("-" * 40)

    arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    print(f"Array: {arr}")

    # Select specific indices
    indices = [0, 2, 5, 7]
    print(f"\nIndices {indices}: {arr[indices]}")

    # Negative indices
    indices = [-1, -3, -5]
    print(f"Indices {indices}: {arr[indices]}")

    # 2D fancy indexing
    arr2d = np.arange(12).reshape(3, 4)
    print(f"\n2D Array (3x4):\n{arr2d}")

    # Select specific rows
    row_indices = [0, 2]
    print(f"\nRows {row_indices}:\n{arr2d[row_indices]}")

    # Select specific elements
    rows = [0, 1, 2]
    cols = [1, 2, 3]
    print(f"\nElements at (0,1), (1,2), (2,3): {arr2d[rows, cols]}")

    # 5. Combining indexing methods
    print("\n5. Combining Indexing Methods")
    print("-" * 40)

    arr2d = np.arange(24).reshape(6, 4)
    print(f"Array (6x4):\n{arr2d}")

    # Slice then boolean
    print(f"\nFirst 4 rows:\n{arr2d[:4]}")
    subset = arr2d[:4]
    print(f"Elements > 10 in first 4 rows: {subset[subset > 10]}")

    # Boolean then fancy
    arr = np.array([5, 12, 8, 20, 3, 15, 18, 7])
    print(f"\nArray: {arr}")
    filtered = arr[arr > 10]
    print(f"Values > 10: {filtered}")
    print(f"First 2 of those: {filtered[[0, 1]]}")

    # 6. Modifying arrays with indexing
    print("\n6. Modifying Arrays with Indexing")
    print("-" * 40)

    arr = np.array([1, 2, 3, 4, 5])
    print(f"Original: {arr}")

    arr[2] = 99
    print(f"After arr[2] = 99: {arr}")

    arr[arr > 50] = 0
    print(f"After arr[arr > 50] = 0: {arr}")

    # Modify slice
    arr = np.arange(10)
    print(f"\nOriginal: {arr}")
    arr[3:7] = 100
    print(f"After arr[3:7] = 100: {arr}")

    # Modify with boolean indexing
    arr = np.array([1, 5, 10, 15, 20, 25])
    print(f"\nOriginal: {arr}")
    arr[arr % 2 == 0] *= 2  # Double even numbers
    print(f"After doubling even numbers: {arr}")

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create array [0-19], extract all even numbers using boolean indexing")
    print("2. Create 5x5 array, extract the 4 corner elements using fancy indexing")
    print("3. Create array, replace all values > mean with mean value")
    print("4. Create 4x4 array, extract the 2x2 center subarray using slicing")
    print("=" * 60)

if __name__ == "__main__":
    main()
