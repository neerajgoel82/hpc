"""
NumPy Broadcasting
==================
Broadcasting rules and operations for arrays of different shapes.

Topics:
- Broadcasting fundamentals
- Broadcasting rules
- Common broadcasting patterns
- Broadcasting with different dimensions
- Practical broadcasting examples

Run: python 03_broadcasting.py
"""

import numpy as np

def main():
    print("=" * 60)
    print("NumPy Broadcasting")
    print("=" * 60)

    # 1. Broadcasting basics
    print("\n1. Broadcasting Basics")
    print("-" * 40)

    # Scalar and array
    arr = np.array([1, 2, 3, 4, 5])
    scalar = 10

    print(f"Array: {arr}")
    print(f"Scalar: {scalar}")
    print(f"\nArray + Scalar: {arr + scalar}")
    print(f"Array * Scalar: {arr * scalar}")
    print(f"Array ** 2: {arr ** 2}")

    # 2D array and scalar
    arr2d = np.array([[1, 2, 3],
                      [4, 5, 6]])
    print(f"\n2D Array:\n{arr2d}")
    print(f"Array + 100:\n{arr2d + 100}")

    # 2. Broadcasting with 1D arrays
    print("\n2. Broadcasting with 1D Arrays")
    print("-" * 40)

    # Row vector
    a = np.array([1, 2, 3])
    # Column vector
    b = np.array([[10],
                  [20],
                  [30]])

    print(f"Row vector (shape {a.shape}): {a}")
    print(f"\nColumn vector (shape {b.shape}):\n{b}")

    result = a + b
    print(f"\nRow + Column (broadcasts to 3x3):\n{result}")
    print(f"Result shape: {result.shape}")

    # How broadcasting works here:
    print("\nHow it works:")
    print(f"a [1, 2, 3] broadcasts to:")
    print(f"  [[1, 2, 3],")
    print(f"   [1, 2, 3],")
    print(f"   [1, 2, 3]]")
    print(f"\nb [[10], [20], [30]] broadcasts to:")
    print(f"  [[10, 10, 10],")
    print(f"   [20, 20, 20],")
    print(f"   [30, 30, 30]]")

    # 3. Broadcasting rules
    print("\n3. Broadcasting Rules")
    print("-" * 40)

    print("Rule 1: If arrays have different number of dimensions,")
    print("        pad the smaller shape with ones on the left.")
    print("\nRule 2: If shapes don't match in any dimension,")
    print("        the dimension must be 1 in one of the arrays.")
    print("\nRule 3: After broadcasting, arrays act as if they")
    print("        had shapes that match element-wise.")

    # Examples of compatible shapes
    print("\nCompatible shapes:")
    print("  (3, 4) and (4,)    -> broadcast to (3, 4)")
    print("  (3, 1) and (3, 4)  -> broadcast to (3, 4)")
    print("  (3, 1) and (1, 4)  -> broadcast to (3, 4)")
    print("  (1, 5) and (5, 1)  -> broadcast to (5, 5)")

    print("\nIncompatible shapes:")
    print("  (3, 4) and (5,)    -> cannot broadcast")
    print("  (3, 4) and (3, 5)  -> cannot broadcast")

    # 4. Common broadcasting patterns
    print("\n4. Common Broadcasting Patterns")
    print("-" * 40)

    # Add a vector to each row of a matrix
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    vector = np.array([10, 20, 30])

    print(f"Matrix (3x3):\n{matrix}")
    print(f"\nVector (3,): {vector}")
    print(f"\nMatrix + Vector (add to each row):\n{matrix + vector}")

    # Add a vector to each column
    col_vector = np.array([[10], [20], [30]])
    print(f"\nColumn vector (3x1):\n{col_vector}")
    print(f"\nMatrix + Column (add to each column):\n{matrix + col_vector}")

    # 5. Practical examples
    print("\n5. Practical Broadcasting Examples")
    print("-" * 40)

    # Normalize data (subtract mean, divide by std)
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=float)

    print(f"Data:\n{data}")

    # Compute mean and std for each column
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    print(f"\nMean per column: {mean}")
    print(f"Std per column: {std}")

    # Normalize using broadcasting
    normalized = (data - mean) / std
    print(f"\nNormalized data:\n{normalized}")
    print(f"Normalized mean per column: {normalized.mean(axis=0)}")
    print(f"Normalized std per column: {normalized.std(axis=0)}")

    # 6. Distance calculation
    print("\n6. Broadcasting for Distance Calculations")
    print("-" * 40)

    # Points in 2D space
    points = np.array([[1, 2],
                       [3, 4],
                       [5, 6]])

    print(f"Points (3 points in 2D):\n{points}")

    # Compute pairwise distances
    # Reshape to (3, 1, 2) and (1, 3, 2)
    p1 = points[:, np.newaxis, :]  # Shape: (3, 1, 2)
    p2 = points[np.newaxis, :, :]  # Shape: (1, 3, 2)

    print(f"\np1 shape: {p1.shape}")
    print(f"p2 shape: {p2.shape}")

    # Euclidean distance using broadcasting
    diff = p1 - p2  # Shape: (3, 3, 2)
    distances = np.sqrt((diff ** 2).sum(axis=2))

    print(f"\nPairwise distances:\n{distances}")

    # 7. Broadcasting performance
    print("\n7. Broadcasting Performance Benefits")
    print("-" * 40)

    # Example: Matrix operation without and with broadcasting
    matrix = np.arange(12).reshape(3, 4)
    vector = np.array([1, 2, 3, 4])

    print(f"Matrix:\n{matrix}")
    print(f"Vector: {vector}")

    # Using broadcasting (efficient)
    result_broadcast = matrix + vector
    print(f"\nUsing broadcasting:\n{result_broadcast}")

    # Without broadcasting (inefficient, for demonstration)
    result_loop = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result_loop[i, j] = matrix[i, j] + vector[j]

    print(f"\nUsing loops (same result but slower):\n{result_loop}")
    print(f"\nResults are equal: {np.array_equal(result_broadcast, result_loop)}")

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create 4x4 matrix and normalize each row (mean=0, std=1)")
    print("2. Create two 1D arrays of length 5, compute outer product using broadcasting")
    print("3. Given matrix (5x3), subtract column minimums from each column")
    print("4. Create coordinate grid using broadcasting: x from 0-4, y from 0-3")
    print("=" * 60)

if __name__ == "__main__":
    main()
