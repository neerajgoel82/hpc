"""
NumPy Arrays Basics
===================
Introduction to NumPy arrays, the foundation of numerical computing in Python.

Topics:
- Creating arrays
- Array attributes
- Basic operations
- Array types
"""

import numpy as np

def main():
    print("=" * 60)
    print("NumPy Arrays Basics")
    print("=" * 60)
    
    # 1. Creating arrays
    print("\n1. Creating Arrays")
    print("-" * 40)
    
    # From Python lists
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"From list: {arr1}")
    
    # 2D array
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"\n2D array:\n{arr2d}")
    
    # Using built-in functions
    zeros = np.zeros((3, 4))
    ones = np.ones((2, 3))
    identity = np.eye(3)
    
    print(f"\nZeros (3x4):\n{zeros}")
    print(f"\nOnes (2x3):\n{ones}")
    print(f"\nIdentity (3x3):\n{identity}")
    
    # Range of values
    arange_arr = np.arange(0, 10, 2)  # start, stop, step
    linspace_arr = np.linspace(0, 1, 5)  # start, stop, num_points
    
    print(f"\nArange (0 to 10, step 2): {arange_arr}")
    print(f"Linspace (0 to 1, 5 points): {linspace_arr}")
    
    # 2. Array attributes
    print("\n2. Array Attributes")
    print("-" * 40)
    
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    
    print(f"Array:\n{arr}")
    print(f"\nShape: {arr.shape}")
    print(f"Dimensions: {arr.ndim}")
    print(f"Size (total elements): {arr.size}")
    print(f"Data type: {arr.dtype}")
    print(f"Item size (bytes): {arr.itemsize}")
    
    # 3. Basic operations
    print("\n3. Basic Operations")
    print("-" * 40)
    
    a = np.array([1, 2, 3, 4])
    b = np.array([10, 20, 30, 40])
    
    print(f"a = {a}")
    print(f"b = {b}")
    
    # Element-wise operations
    print(f"\na + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    print(f"a ** 2 = {a ** 2}")
    
    # Scalar operations
    print(f"\na * 10 = {a * 10}")
    print(f"a + 5 = {a + 5}")
    
    # Universal functions
    print(f"\nsqrt(a) = {np.sqrt(a)}")
    print(f"exp(a) = {np.exp(a)}")
    print(f"sin(a) = {np.sin(a)}")
    
    # Aggregation functions
    print("\n4. Aggregation Functions")
    print("-" * 40)
    
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Array:\n{arr}")
    
    print(f"\nSum: {arr.sum()}")
    print(f"Mean: {arr.mean()}")
    print(f"Std: {arr.std()}")
    print(f"Min: {arr.min()}")
    print(f"Max: {arr.max()}")
    
    # Along axes
    print(f"\nSum along axis 0 (columns): {arr.sum(axis=0)}")
    print(f"Sum along axis 1 (rows): {arr.sum(axis=1)}")
    
    # 5. Array shapes and reshaping
    print("\n5. Reshaping Arrays")
    print("-" * 40)
    
    arr = np.arange(12)
    print(f"Original (12,): {arr}")
    
    reshaped = arr.reshape(3, 4)
    print(f"\nReshaped to (3, 4):\n{reshaped}")
    
    reshaped2 = arr.reshape(4, 3)
    print(f"\nReshaped to (4, 3):\n{reshaped2}")
    
    # Flatten
    flat = reshaped.flatten()
    print(f"\nFlattened: {flat}")
    
    # Transpose
    transposed = reshaped.T
    print(f"\nTransposed (4, 3):\n{transposed}")
    
    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create a 5x5 matrix of random integers between 1 and 100")
    print("2. Calculate mean, std, min, max of each row")
    print("3. Find all elements greater than 50")
    print("4. Replace all elements < 25 with 0")
    print("=" * 60)

if __name__ == "__main__":
    main()
