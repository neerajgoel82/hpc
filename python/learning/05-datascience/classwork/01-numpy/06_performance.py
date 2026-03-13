"""
NumPy Performance Optimization
===============================
Vectorization vs loops, memory efficiency, and timing comparisons.

Topics:
- Vectorization benefits
- Loop vs vectorized operations
- Memory views and copies
- Broadcasting for performance
- Timing and benchmarking

Run: python 06_performance.py
"""

import numpy as np
import time

def main():
    print("=" * 60)
    print("NumPy Performance Optimization")
    print("=" * 60)

    # 1. Vectorization vs loops
    print("\n1. Vectorization vs Loops")
    print("-" * 40)

    # Create large array
    n = 1000000
    arr = np.random.randn(n)

    # Using loops (slow)
    start = time.time()
    result_loop = []
    for x in arr:
        result_loop.append(x ** 2)
    result_loop = np.array(result_loop)
    time_loop = time.time() - start

    # Using vectorization (fast)
    start = time.time()
    result_vectorized = arr ** 2
    time_vectorized = time.time() - start

    print(f"Array size: {n:,}")
    print(f"\nLoop time: {time_loop:.4f} seconds")
    print(f"Vectorized time: {time_vectorized:.4f} seconds")
    print(f"Speedup: {time_loop / time_vectorized:.1f}x")
    print(f"Results equal: {np.allclose(result_loop, result_vectorized)}")

    # 2. More vectorization examples
    print("\n2. More Vectorization Examples")
    print("-" * 40)

    arr1 = np.random.randn(100000)
    arr2 = np.random.randn(100000)

    # Addition with loops
    start = time.time()
    result_loop = np.zeros(len(arr1))
    for i in range(len(arr1)):
        result_loop[i] = arr1[i] + arr2[i]
    time_loop = time.time() - start

    # Addition vectorized
    start = time.time()
    result_vec = arr1 + arr2
    time_vec = time.time() - start

    print(f"Element-wise addition:")
    print(f"  Loop: {time_loop:.4f}s")
    print(f"  Vectorized: {time_vec:.4f}s")
    print(f"  Speedup: {time_loop / time_vec:.1f}x")

    # Conditional with loops
    arr = np.random.randn(100000)

    start = time.time()
    result_loop = []
    for x in arr:
        if x > 0:
            result_loop.append(x ** 2)
        else:
            result_loop.append(0)
    result_loop = np.array(result_loop)
    time_loop = time.time() - start

    # Conditional vectorized
    start = time.time()
    result_vec = np.where(arr > 0, arr ** 2, 0)
    time_vec = time.time() - start

    print(f"\nConditional operation:")
    print(f"  Loop: {time_loop:.4f}s")
    print(f"  Vectorized: {time_vec:.4f}s")
    print(f"  Speedup: {time_loop / time_vec:.1f}x")

    # 3. Memory: Views vs copies
    print("\n3. Memory: Views vs Copies")
    print("-" * 40)

    # Original array
    arr = np.arange(10)
    print(f"Original: {arr}")

    # View (shares memory)
    view = arr[2:7]
    print(f"\nView [2:7]: {view}")

    # Modifying view affects original
    view[0] = 999
    print(f"After modifying view[0] = 999:")
    print(f"  View: {view}")
    print(f"  Original: {arr}")

    # Copy (independent)
    arr = np.arange(10)
    copy = arr[2:7].copy()
    copy[0] = 999
    print(f"\nAfter modifying copy[0] = 999:")
    print(f"  Copy: {copy}")
    print(f"  Original: {arr}")

    # Check if array is view or copy
    arr = np.arange(10)
    view = arr[2:7]
    copy = arr[2:7].copy()

    print(f"\nBase of view is arr: {view.base is arr}")
    print(f"Base of copy is arr: {copy.base is arr}")

    # 4. In-place operations
    print("\n4. In-Place Operations (Memory Efficient)")
    print("-" * 40)

    # Create large array
    arr = np.random.randn(1000000)

    # Not in-place (creates new array)
    start = time.time()
    result = arr * 2
    time_new = time.time() - start

    # In-place (modifies existing array)
    arr = np.random.randn(1000000)
    start = time.time()
    arr *= 2
    time_inplace = time.time() - start

    print(f"Array size: {len(arr):,}")
    print(f"\nNew array (result = arr * 2): {time_new:.4f}s")
    print(f"In-place (arr *= 2): {time_inplace:.4f}s")
    print(f"Speedup: {time_new / time_inplace:.2f}x")

    # In-place operations
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"\nOriginal: {arr}")

    arr += 10  # In-place addition
    print(f"After arr += 10: {arr}")

    arr *= 2  # In-place multiplication
    print(f"After arr *= 2: {arr}")

    # 5. Broadcasting performance
    print("\n5. Broadcasting Performance")
    print("-" * 40)

    matrix = np.random.randn(1000, 1000)
    vector = np.random.randn(1000)

    # Using loops
    start = time.time()
    result_loop = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result_loop[i, j] = matrix[i, j] + vector[j]
    time_loop = time.time() - start

    # Using broadcasting
    start = time.time()
    result_broadcast = matrix + vector
    time_broadcast = time.time() - start

    print(f"Matrix shape: {matrix.shape}")
    print(f"Vector shape: {vector.shape}")
    print(f"\nLoop time: {time_loop:.4f}s")
    print(f"Broadcasting time: {time_broadcast:.4f}s")
    print(f"Speedup: {time_loop / time_broadcast:.1f}x")
    print(f"Results equal: {np.allclose(result_loop, result_broadcast)}")

    # 6. Contiguous arrays
    print("\n6. Contiguous Arrays (Cache Locality)")
    print("-" * 40)

    # Create array and transpose
    arr = np.random.randn(1000, 1000)
    arr_t = arr.T

    # Sum over rows (C-contiguous)
    start = time.time()
    for _ in range(100):
        result1 = arr.sum(axis=1)
    time_c_order = time.time() - start

    # Sum over rows (non-contiguous)
    start = time.time()
    for _ in range(100):
        result2 = arr_t.sum(axis=0)
    time_f_order = time.time() - start

    print(f"C-contiguous: {arr.flags['C_CONTIGUOUS']}")
    print(f"Transposed contiguous: {arr_t.flags['C_CONTIGUOUS']}")
    print(f"\nC-contiguous operations: {time_c_order:.4f}s")
    print(f"Non-contiguous operations: {time_f_order:.4f}s")
    print(f"Ratio: {time_f_order / time_c_order:.2f}x")

    # 7. Universal functions (ufuncs)
    print("\n7. Universal Functions (ufuncs)")
    print("-" * 40)

    arr = np.random.uniform(0, 10, 100000)

    # Python math (slow)
    import math
    start = time.time()
    result_python = [math.sqrt(x) for x in arr]
    time_python = time.time() - start

    # NumPy ufunc (fast)
    start = time.time()
    result_numpy = np.sqrt(arr)
    time_numpy = time.time() - start

    print(f"Python math.sqrt: {time_python:.4f}s")
    print(f"NumPy np.sqrt: {time_numpy:.4f}s")
    print(f"Speedup: {time_python / time_numpy:.1f}x")

    # 8. Aggregation operations
    print("\n8. Aggregation Operations")
    print("-" * 40)

    arr = np.random.randn(10000, 100)

    # Python sum
    start = time.time()
    result_python = sum(sum(row) for row in arr.tolist())
    time_python = time.time() - start

    # NumPy sum
    start = time.time()
    result_numpy = arr.sum()
    time_numpy = time.time() - start

    print(f"Array shape: {arr.shape}")
    print(f"\nPython sum: {time_python:.4f}s")
    print(f"NumPy sum: {time_numpy:.4f}s")
    print(f"Speedup: {time_python / time_numpy:.1f}x")

    # 9. Memory efficiency tips
    print("\n9. Memory Efficiency Tips")
    print("-" * 40)

    # Data type matters
    arr_float64 = np.random.randn(1000000)
    arr_float32 = arr_float64.astype(np.float32)

    print(f"float64 size: {arr_float64.nbytes / 1024 / 1024:.2f} MB")
    print(f"float32 size: {arr_float32.nbytes / 1024 / 1024:.2f} MB")
    print(f"Memory saved: {(1 - arr_float32.nbytes / arr_float64.nbytes) * 100:.0f}%")

    # Integer types
    arr_int64 = np.arange(1000000, dtype=np.int64)
    arr_int32 = np.arange(1000000, dtype=np.int32)
    arr_int16 = np.arange(1000, dtype=np.int16)

    print(f"\nint64 size: {arr_int64.itemsize} bytes per element")
    print(f"int32 size: {arr_int32.itemsize} bytes per element")
    print(f"int16 size: {arr_int16.itemsize} bytes per element")

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Compare loop vs vectorized calculation of arr1 * arr2 + arr3")
    print("2. Time different ways to normalize a matrix (subtract mean, divide std)")
    print("3. Create large array, compare copy vs view memory usage")
    print("4. Benchmark float64 vs float32 for matrix multiplication")
    print("=" * 60)

if __name__ == "__main__":
    main()
