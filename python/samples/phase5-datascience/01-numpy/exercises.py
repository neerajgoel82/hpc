"""
NumPy Exercises
===============
Practice problems covering all NumPy concepts from this module.

Topics covered:
- Array creation and manipulation
- Indexing and slicing
- Broadcasting
- Linear algebra
- Random sampling
- Performance optimization

Run: python exercises.py
"""

import numpy as np

def exercise_1():
    """Array creation and basic operations"""
    print("\nExercise 1: Array Creation and Basic Operations")
    print("-" * 40)

    # Create array from 0 to 20
    arr = np.arange(21)
    print(f"Array 0-20: {arr}")

    # Extract even numbers
    evens = arr[arr % 2 == 0]
    print(f"Even numbers: {evens}")

    # Calculate statistics
    print(f"Mean: {arr.mean():.2f}")
    print(f"Std: {arr.std():.2f}")
    print(f"Sum: {arr.sum()}")

def exercise_2():
    """2D array manipulation"""
    print("\nExercise 2: 2D Array Manipulation")
    print("-" * 40)

    # Create 5x5 array
    arr = np.arange(25).reshape(5, 5)
    print(f"5x5 array:\n{arr}")

    # Extract corners using fancy indexing
    rows = [0, 0, 4, 4]
    cols = [0, 4, 0, 4]
    corners = arr[rows, cols]
    print(f"\nFour corners: {corners}")

    # Extract center 3x3
    center = arr[1:4, 1:4]
    print(f"\nCenter 3x3:\n{center}")

def exercise_3():
    """Broadcasting and normalization"""
    print("\nExercise 3: Broadcasting and Normalization")
    print("-" * 40)

    # Create 4x4 matrix
    np.random.seed(42)
    matrix = np.random.randn(4, 4)
    print(f"Original matrix:\n{matrix}")

    # Normalize each row (mean=0, std=1)
    row_means = matrix.mean(axis=1, keepdims=True)
    row_stds = matrix.std(axis=1, keepdims=True)
    normalized = (matrix - row_means) / row_stds

    print(f"\nNormalized (by row):\n{normalized}")
    print(f"\nRow means after normalization: {normalized.mean(axis=1)}")
    print(f"Row stds after normalization: {normalized.std(axis=1)}")

def exercise_4():
    """Linear algebra operations"""
    print("\nExercise 4: Linear Algebra")
    print("-" * 40)

    # Create 3x3 matrix
    A = np.array([[2, 1, 1],
                  [1, 3, 2],
                  [1, 2, 2]], dtype=float)

    print(f"Matrix A:\n{A}")

    # Compute inverse
    A_inv = np.linalg.inv(A)
    print(f"\nInverse of A:\n{A_inv}")

    # Verify A @ A_inv = I
    identity = A @ A_inv
    print(f"\nA @ A_inv (should be I):\n{identity}")
    print(f"Is identity: {np.allclose(identity, np.eye(3))}")

    # Determinant and rank
    det = np.linalg.det(A)
    rank = np.linalg.matrix_rank(A)
    print(f"\nDeterminant: {det:.4f}")
    print(f"Rank: {rank}")

def exercise_5():
    """Solving linear system"""
    print("\nExercise 5: Solve Linear System")
    print("-" * 40)

    # System: x + 2y + z = 6
    #         2x + y - z = 1
    #         x - y + z = 2

    A = np.array([[1, 2, 1],
                  [2, 1, -1],
                  [1, -1, 1]], dtype=float)

    b = np.array([6, 1, 2], dtype=float)

    print("System:")
    print("  x + 2y + z = 6")
    print("  2x + y - z = 1")
    print("  x - y + z = 2")

    # Solve
    solution = np.linalg.solve(A, b)
    print(f"\nSolution: x={solution[0]:.2f}, y={solution[1]:.2f}, z={solution[2]:.2f}")

    # Verify
    verification = A @ solution
    print(f"\nVerification (A @ solution): {verification}")
    print(f"Expected b: {b}")
    print(f"Match: {np.allclose(verification, b)}")

def exercise_6():
    """Eigenvalues and eigenvectors"""
    print("\nExercise 6: Eigenvalues and Eigenvectors")
    print("-" * 40)

    # Matrix [[2, 1], [1, 2]]
    A = np.array([[2, 1],
                  [1, 2]], dtype=float)

    print(f"Matrix A:\n{A}")

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print(f"\nEigenvalues: {eigenvalues}")
    print(f"\nEigenvectors:\n{eigenvectors}")

    # Verify
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        Av = A @ vec
        lambda_v = val * vec
        print(f"\nEigenvalue {i+1}: {val:.4f}")
        print(f"  A @ v = {Av}")
        print(f"  λ * v = {lambda_v}")
        print(f"  Match: {np.allclose(Av, lambda_v)}")

def exercise_7():
    """Random sampling and statistics"""
    print("\nExercise 7: Random Sampling")
    print("-" * 40)

    # Simulate 10000 dice rolls
    np.random.seed(42)
    dice_rolls = np.random.randint(1, 7, size=10000)

    print(f"Number of rolls: {len(dice_rolls)}")
    print(f"Mean: {dice_rolls.mean():.4f} (expected: 3.5)")
    print(f"Std: {dice_rolls.std():.4f}")

    # Count each outcome
    for i in range(1, 7):
        count = (dice_rolls == i).sum()
        percentage = count / len(dice_rolls) * 100
        print(f"  {i}: {count} times ({percentage:.1f}%)")

def exercise_8():
    """Random walk simulation"""
    print("\nExercise 8: Random Walk")
    print("-" * 40)

    # Random walk: cumulative sum of random steps {-1, +1}
    np.random.seed(42)
    steps = np.random.choice([-1, 1], size=1000)
    position = np.cumsum(steps)

    print(f"Number of steps: {len(steps)}")
    print(f"Final position: {position[-1]}")
    print(f"Max position: {position.max()}")
    print(f"Min position: {position.min()}")
    print(f"First 20 positions: {position[:20]}")

def exercise_9():
    """Outer product using broadcasting"""
    print("\nExercise 9: Outer Product with Broadcasting")
    print("-" * 40)

    # Two arrays
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])

    print(f"Array a: {a}")
    print(f"Array b: {b}")

    # Outer product using broadcasting
    outer = a[:, np.newaxis] * b[np.newaxis, :]
    print(f"\nOuter product (5x5):\n{outer}")

    # Verify with np.outer
    outer_verify = np.outer(a, b)
    print(f"\nUsing np.outer:\n{outer_verify}")
    print(f"Results match: {np.array_equal(outer, outer_verify)}")

def exercise_10():
    """Performance comparison"""
    print("\nExercise 10: Performance - Vectorization")
    print("-" * 40)

    import time

    # Create arrays
    np.random.seed(42)
    a = np.random.randn(100000)
    b = np.random.randn(100000)
    c = np.random.randn(100000)

    # Using loops
    start = time.time()
    result_loop = []
    for i in range(len(a)):
        result_loop.append(a[i] * b[i] + c[i])
    result_loop = np.array(result_loop)
    time_loop = time.time() - start

    # Using vectorization
    start = time.time()
    result_vec = a * b + c
    time_vec = time.time() - start

    print(f"Array size: {len(a):,}")
    print(f"\nLoop time: {time_loop:.4f}s")
    print(f"Vectorized time: {time_vec:.4f}s")
    print(f"Speedup: {time_loop / time_vec:.1f}x")
    print(f"Results equal: {np.allclose(result_loop, result_vec)}")

def main():
    print("=" * 60)
    print("NumPy Exercises")
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
    print("1. Create 5x5 matrix with random ints, no duplicates (use np.random.choice)")
    print("2. Normalize matrix columns (mean=0, std=1) using broadcasting")
    print("3. Compute pairwise distances between 10 random 2D points")
    print("4. Create coordinate grid: x in [0, 4], y in [0, 3] using meshgrid")
    print("5. Time float64 vs float32 for 1000x1000 matrix multiplication")
    print("=" * 60)

if __name__ == "__main__":
    main()
