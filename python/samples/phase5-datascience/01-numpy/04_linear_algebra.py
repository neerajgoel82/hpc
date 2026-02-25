"""
Linear Algebra with NumPy
==========================
Matrix operations, dot products, eigenvalues, and linear equation solving.

Topics:
- Matrix creation and operations
- Dot products and matrix multiplication
- Matrix properties (transpose, inverse, determinant)
- Solving linear systems
- Eigenvalues and eigenvectors

Run: python 04_linear_algebra.py
"""

import numpy as np

def main():
    print("=" * 60)
    print("Linear Algebra with NumPy")
    print("=" * 60)

    # 1. Matrix operations
    print("\n1. Basic Matrix Operations")
    print("-" * 40)

    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    B = np.array([[9, 8, 7],
                  [6, 5, 4],
                  [3, 2, 1]])

    print(f"Matrix A:\n{A}")
    print(f"\nMatrix B:\n{B}")

    # Element-wise operations
    print(f"\nA + B:\n{A + B}")
    print(f"\nA - B:\n{A - B}")
    print(f"\nA * B (element-wise):\n{A * B}")

    # Transpose
    print(f"\nA transpose (A.T):\n{A.T}")

    # 2. Dot product and matrix multiplication
    print("\n2. Dot Product and Matrix Multiplication")
    print("-" * 40)

    # Vector dot product
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])

    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")
    print(f"\nDot product (v1 · v2): {np.dot(v1, v2)}")
    print(f"Alternative: {v1 @ v2}")
    print(f"Manual calculation: {1*4 + 2*5 + 3*6} = {np.dot(v1, v2)}")

    # Matrix multiplication
    A = np.array([[1, 2],
                  [3, 4]])

    B = np.array([[5, 6],
                  [7, 8]])

    print(f"\nMatrix A (2x2):\n{A}")
    print(f"\nMatrix B (2x2):\n{B}")

    print(f"\nA @ B (matrix multiplication):\n{A @ B}")
    print(f"Also: np.dot(A, B):\n{np.dot(A, B)}")
    print(f"Also: np.matmul(A, B):\n{np.matmul(A, B)}")

    # Non-square matrices
    C = np.array([[1, 2, 3],
                  [4, 5, 6]])
    D = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])

    print(f"\nMatrix C (2x3):\n{C}")
    print(f"\nMatrix D (3x2):\n{D}")
    print(f"\nC @ D (result is 2x2):\n{C @ D}")
    print(f"C shape: {C.shape}, D shape: {D.shape}, Result shape: {(C @ D).shape}")

    # 3. Matrix properties
    print("\n3. Matrix Properties")
    print("-" * 40)

    A = np.array([[4, 2],
                  [3, 1]], dtype=float)

    print(f"Matrix A:\n{A}")

    # Determinant
    det = np.linalg.det(A)
    print(f"\nDeterminant: {det:.4f}")

    # Inverse
    A_inv = np.linalg.inv(A)
    print(f"\nInverse of A:\n{A_inv}")

    # Verify: A @ A_inv = I
    identity = A @ A_inv
    print(f"\nA @ A_inverse (should be identity):\n{identity}")

    # Trace (sum of diagonal elements)
    trace = np.trace(A)
    print(f"\nTrace (sum of diagonal): {trace}")

    # Rank
    rank = np.linalg.matrix_rank(A)
    print(f"Rank: {rank}")

    # 4. Solving linear systems
    print("\n4. Solving Linear Systems (Ax = b)")
    print("-" * 40)

    # System: 2x + 3y = 8
    #         3x + 4y = 11
    A = np.array([[2, 3],
                  [3, 4]], dtype=float)

    b = np.array([8, 11], dtype=float)

    print("System of equations:")
    print("  2x + 3y = 8")
    print("  3x + 4y = 11")

    print(f"\nCoefficient matrix A:\n{A}")
    print(f"\nRight-hand side b: {b}")

    # Solve using np.linalg.solve
    x = np.linalg.solve(A, b)
    print(f"\nSolution: x = {x[0]:.2f}, y = {x[1]:.2f}")

    # Verify solution
    verification = A @ x
    print(f"\nVerification (A @ x): {verification}")
    print(f"Original b: {b}")
    print(f"Match: {np.allclose(verification, b)}")

    # Larger system
    print("\nLarger system (3x3):")
    A = np.array([[3, 1, -1],
                  [2, 4, 1],
                  [-1, 2, 5]], dtype=float)

    b = np.array([4, 1, 1], dtype=float)

    print(f"A:\n{A}")
    print(f"b: {b}")

    x = np.linalg.solve(A, b)
    print(f"Solution x: {x}")
    print(f"Verification (A @ x): {A @ x}")

    # 5. Eigenvalues and eigenvectors
    print("\n5. Eigenvalues and Eigenvectors")
    print("-" * 40)

    A = np.array([[4, -2],
                  [1, 1]], dtype=float)

    print(f"Matrix A:\n{A}")

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print(f"\nEigenvalues: {eigenvalues}")
    print(f"\nEigenvectors (as columns):\n{eigenvectors}")

    # Verify: A @ v = λ @ v
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        print(f"\nEigenvalue {i+1}: {val:.4f}")
        print(f"Eigenvector {i+1}: {vec}")
        print(f"A @ v: {A @ vec}")
        print(f"λ * v: {val * vec}")
        print(f"Match: {np.allclose(A @ vec, val * vec)}")

    # 6. Other useful operations
    print("\n6. Other Useful Linear Algebra Operations")
    print("-" * 40)

    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 10]], dtype=float)

    print(f"Matrix A:\n{A}")

    # Norm (magnitude)
    norm = np.linalg.norm(A)
    print(f"\nFrobenius norm: {norm:.4f}")

    # Condition number
    cond = np.linalg.cond(A)
    print(f"Condition number: {cond:.4f}")

    # QR decomposition
    Q, R = np.linalg.qr(A)
    print(f"\nQR Decomposition:")
    print(f"Q (orthogonal):\n{Q}")
    print(f"\nR (upper triangular):\n{R}")
    print(f"\nQ @ R (should equal A):\n{Q @ R}")

    # Singular Value Decomposition (SVD)
    U, s, Vt = np.linalg.svd(A)
    print(f"\nSVD:")
    print(f"U shape: {U.shape}")
    print(f"Singular values: {s}")
    print(f"Vt shape: {Vt.shape}")

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Create 3x3 matrix, compute its inverse, verify A @ A_inv = I")
    print("2. Solve system: x + 2y + z = 6, 2x + y - z = 1, x - y + z = 2")
    print("3. Find eigenvalues of [[2, 1], [1, 2]]")
    print("4. Compute determinant and rank of a 4x4 random matrix")
    print("=" * 60)

if __name__ == "__main__":
    main()
