"""
Mathematical Operations
Element-wise operations, matrix operations, and aggregations

This module demonstrates:
- Element-wise operations (add, sub, mul, div)
- Matrix operations (matmul, dot, cross)
- Aggregation operations (sum, mean, std)
- Comparison operations
- Advanced mathematical functions

Run: python 03_math_operations.py
"""

import torch


def check_environment():
    """Display PyTorch environment information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()


def element_wise_operations():
    """Element-wise arithmetic operations."""
    print("1. Element-wise Operations")
    print("-" * 50)

    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([2.0, 3.0, 4.0, 5.0])

    print(f"Tensor a: {a}")
    print(f"Tensor b: {b}")

    # Addition
    add_result = a + b
    add_result2 = torch.add(a, b)
    print(f"\nAddition (a + b): {add_result}")

    # Subtraction
    sub_result = a - b
    print(f"Subtraction (a - b): {sub_result}")

    # Multiplication
    mul_result = a * b
    print(f"Multiplication (a * b): {mul_result}")

    # Division
    div_result = a / b
    print(f"Division (a / b): {div_result}")

    # Power
    pow_result = a ** 2
    print(f"Power (a ** 2): {pow_result}")

    # In-place operations (modify original)
    c = torch.tensor([1.0, 2.0, 3.0])
    c.add_(1.0)  # Add 1 to all elements
    print(f"\nIn-place add (c += 1): {c}")
    print()


def matrix_operations():
    """Matrix multiplication and operations."""
    print("2. Matrix Operations")
    print("-" * 50)

    # Matrix multiplication
    mat1 = torch.tensor([[1, 2], [3, 4]])
    mat2 = torch.tensor([[5, 6], [7, 8]])

    print(f"Matrix 1:\n{mat1}")
    print(f"Matrix 2:\n{mat2}")

    # Element-wise multiplication
    elem_mul = mat1 * mat2
    print(f"\nElement-wise multiplication:\n{elem_mul}")

    # Matrix multiplication
    matmul = torch.matmul(mat1, mat2)
    matmul2 = mat1 @ mat2  # Alternative syntax
    print(f"\nMatrix multiplication (matmul):\n{matmul}")

    # Matrix-vector multiplication
    vec = torch.tensor([1, 2])
    mat_vec = mat1 @ vec
    print(f"\nMatrix-vector multiplication:\n{mat_vec}")

    # Batch matrix multiplication
    batch1 = torch.randn(3, 2, 4)  # 3 matrices of size 2x4
    batch2 = torch.randn(3, 4, 5)  # 3 matrices of size 4x5
    batch_result = torch.bmm(batch1, batch2)  # Result: 3 matrices of 2x5
    print(f"\nBatch matrix multiply shapes: {batch1.shape} @ {batch2.shape} = {batch_result.shape}")

    # Transpose
    transposed = mat1.T
    print(f"\nTranspose of Matrix 1:\n{transposed}")
    print()


def dot_and_cross_products():
    """Dot and cross products."""
    print("3. Dot and Cross Products")
    print("-" * 50)

    vec1 = torch.tensor([1.0, 2.0, 3.0])
    vec2 = torch.tensor([4.0, 5.0, 6.0])

    print(f"Vector 1: {vec1}")
    print(f"Vector 2: {vec2}")

    # Dot product
    dot_prod = torch.dot(vec1, vec2)
    print(f"\nDot product: {dot_prod}")

    # Cross product (3D vectors)
    cross_prod = torch.cross(vec1, vec2)
    print(f"Cross product: {cross_prod}")

    # Inner product (generalizes dot product)
    inner_prod = torch.inner(vec1, vec2)
    print(f"Inner product: {inner_prod}")

    # Outer product
    outer_prod = torch.outer(vec1, vec2)
    print(f"\nOuter product:\n{outer_prod}")
    print()


def aggregation_operations():
    """Aggregation and reduction operations."""
    print("4. Aggregation Operations")
    print("-" * 50)

    tensor = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])

    print(f"Tensor:\n{tensor}")

    # Sum
    total_sum = torch.sum(tensor)
    sum_rows = torch.sum(tensor, dim=0)  # Sum along rows
    sum_cols = torch.sum(tensor, dim=1)  # Sum along columns
    print(f"\nTotal sum: {total_sum}")
    print(f"Sum along dim 0 (columns): {sum_rows}")
    print(f"Sum along dim 1 (rows): {sum_cols}")

    # Mean
    mean_val = torch.mean(tensor)
    mean_cols = torch.mean(tensor, dim=1)
    print(f"\nMean: {mean_val:.4f}")
    print(f"Mean along dim 1: {mean_cols}")

    # Standard deviation
    std_val = torch.std(tensor)
    print(f"Standard deviation: {std_val:.4f}")

    # Min and Max
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    print(f"\nMin: {min_val}, Max: {max_val}")

    # Argmin and Argmax
    argmin = torch.argmin(tensor)
    argmax = torch.argmax(tensor)
    print(f"Argmin: {argmin}, Argmax: {argmax}")

    # Min/Max along dimension (returns values and indices)
    max_vals, max_indices = torch.max(tensor, dim=1)
    print(f"\nMax values along dim 1: {max_vals}")
    print(f"Max indices along dim 1: {max_indices}")
    print()


def comparison_operations():
    """Comparison and logical operations."""
    print("5. Comparison Operations")
    print("-" * 50)

    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([5, 4, 3, 2, 1])

    print(f"Tensor a: {a}")
    print(f"Tensor b: {b}")

    # Element-wise comparisons
    equal = a == b
    not_equal = a != b
    greater = a > b
    less_equal = a <= b

    print(f"\na == b: {equal}")
    print(f"a > b: {greater}")
    print(f"a <= b: {less_equal}")

    # Logical operations
    mask1 = a > 2
    mask2 = a < 5
    logical_and = mask1 & mask2
    logical_or = mask1 | mask2
    logical_not = ~mask1

    print(f"\na > 2: {mask1}")
    print(f"a < 5: {mask2}")
    print(f"(a > 2) & (a < 5): {logical_and}")

    # All and Any
    all_true = torch.all(a > 0)
    any_true = torch.any(a > 3)
    print(f"\nAll elements > 0: {all_true}")
    print(f"Any element > 3: {any_true}")
    print()


def advanced_math_functions():
    """Advanced mathematical functions."""
    print("6. Advanced Math Functions")
    print("-" * 50)

    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    print(f"Tensor x: {x}")

    # Exponential and logarithm
    exp_x = torch.exp(x)
    log_x = torch.log(x + 1)  # Avoid log(0)
    print(f"\nexp(x): {exp_x}")
    print(f"log(x+1): {log_x}")

    # Trigonometric functions
    angles = torch.tensor([0.0, torch.pi/4, torch.pi/2])
    sin_vals = torch.sin(angles)
    cos_vals = torch.cos(angles)
    print(f"\nAngles: {angles}")
    print(f"sin(angles): {sin_vals}")
    print(f"cos(angles): {cos_vals}")

    # Square root
    nums = torch.tensor([1.0, 4.0, 9.0, 16.0])
    sqrt_vals = torch.sqrt(nums)
    print(f"\nsqrt({nums}): {sqrt_vals}")

    # Absolute value
    vals = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    abs_vals = torch.abs(vals)
    print(f"\nabs({vals}): {abs_vals}")

    # Clamp (clip values)
    clamped = torch.clamp(vals, min=-1.0, max=1.0)
    print(f"clamp({vals}, -1, 1): {clamped}")

    # Round, floor, ceil
    decimals = torch.tensor([1.2, 2.5, 3.7, 4.9])
    rounded = torch.round(decimals)
    floored = torch.floor(decimals)
    ceiled = torch.ceil(decimals)
    print(f"\nDecimals: {decimals}")
    print(f"Round: {rounded}")
    print(f"Floor: {floored}")
    print(f"Ceil: {ceiled}")
    print()


def practical_examples():
    """Practical examples combining operations."""
    print("7. Practical Examples")
    print("-" * 50)

    # Normalize a tensor (zero mean, unit variance)
    data = torch.randn(5, 3)
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    normalized = (data - mean) / std
    print(f"Original data mean: {data.mean(dim=0)}")
    print(f"Normalized data mean: {normalized.mean(dim=0)}")
    print(f"Normalized data std: {normalized.std(dim=0)}")

    # Euclidean distance
    point1 = torch.tensor([1.0, 2.0, 3.0])
    point2 = torch.tensor([4.0, 5.0, 6.0])
    distance = torch.sqrt(torch.sum((point1 - point2) ** 2))
    print(f"\nEuclidean distance: {distance:.4f}")

    # Cosine similarity
    vec1 = torch.tensor([1.0, 2.0, 3.0])
    vec2 = torch.tensor([4.0, 5.0, 6.0])
    cosine_sim = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
    print(f"Cosine similarity: {cosine_sim:.4f}")

    # Softmax (common in neural networks)
    logits = torch.tensor([2.0, 1.0, 0.1])
    softmax = torch.exp(logits) / torch.sum(torch.exp(logits))
    print(f"\nLogits: {logits}")
    print(f"Softmax: {softmax}")
    print(f"Sum: {softmax.sum()}")
    print()


def main():
    print("=" * 60)
    print("MATHEMATICAL OPERATIONS")
    print("=" * 60)
    print()

    check_environment()
    element_wise_operations()
    matrix_operations()
    dot_and_cross_products()
    aggregation_operations()
    comparison_operations()
    advanced_math_functions()
    practical_examples()

    print("=" * 60)


if __name__ == "__main__":
    main()
