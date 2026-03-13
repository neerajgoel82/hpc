"""
Tensor Operations Exercises
Practice problems for tensor manipulation and operations

This module contains 8 exercises covering:
- Tensor creation and manipulation
- Indexing and slicing
- Mathematical operations
- Broadcasting
- Reshaping and views
- Memory management

Run: python exercises.py
"""

import torch
import numpy as np


def check_environment():
    """Display PyTorch environment information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()


def exercise_1():
    """
    Exercise 1: Create a custom initialization pattern

    Create a 5x5 tensor where:
    - Diagonal elements are 1
    - Elements above diagonal are 2
    - Elements below diagonal are 0
    """
    print("Exercise 1: Custom Tensor Initialization")
    print("-" * 50)

    # TODO: Implement this
    # Hint: Use torch.triu, torch.tril, and torch.eye

    # Solution
    tensor = torch.zeros(5, 5)
    tensor = tensor + torch.triu(torch.ones(5, 5) * 2, diagonal=1)
    tensor = tensor + torch.eye(5)

    print("Result:")
    print(tensor)
    print()

    # Verify
    expected = torch.tensor([
        [1., 2., 2., 2., 2.],
        [0., 1., 2., 2., 2.],
        [0., 0., 1., 2., 2.],
        [0., 0., 0., 1., 2.],
        [0., 0., 0., 0., 1.]
    ])
    assert torch.allclose(tensor, expected), "Exercise 1 failed!"
    print("Exercise 1 passed!")
    print()


def exercise_2():
    """
    Exercise 2: Advanced indexing and filtering

    Given a tensor of random values:
    1. Find all elements greater than 0.5
    2. Replace elements less than -0.5 with 0
    3. Extract elements in the range [-0.5, 0.5]
    4. Get indices of top 5 largest elements
    """
    print("Exercise 2: Advanced Indexing")
    print("-" * 50)

    # Create random tensor
    torch.manual_seed(42)
    tensor = torch.randn(20)
    print(f"Original tensor:\n{tensor}")

    # TODO: Implement the tasks

    # Solution
    # 1. Elements > 0.5
    greater_than_half = tensor[tensor > 0.5]
    print(f"\n1. Elements > 0.5 ({len(greater_than_half)} elements):")
    print(greater_than_half)

    # 2. Replace < -0.5 with 0
    tensor_modified = tensor.clone()
    tensor_modified[tensor_modified < -0.5] = 0
    print(f"\n2. After replacing < -0.5 with 0:")
    print(tensor_modified)

    # 3. Elements in range [-0.5, 0.5]
    in_range = tensor[(tensor >= -0.5) & (tensor <= 0.5)]
    print(f"\n3. Elements in [-0.5, 0.5] ({len(in_range)} elements):")
    print(in_range)

    # 4. Indices of top 5 largest
    top5_indices = torch.topk(tensor, 5).indices
    top5_values = torch.topk(tensor, 5).values
    print(f"\n4. Top 5 largest elements:")
    print(f"Indices: {top5_indices}")
    print(f"Values: {top5_values}")
    print()

    print("Exercise 2 completed!")
    print()


def exercise_3():
    """
    Exercise 3: Matrix operations

    Implement the following:
    1. Create two random 3x3 matrices A and B
    2. Compute A @ B (matrix multiplication)
    3. Compute A * B (element-wise)
    4. Compute trace of A (sum of diagonal)
    5. Compute determinant of A
    6. Find eigenvalues of A
    """
    print("Exercise 3: Matrix Operations")
    print("-" * 50)

    # TODO: Implement the tasks

    # Solution
    torch.manual_seed(42)
    A = torch.randn(3, 3)
    B = torch.randn(3, 3)

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    # 1. Matrix multiplication
    matmul = A @ B
    print(f"\n1. A @ B:")
    print(matmul)

    # 2. Element-wise multiplication
    elemwise = A * B
    print(f"\n2. A * B (element-wise):")
    print(elemwise)

    # 3. Trace
    trace = torch.trace(A)
    print(f"\n3. Trace of A: {trace:.4f}")

    # 4. Determinant
    det = torch.det(A)
    print(f"4. Determinant of A: {det:.4f}")

    # 5. Eigenvalues
    eigenvalues = torch.linalg.eigvals(A)
    print(f"5. Eigenvalues of A:")
    print(eigenvalues)
    print()

    print("Exercise 3 completed!")
    print()


def exercise_4():
    """
    Exercise 4: Broadcasting challenge

    Given:
    - A batch of 32 images (32, 3, 64, 64)
    - Per-channel mean: [0.485, 0.456, 0.406]
    - Per-channel std: [0.229, 0.224, 0.225]

    Normalize the batch using broadcasting
    """
    print("Exercise 4: Broadcasting for Normalization")
    print("-" * 50)

    # TODO: Implement normalization

    # Solution
    torch.manual_seed(42)
    batch = torch.randn(32, 3, 64, 64)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    normalized = (batch - mean) / std

    print(f"Batch shape: {batch.shape}")
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Normalized shape: {normalized.shape}")

    # Verify normalization (approximately)
    print(f"\nOriginal batch - Channel 0 mean: {batch[:, 0, :, :].mean():.4f}")
    print(f"Normalized batch - Channel 0 mean: {normalized[:, 0, :, :].mean():.4f}")

    print("\nExercise 4 completed!")
    print()


def exercise_5():
    """
    Exercise 5: Tensor reshaping challenge

    Given a tensor of shape (2, 3, 4):
    1. Flatten it to 1D
    2. Reshape to (6, 4)
    3. Transpose to (4, 6)
    4. Add batch dimension: (1, 4, 6)
    5. Permute to (4, 1, 6)
    6. Squeeze to (4, 6)
    """
    print("Exercise 5: Reshaping Pipeline")
    print("-" * 50)

    # TODO: Implement the reshaping pipeline

    # Solution
    original = torch.arange(24).reshape(2, 3, 4)
    print(f"Original shape: {original.shape}")

    # 1. Flatten
    step1 = original.flatten()
    print(f"1. Flattened: {step1.shape}")

    # 2. Reshape
    step2 = step1.reshape(6, 4)
    print(f"2. Reshaped: {step2.shape}")

    # 3. Transpose
    step3 = step2.t()
    print(f"3. Transposed: {step3.shape}")

    # 4. Add batch dimension
    step4 = step3.unsqueeze(0)
    print(f"4. Added batch dim: {step4.shape}")

    # 5. Permute
    step5 = step4.permute(1, 0, 2)
    print(f"5. Permuted: {step5.shape}")

    # 6. Squeeze
    step6 = step5.squeeze(1)
    print(f"6. Squeezed: {step6.shape}")

    print(f"\nFinal shape: {step6.shape}")
    assert step6.shape == (4, 6), "Exercise 5 failed!"
    print("Exercise 5 passed!")
    print()


def exercise_6():
    """
    Exercise 6: Memory-efficient operations

    Implement a function that:
    1. Takes a large tensor (1000, 1000)
    2. Normalizes it (subtract mean, divide by std) IN-PLACE
    3. Clips values to [-3, 3] IN-PLACE
    4. Returns the modified tensor

    Compare memory usage with non-in-place version
    """
    print("Exercise 6: Memory-Efficient Processing")
    print("-" * 50)

    def process_inplace(tensor):
        """Process tensor using in-place operations."""
        # TODO: Implement in-place processing
        mean = tensor.mean()
        std = tensor.std()
        tensor.sub_(mean).div_(std)
        tensor.clamp_(-3, 3)
        return tensor

    def process_regular(tensor):
        """Process tensor using regular operations."""
        mean = tensor.mean()
        std = tensor.std()
        tensor = (tensor - mean) / std
        tensor = torch.clamp(tensor, -3, 3)
        return tensor

    # Solution
    import time

    # Test in-place
    torch.manual_seed(42)
    tensor1 = torch.randn(1000, 1000)
    start = time.time()
    result1 = process_inplace(tensor1)
    inplace_time = time.time() - start

    # Test regular
    torch.manual_seed(42)
    tensor2 = torch.randn(1000, 1000)
    start = time.time()
    result2 = process_regular(tensor2)
    regular_time = time.time() - start

    print(f"In-place time: {inplace_time:.4f}s")
    print(f"Regular time: {regular_time:.4f}s")
    print(f"Speedup: {regular_time / inplace_time:.2f}x")

    # Verify results are similar
    print(f"\nResults match: {torch.allclose(result1, result2, rtol=1e-4)}")
    print("Exercise 6 completed!")
    print()


def exercise_7():
    """
    Exercise 7: Batch matrix operations

    Given:
    - A batch of 16 matrices, each 4x3
    - A batch of 16 matrices, each 3x5

    Perform batch matrix multiplication and then:
    1. Sum over the batch dimension
    2. Compute mean for each element across batch
    3. Find the maximum value in each matrix
    """
    print("Exercise 7: Batch Matrix Operations")
    print("-" * 50)

    # TODO: Implement batch operations

    # Solution
    torch.manual_seed(42)
    batch_a = torch.randn(16, 4, 3)
    batch_b = torch.randn(16, 3, 5)

    print(f"Batch A shape: {batch_a.shape}")
    print(f"Batch B shape: {batch_b.shape}")

    # Batch matrix multiplication
    result = torch.bmm(batch_a, batch_b)
    print(f"\nBatch multiplication result: {result.shape}")

    # 1. Sum over batch
    sum_batch = result.sum(dim=0)
    print(f"1. Sum over batch: {sum_batch.shape}")

    # 2. Mean across batch
    mean_batch = result.mean(dim=0)
    print(f"2. Mean across batch: {mean_batch.shape}")

    # 3. Max value in each matrix
    max_per_matrix = result.flatten(start_dim=1).max(dim=1).values
    print(f"3. Max per matrix: {max_per_matrix.shape}")
    print(f"   Values: {max_per_matrix}")

    print("\nExercise 7 completed!")
    print()


def exercise_8():
    """
    Exercise 8: Real-world scenario - Image preprocessing

    You have a batch of grayscale images (32, 1, 28, 28)
    1. Flatten each image to a vector
    2. Compute pairwise Euclidean distances between first 5 images
    3. Find the two most similar images
    4. Extract 7x7 patches from center of each image
    """
    print("Exercise 8: Image Preprocessing Pipeline")
    print("-" * 50)

    # TODO: Implement image preprocessing

    # Solution
    torch.manual_seed(42)
    images = torch.randn(32, 1, 28, 28)
    print(f"Images shape: {images.shape}")

    # 1. Flatten each image
    flattened = images.flatten(start_dim=1)
    print(f"\n1. Flattened shape: {flattened.shape}")

    # 2. Pairwise distances for first 5
    first_five = flattened[:5]
    print(f"2. Computing distances for first 5 images")

    # Compute pairwise distances
    # dist[i,j] = ||x_i - x_j||
    distances = torch.cdist(first_five, first_five, p=2)
    print(f"   Distance matrix shape: {distances.shape}")
    print(f"   Distance matrix:\n{distances}")

    # 3. Find most similar (excluding diagonal)
    # Set diagonal to inf to exclude self-similarity
    distances_copy = distances.clone()
    distances_copy.fill_diagonal_(float('inf'))
    min_dist = distances_copy.min()
    min_idx = (distances_copy == min_dist).nonzero()[0]

    print(f"\n3. Most similar images: {min_idx[0].item()} and {min_idx[1].item()}")
    print(f"   Distance: {min_dist:.4f}")

    # 4. Extract 7x7 center patches
    center_start = (28 - 7) // 2  # Start at 10
    patches = images[:, :, center_start:center_start+7, center_start:center_start+7]
    print(f"\n4. Center patches shape: {patches.shape}")

    print("\nExercise 8 completed!")
    print()


def bonus_exercise():
    """
    Bonus Exercise: Implement a custom attention mechanism

    Given queries (Q), keys (K), and values (V):
    1. Compute attention scores: Q @ K^T
    2. Scale by sqrt(d_k)
    3. Apply softmax
    4. Multiply by V

    This is the core of transformer attention!
    """
    print("Bonus Exercise: Attention Mechanism")
    print("-" * 50)

    # TODO: Implement attention

    # Solution
    def scaled_dot_product_attention(Q, K, V):
        """Compute scaled dot-product attention."""
        d_k = Q.shape[-1]

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale
        scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply to values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    # Test with sample data
    torch.manual_seed(42)
    batch_size, seq_len, d_model = 2, 4, 8

    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")

    output, weights = scaled_dot_product_attention(Q, K, V)

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nAttention weights for first sample:")
    print(weights[0])
    print(f"Row sums (should be 1): {weights[0].sum(dim=1)}")

    print("\nBonus exercise completed!")
    print("You've implemented the core of transformer attention!")
    print()


def main():
    print("=" * 60)
    print("TENSOR OPERATIONS EXERCISES")
    print("=" * 60)
    print()

    check_environment()

    exercises = [
        exercise_1,
        exercise_2,
        exercise_3,
        exercise_4,
        exercise_5,
        exercise_6,
        exercise_7,
        exercise_8,
        bonus_exercise,
    ]

    for exercise in exercises:
        try:
            exercise()
        except Exception as e:
            print(f"Error in {exercise.__name__}: {e}")
            print()

    print("=" * 60)
    print("All exercises completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
