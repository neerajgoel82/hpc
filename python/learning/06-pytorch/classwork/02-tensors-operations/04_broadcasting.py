"""
Broadcasting in PyTorch
Broadcasting rules and examples for efficient tensor operations

This module demonstrates:
- Broadcasting rules and semantics
- Common broadcasting patterns
- Broadcasting with different dimensions
- Pitfalls and best practices
- Performance considerations

Run: python 04_broadcasting.py
"""

import torch


def check_environment():
    """Display PyTorch environment information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()


def broadcasting_basics():
    """Basic broadcasting examples."""
    print("1. Broadcasting Basics")
    print("-" * 50)

    # Scalar with tensor
    tensor = torch.tensor([1, 2, 3, 4])
    scalar = 10
    result = tensor + scalar
    print(f"Tensor: {tensor}")
    print(f"Scalar: {scalar}")
    print(f"Tensor + Scalar: {result}")

    # 1D with 2D
    vec = torch.tensor([1, 2, 3])
    mat = torch.tensor([[10, 20, 30],
                        [40, 50, 60]])
    result = mat + vec
    print(f"\nVector shape: {vec.shape}")
    print(f"Matrix shape: {mat.shape}")
    print(f"Matrix:\n{mat}")
    print(f"Vector: {vec}")
    print(f"Matrix + Vector:\n{result}")
    print()


def broadcasting_rules():
    """Demonstrate broadcasting rules."""
    print("2. Broadcasting Rules")
    print("-" * 50)

    print("Broadcasting Rules:")
    print("1. If tensors have different number of dimensions,")
    print("   prepend 1s to the shape of smaller tensor")
    print("2. Tensors are compatible if, for each dimension:")
    print("   - They are equal, OR")
    print("   - One of them is 1")
    print()

    # Rule 1: Different number of dimensions
    a = torch.ones(3, 4)      # Shape: (3, 4)
    b = torch.ones(4)          # Shape: (4,) -> becomes (1, 4)
    result = a + b
    print(f"a shape: {a.shape}")
    print(f"b shape: {b.shape} -> broadcasted to (1, 4)")
    print(f"Result shape: {result.shape}")

    # Rule 2: Compatible dimensions
    c = torch.ones(3, 1)       # Shape: (3, 1)
    d = torch.ones(1, 4)       # Shape: (1, 4)
    result = c + d             # Result: (3, 4)
    print(f"\nc shape: {c.shape}")
    print(f"d shape: {d.shape}")
    print(f"Result shape: {result.shape}")

    # Incompatible shapes (would raise error)
    e = torch.ones(3, 4)
    f = torch.ones(3)
    # This works: f is broadcasted to (1, 3) then to (3, 3)? No!
    # Actually f shape (3,) becomes (1, 3) which doesn't match (3, 4)
    try:
        result = e + f
        print(f"\ne + f worked! Shape: {result.shape}")
    except RuntimeError as err:
        print(f"\ne shape {e.shape} + f shape {f.shape} failed (as expected)")
    print()


def common_patterns():
    """Common broadcasting patterns."""
    print("3. Common Broadcasting Patterns")
    print("-" * 50)

    # Pattern 1: Add bias to each row
    data = torch.ones(3, 4)
    row_bias = torch.tensor([1, 2, 3, 4])
    result = data + row_bias
    print("Pattern 1: Add bias to each row")
    print(f"Data shape: {data.shape}, Bias shape: {row_bias.shape}")
    print(f"Result:\n{result}")

    # Pattern 2: Add bias to each column
    col_bias = torch.tensor([[1], [2], [3]])
    result = data + col_bias
    print(f"\nPattern 2: Add bias to each column")
    print(f"Data shape: {data.shape}, Bias shape: {col_bias.shape}")
    print(f"Result:\n{result}")

    # Pattern 3: Normalize each feature
    features = torch.randn(100, 5)  # 100 samples, 5 features
    mean = features.mean(dim=0, keepdim=True)  # Shape: (1, 5)
    std = features.std(dim=0, keepdim=True)    # Shape: (1, 5)
    normalized = (features - mean) / std
    print(f"\nPattern 3: Feature normalization")
    print(f"Features shape: {features.shape}")
    print(f"Mean shape: {mean.shape}")
    print(f"Normalized shape: {normalized.shape}")

    # Pattern 4: Batch operations
    batch = torch.randn(32, 3, 28, 28)  # Batch of 32 RGB images
    channel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    normalized_batch = batch - channel_mean
    print(f"\nPattern 4: Batch normalization")
    print(f"Batch shape: {batch.shape}")
    print(f"Channel mean shape: {channel_mean.shape}")
    print(f"Result shape: {normalized_batch.shape}")
    print()


def multidimensional_broadcasting():
    """Broadcasting with multiple dimensions."""
    print("4. Multidimensional Broadcasting")
    print("-" * 50)

    # 3D tensor broadcasting
    tensor_3d = torch.ones(2, 3, 4)
    vec_1d = torch.ones(4)
    result = tensor_3d + vec_1d
    print(f"3D tensor shape: {tensor_3d.shape}")
    print(f"1D vector shape: {vec_1d.shape}")
    print(f"Result shape: {result.shape}")

    # Different broadcasting directions
    tensor_a = torch.ones(5, 1, 3)
    tensor_b = torch.ones(1, 4, 3)
    result = tensor_a + tensor_b
    print(f"\nTensor A shape: {tensor_a.shape}")
    print(f"Tensor B shape: {tensor_b.shape}")
    print(f"Result shape: {result.shape}")

    # Complex broadcasting
    a = torch.ones(8, 1, 6, 1)
    b = torch.ones(7, 1, 5)
    result = a + b
    print(f"\nComplex broadcasting:")
    print(f"A shape: {a.shape} (8, 1, 6, 1)")
    print(f"B shape: {b.shape} (7, 1, 5) -> (1, 7, 1, 5)")
    print(f"Result shape: {result.shape} (8, 7, 6, 5)")
    print()


def explicit_broadcasting():
    """Explicit broadcasting using expand and repeat."""
    print("5. Explicit Broadcasting")
    print("-" * 50)

    tensor = torch.tensor([[1], [2], [3]])
    print(f"Original tensor shape: {tensor.shape}")
    print(f"Original:\n{tensor}")

    # expand (returns a view, efficient)
    expanded = tensor.expand(3, 4)
    print(f"\nExpanded to (3, 4):\n{expanded}")
    print(f"Is view: {expanded.data_ptr() == tensor.data_ptr()}")

    # expand_as (expand to match another tensor)
    target = torch.zeros(3, 5)
    expanded_as = tensor.expand_as(target)
    print(f"\nExpanded as {target.shape}:\n{expanded_as}")

    # repeat (creates a copy, less efficient)
    repeated = tensor.repeat(2, 3)
    print(f"\nRepeated (2, 3) times:\n{repeated}")
    print(f"Shape: {repeated.shape}")

    # broadcast_to (explicit broadcasting)
    broadcasted = torch.broadcast_to(tensor, (3, 4))
    print(f"\nBroadcast to (3, 4):\n{broadcasted}")
    print()


def broadcasting_pitfalls():
    """Common pitfalls and how to avoid them."""
    print("6. Broadcasting Pitfalls")
    print("-" * 50)

    # Pitfall 1: Unintended broadcasting
    weights = torch.randn(5)
    data = torch.randn(3, 4)
    # This will fail because shapes don't align
    try:
        result = data @ weights
        print(f"Success: {result.shape}")
    except RuntimeError as e:
        print("Pitfall 1: Shape mismatch in matrix multiplication")
        print(f"Weights shape: {weights.shape}, Data shape: {data.shape}")
        print("Need to ensure dimensions align correctly")

    # Pitfall 2: Memory usage
    print("\nPitfall 2: Memory considerations")
    small = torch.ones(1, 1000)
    large = torch.ones(10000, 1000)
    # Broadcasting creates implicit copies
    result = small + large
    print(f"Small shape: {small.shape}, Large shape: {large.shape}")
    print(f"Result shape: {result.shape}")
    print("Broadcasting can lead to large memory usage!")

    # Pitfall 3: Loss of keepdim
    tensor = torch.randn(3, 4)
    mean_wrong = tensor.mean(dim=1)  # Shape: (3,)
    mean_right = tensor.mean(dim=1, keepdim=True)  # Shape: (3, 1)

    print(f"\nPitfall 3: keepdim importance")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Mean without keepdim: {mean_wrong.shape}")
    print(f"Mean with keepdim: {mean_right.shape}")
    print("Use keepdim=True to preserve broadcasting compatibility")
    print()


def performance_considerations():
    """Performance aspects of broadcasting."""
    print("7. Performance Considerations")
    print("-" * 50)

    import time

    # Efficient: Broadcasting
    n = 1000
    matrix = torch.randn(n, n)
    vector = torch.randn(n)

    start = time.time()
    for _ in range(100):
        result = matrix + vector  # Broadcasting
    broadcast_time = time.time() - start

    # Inefficient: Manual expansion
    start = time.time()
    for _ in range(100):
        expanded = vector.unsqueeze(0).expand(n, n)
        result = matrix + expanded
    expand_time = time.time() - start

    print(f"Matrix shape: {matrix.shape}, Vector shape: {vector.shape}")
    print(f"Broadcasting time: {broadcast_time:.4f}s")
    print(f"Manual expansion time: {expand_time:.4f}s")
    print(f"Speedup: {expand_time / broadcast_time:.2f}x")

    print("\nBest practices:")
    print("1. Let PyTorch handle broadcasting automatically")
    print("2. Use keepdim=True in reductions for broadcasting")
    print("3. Understand memory implications of broadcasting")
    print("4. Use expand() instead of repeat() when possible")
    print()


def main():
    print("=" * 60)
    print("BROADCASTING IN PYTORCH")
    print("=" * 60)
    print()

    check_environment()
    broadcasting_basics()
    broadcasting_rules()
    common_patterns()
    multidimensional_broadcasting()
    explicit_broadcasting()
    broadcasting_pitfalls()
    performance_considerations()

    print("=" * 60)


if __name__ == "__main__":
    main()
