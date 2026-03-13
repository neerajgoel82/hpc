"""
Tensor Basics
=============
Creating and manipulating tensors, basic operations.

Topics:
- Tensor creation methods
- Data types and shapes
- Indexing and slicing
- Reshaping and transposing
- Common tensor operations

Run: python 03_tensor_basics.py
"""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tensor_creation():
    """Comprehensive tensor creation methods."""
    print("\n" + "=" * 60)
    print("Tensor Creation Methods")
    print("=" * 60)

    print("\n1. From Python data structures:")
    scalar = torch.tensor(42)
    vector = torch.tensor([1, 2, 3, 4])
    matrix = torch.tensor([[1, 2], [3, 4]])
    tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    print(f"   Scalar (0D): {scalar}, shape: {scalar.shape}")
    print(f"   Vector (1D): {vector}, shape: {vector.shape}")
    print(f"   Matrix (2D): shape: {matrix.shape}\n{matrix}")
    print(f"   3D Tensor: shape: {tensor_3d.shape}")

    print("\n2. Factory functions (specific values):")
    zeros = torch.zeros(2, 3)
    ones = torch.ones(3, 2)
    full = torch.full((2, 2), 7)
    eye = torch.eye(3)  # Identity matrix

    print(f"   Zeros (2x3):\n{zeros}")
    print(f"   Ones (3x2):\n{ones}")
    print(f"   Full of 7s (2x2):\n{full}")
    print(f"   Identity (3x3):\n{eye}")

    print("\n3. Random tensors:")
    rand_uniform = torch.rand(2, 3)      # Uniform [0, 1)
    rand_normal = torch.randn(2, 3)      # Normal(0, 1)
    rand_int = torch.randint(0, 10, (2, 3))  # Random integers

    print(f"   Uniform [0,1): \n{rand_uniform}")
    print(f"   Normal(0,1): \n{rand_normal}")
    print(f"   Random int [0,10): \n{rand_int}")

    print("\n4. Range tensors:")
    arange = torch.arange(0, 10, 2)
    linspace = torch.linspace(0, 1, 5)

    print(f"   Arange (0 to 10, step 2): {arange}")
    print(f"   Linspace (0 to 1, 5 points): {linspace}")

    print("\n5. Like functions (same shape as another tensor):")
    template = torch.tensor([[1, 2], [3, 4]])
    zeros_like = torch.zeros_like(template)
    ones_like = torch.ones_like(template)
    rand_like = torch.rand_like(template.float())

    print(f"   Template: \n{template}")
    print(f"   Zeros like: \n{zeros_like}")
    print(f"   Ones like: \n{ones_like}")


def tensor_properties():
    """Explore tensor properties and attributes."""
    print("\n" + "=" * 60)
    print("Tensor Properties")
    print("=" * 60)

    x = torch.randn(3, 4, 5)

    print(f"\nTensor x:")
    print(f"  Shape: {x.shape} or {x.size()}")
    print(f"  Dimensions: {x.ndim} (also called rank)")
    print(f"  Total elements: {x.numel()}")
    print(f"  Data type: {x.dtype}")
    print(f"  Device: {x.device}")
    print(f"  Requires grad: {x.requires_grad}")

    print("\nData types (dtypes):")
    int_tensor = torch.tensor([1, 2, 3])
    float_tensor = torch.tensor([1.0, 2.0, 3.0])
    bool_tensor = torch.tensor([True, False, True])

    print(f"  Integer: {int_tensor.dtype}")
    print(f"  Float: {float_tensor.dtype}")
    print(f"  Boolean: {bool_tensor.dtype}")

    print("\nExplicit dtype specification:")
    float32 = torch.tensor([1, 2, 3], dtype=torch.float32)
    float64 = torch.tensor([1, 2, 3], dtype=torch.float64)
    int32 = torch.tensor([1.5, 2.5, 3.5], dtype=torch.int32)

    print(f"  Float32: {float32}, dtype: {float32.dtype}")
    print(f"  Float64: {float64}, dtype: {float64.dtype}")
    print(f"  Int32: {int32}, dtype: {int32.dtype}")


def indexing_slicing():
    """Tensor indexing and slicing operations."""
    print("\n" + "=" * 60)
    print("Indexing and Slicing")
    print("=" * 60)

    x = torch.arange(12).reshape(3, 4)
    print(f"\nOriginal tensor (3x4):\n{x}")

    print("\n1. Basic indexing:")
    print(f"   x[0] (first row): {x[0]}")
    print(f"   x[1, 2] (row 1, col 2): {x[1, 2].item()}")
    print(f"   x[-1] (last row): {x[-1]}")

    print("\n2. Slicing:")
    print(f"   x[:2] (first 2 rows):\n{x[:2]}")
    print(f"   x[:, 1:3] (all rows, cols 1-2):\n{x[:, 1:3]}")
    print(f"   x[1:, ::2] (rows 1+, every 2nd col):\n{x[1:, ::2]}")

    print("\n3. Boolean indexing:")
    mask = x > 5
    print(f"   Mask (x > 5):\n{mask}")
    print(f"   x[x > 5]: {x[x > 5]}")

    print("\n4. Fancy indexing:")
    indices = torch.tensor([0, 2])
    print(f"   x[indices] (rows 0 and 2):\n{x[indices]}")


def reshaping_operations():
    """Reshaping and transposing tensors."""
    print("\n" + "=" * 60)
    print("Reshaping Operations")
    print("=" * 60)

    x = torch.arange(12)
    print(f"\nOriginal (12,): {x}")

    print("\n1. Reshape:")
    reshaped = x.reshape(3, 4)
    print(f"   reshape(3, 4):\n{reshaped}")

    reshaped2 = x.reshape(2, 2, 3)
    print(f"   reshape(2, 2, 3) shape: {reshaped2.shape}")

    print("\n2. View (same as reshape but shares memory):")
    viewed = x.view(4, 3)
    print(f"   view(4, 3):\n{viewed}")

    print("\n3. Reshape with -1 (infer dimension):")
    auto_reshape = x.reshape(3, -1)  # -1 means "figure it out"
    print(f"   reshape(3, -1):\n{auto_reshape}")

    print("\n4. Transpose:")
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"   Original (2x3):\n{matrix}")
    print(f"   Transpose (3x2):\n{matrix.T}")

    print("\n5. Permute (multi-dimensional transpose):")
    tensor_3d = torch.randn(2, 3, 4)
    permuted = tensor_3d.permute(2, 0, 1)  # Reorder dimensions
    print(f"   Original shape: {tensor_3d.shape}")
    print(f"   Permuted (2,0,1) shape: {permuted.shape}")

    print("\n6. Squeeze and unsqueeze:")
    x = torch.randn(1, 3, 1, 4)
    print(f"   Original shape: {x.shape}")
    print(f"   squeeze() (remove 1s): {x.squeeze().shape}")
    print(f"   unsqueeze(0) (add dim): {x.squeeze().unsqueeze(0).shape}")

    print("\n7. Flatten:")
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    flat = matrix.flatten()
    print(f"   Original:\n{matrix}")
    print(f"   Flattened: {flat}")


def basic_operations():
    """Basic tensor operations."""
    print("\n" + "=" * 60)
    print("Basic Tensor Operations")
    print("=" * 60)

    print("\n1. Arithmetic operations:")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print(f"   a = {a}")
    print(f"   b = {b}")
    print(f"   a + b = {a + b}")
    print(f"   a - b = {a - b}")
    print(f"   a * b = {a * b}")  # Element-wise
    print(f"   a / b = {a / b}")
    print(f"   a ** 2 = {a ** 2}")

    print("\n2. Matrix operations:")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    print(f"   A @ B (matmul):\n{A @ B}")
    print(f"   torch.matmul(A, B):\n{torch.matmul(A, B)}")
    print(f"   A * B (element-wise):\n{A * B}")

    print("\n3. Reduction operations:")
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"   x:\n{x}")
    print(f"   sum(x): {torch.sum(x).item()}")
    print(f"   sum(x, dim=0): {torch.sum(x, dim=0)}")  # Sum columns
    print(f"   sum(x, dim=1): {torch.sum(x, dim=1)}")  # Sum rows
    print(f"   mean(x): {torch.mean(x).item()}")
    print(f"   max(x): {torch.max(x).item()}")

    print("\n4. Comparison operations:")
    a = torch.tensor([1, 2, 3, 4])
    b = torch.tensor([2, 2, 2, 2])
    print(f"   a = {a}")
    print(f"   b = {b}")
    print(f"   a > b: {a > b}")
    print(f"   a == b: {a == b}")
    print(f"   torch.eq(a, b): {torch.eq(a, b)}")


def concatenation_stacking():
    """Concatenating and stacking tensors."""
    print("\n" + "=" * 60)
    print("Concatenation and Stacking")
    print("=" * 60)

    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])

    print(f"a:\n{a}")
    print(f"b:\n{b}")

    print("\n1. Concatenation (along existing dimension):")
    cat_dim0 = torch.cat([a, b], dim=0)  # Stack vertically
    cat_dim1 = torch.cat([a, b], dim=1)  # Stack horizontally

    print(f"   cat([a, b], dim=0):\n{cat_dim0}")
    print(f"   Shape: {cat_dim0.shape}")
    print(f"   cat([a, b], dim=1):\n{cat_dim1}")
    print(f"   Shape: {cat_dim1.shape}")

    print("\n2. Stack (creates new dimension):")
    stacked = torch.stack([a, b], dim=0)
    print(f"   stack([a, b], dim=0):")
    print(f"   Shape: {stacked.shape}")
    print(f"   {stacked}")


def in_place_operations():
    """In-place vs out-of-place operations."""
    print("\n" + "=" * 60)
    print("In-place vs Out-of-place Operations")
    print("=" * 60)

    print("\n1. Out-of-place (creates new tensor):")
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"   Original x: {x}")
    y = x.add(5)
    print(f"   y = x.add(5): {y}")
    print(f"   x unchanged: {x}")

    print("\n2. In-place (modifies existing tensor):")
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"   Original x: {x}")
    x.add_(5)  # Note the underscore!
    print(f"   After x.add_(5): {x}")
    print("   (x was modified!)")

    print("\n3. Warning about in-place operations:")
    print("   - Faster (no new memory allocation)")
    print("   - Can cause issues with autograd (gradients)")
    print("   - Use cautiously, especially during training")


def main():
    """Main function to run all demonstrations."""
    print("=" * 60)
    print("Tensor Basics")
    print("=" * 60)
    print(f"\nRunning on device: {device}")

    # Run all demonstrations
    tensor_creation()
    tensor_properties()
    indexing_slicing()
    reshaping_operations()
    basic_operations()
    concatenation_stacking()
    in_place_operations()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Tensors are multi-dimensional arrays (scalars, vectors, matrices, etc.)")
    print("2. Many ways to create tensors: from data, factories, random, ranges")
    print("3. Indexing/slicing similar to NumPy")
    print("4. Reshape, view, transpose for changing tensor shape")
    print("5. Rich set of operations: arithmetic, matrix, reductions")
    print("6. In-place operations end with '_' (e.g., add_)")
    print("\nNext: 04_numpy_vs_pytorch.py to compare with NumPy")
    print("=" * 60)


if __name__ == "__main__":
    main()
