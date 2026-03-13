"""
NumPy vs PyTorch
=================
Comparison and conversion between NumPy and PyTorch.

Topics:
- NumPy and PyTorch similarities
- Key differences
- Converting between NumPy and PyTorch
- When to use which
- Performance comparison

Run: python 04_numpy_vs_pytorch.py
"""

import torch
import numpy as np
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def similarities():
    """Show similarities between NumPy and PyTorch."""
    print("\n" + "=" * 60)
    print("NumPy vs PyTorch: Similarities")
    print("=" * 60)

    print("\n1. Array/Tensor creation:")
    np_array = np.array([1, 2, 3, 4])
    torch_tensor = torch.tensor([1, 2, 3, 4])
    print(f"   NumPy:   {np_array}")
    print(f"   PyTorch: {torch_tensor}")

    print("\n2. Zeros and ones:")
    np_zeros = np.zeros((2, 3))
    torch_zeros = torch.zeros(2, 3)
    print(f"   NumPy zeros:\n{np_zeros}")
    print(f"   PyTorch zeros:\n{torch_zeros}")

    print("\n3. Random values:")
    np_rand = np.random.randn(2, 2)
    torch_rand = torch.randn(2, 2)
    print(f"   NumPy random:\n{np_rand}")
    print(f"   PyTorch random:\n{torch_rand}")

    print("\n4. Indexing and slicing:")
    np_arr = np.arange(12).reshape(3, 4)
    torch_arr = torch.arange(12).reshape(3, 4)
    print(f"   NumPy [0, :]: {np_arr[0, :]}")
    print(f"   PyTorch [0, :]: {torch_arr[0, :]}")

    print("\n5. Arithmetic operations:")
    a_np = np.array([1, 2, 3])
    b_np = np.array([4, 5, 6])
    a_torch = torch.tensor([1, 2, 3])
    b_torch = torch.tensor([4, 5, 6])
    print(f"   NumPy: {a_np} + {b_np} = {a_np + b_np}")
    print(f"   PyTorch: {a_torch} + {b_torch} = {a_torch + b_torch}")

    print("\n6. Matrix operations:")
    A_np = np.array([[1, 2], [3, 4]])
    B_np = np.array([[5, 6], [7, 8]])
    A_torch = torch.tensor([[1, 2], [3, 4]])
    B_torch = torch.tensor([[5, 6], [7, 8]])
    print(f"   NumPy matmul:\n{np.matmul(A_np, B_np)}")
    print(f"   PyTorch matmul:\n{torch.matmul(A_torch, B_torch)}")


def key_differences():
    """Highlight key differences between NumPy and PyTorch."""
    print("\n" + "=" * 60)
    print("NumPy vs PyTorch: Key Differences")
    print("=" * 60)

    print("\n1. GPU Support:")
    print("   NumPy: CPU only")
    print("   PyTorch: CPU and GPU (CUDA)")
    x_torch = torch.randn(3, 3)
    print(f"   PyTorch tensor on {x_torch.device}")
    if torch.cuda.is_available():
        x_gpu = x_torch.to('cuda')
        print(f"   Can move to GPU: {x_gpu.device}")
    else:
        print("   (GPU not available, but feature exists)")

    print("\n2. Automatic Differentiation:")
    print("   NumPy: No built-in autodiff")
    print("   PyTorch: Automatic gradient computation")
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 3
    y.backward()
    print(f"   x = {x.item()}, y = x^3 = {y.item()}")
    print(f"   dy/dx = {x.grad.item()} (automatic!)")

    print("\n3. Default data types:")
    np_arr = np.array([1.0, 2.0])
    torch_arr = torch.tensor([1.0, 2.0])
    print(f"   NumPy default: {np_arr.dtype}")
    print(f"   PyTorch default: {torch_arr.dtype}")

    print("\n4. API differences:")
    print("   NumPy: reshape() always returns a view or copy")
    print("   PyTorch: view() requires contiguous memory, reshape() more flexible")
    x = torch.randn(4, 3)
    print(f"   x.view(3, 4) - requires contiguous")
    print(f"   x.reshape(3, 4) - always works")

    print("\n5. Broadcasting:")
    print("   Both support broadcasting, but subtle differences exist")
    a = torch.tensor([[1], [2], [3]])
    b = torch.tensor([10, 20, 30])
    print(f"   PyTorch: (3,1) + (3,) = {(a + b).shape}")


def conversion_operations():
    """Converting between NumPy and PyTorch."""
    print("\n" + "=" * 60)
    print("Converting Between NumPy and PyTorch")
    print("=" * 60)

    print("\n1. NumPy to PyTorch:")
    np_array = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"   NumPy array:\n{np_array}")

    # Method 1: torch.from_numpy() - shares memory!
    torch_from_np = torch.from_numpy(np_array)
    print(f"   torch.from_numpy():\n{torch_from_np}")

    # Method 2: torch.tensor() - copies data
    torch_copy = torch.tensor(np_array)
    print(f"   torch.tensor():\n{torch_copy}")

    print("\n   Important: from_numpy() shares memory!")
    np_array[0, 0] = 999
    print(f"   After modifying np_array[0,0] = 999:")
    print(f"   from_numpy tensor: {torch_from_np[0, 0].item()} (changed!)")
    print(f"   tensor copy: {torch_copy[0, 0].item()} (unchanged)")

    print("\n2. PyTorch to NumPy:")
    torch_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"   PyTorch tensor:\n{torch_tensor}")

    # Convert to NumPy - shares memory!
    np_from_torch = torch_tensor.numpy()
    print(f"   .numpy():\n{np_from_torch}")

    print("\n   Also shares memory:")
    torch_tensor[0, 0] = 999
    print(f"   After modifying tensor[0,0] = 999:")
    print(f"   NumPy array: {np_from_torch[0, 0]} (changed!)")

    print("\n3. GPU tensors to NumPy:")
    if torch.cuda.is_available():
        gpu_tensor = torch.randn(2, 2).cuda()
        print(f"   GPU tensor device: {gpu_tensor.device}")
        np_from_gpu = gpu_tensor.cpu().numpy()
        print(f"   Must move to CPU first: .cpu().numpy()")
    else:
        print("   (GPU not available)")
        print("   GPU tensors need .cpu().numpy() to convert")

    print("\n4. Copying vs sharing memory:")
    np_arr = np.array([1, 2, 3])
    # Sharing memory
    torch_shared = torch.from_numpy(np_arr)
    # Copying data
    torch_copied = torch.tensor(np_arr)

    np_arr[0] = 999
    print(f"   Original np: {np_arr}")
    print(f"   Shared tensor: {torch_shared}")
    print(f"   Copied tensor: {torch_copied}")


def api_comparison():
    """Compare common operations in NumPy and PyTorch."""
    print("\n" + "=" * 60)
    print("API Comparison: Common Operations")
    print("=" * 60)

    print("\n| Operation        | NumPy                | PyTorch              |")
    print("|-----------------|----------------------|----------------------|")
    print("| Creation        | np.array([1,2,3])    | torch.tensor([1,2,3])|")
    print("| Zeros           | np.zeros((3,4))      | torch.zeros(3,4)     |")
    print("| Random          | np.random.randn(3,4) | torch.randn(3,4)     |")
    print("| Reshape         | arr.reshape(3,4)     | tensor.reshape(3,4)  |")
    print("| Transpose       | arr.T                | tensor.T             |")
    print("| Sum             | np.sum(arr)          | torch.sum(tensor)    |")
    print("| Mean            | np.mean(arr)         | torch.mean(tensor)   |")
    print("| Max             | np.max(arr)          | torch.max(tensor)    |")
    print("| Matmul          | np.matmul(a,b)       | torch.matmul(a,b)    |")
    print("| Concatenate     | np.concatenate()     | torch.cat()          |")
    print("| Stack           | np.stack()           | torch.stack()        |")

    print("\nCode examples:")
    print("\n1. Sum along axis:")
    np_arr = np.array([[1, 2, 3], [4, 5, 6]])
    torch_arr = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"   NumPy: np.sum(arr, axis=0) = {np.sum(np_arr, axis=0)}")
    print(f"   PyTorch: torch.sum(arr, dim=0) = {torch.sum(torch_arr, dim=0)}")

    print("\n2. Reshape:")
    np_flat = np.arange(12)
    torch_flat = torch.arange(12)
    print(f"   NumPy: arr.reshape(3, 4) shape = {np_flat.reshape(3, 4).shape}")
    print(f"   PyTorch: tensor.reshape(3, 4) shape = {torch_flat.reshape(3, 4).shape}")


def performance_comparison():
    """Compare performance of NumPy vs PyTorch (CPU) vs PyTorch (GPU)."""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    size = 1000
    iterations = 100

    # NumPy
    np_a = np.random.randn(size, size)
    np_b = np.random.randn(size, size)

    start = time.time()
    for _ in range(iterations):
        np_c = np.matmul(np_a, np_b)
    np_time = time.time() - start

    # PyTorch CPU
    torch_a = torch.randn(size, size)
    torch_b = torch.randn(size, size)

    start = time.time()
    for _ in range(iterations):
        torch_c = torch.matmul(torch_a, torch_b)
    torch_cpu_time = time.time() - start

    print(f"\nMatrix multiplication ({size}x{size}, {iterations} iterations):")
    print(f"  NumPy (CPU):    {np_time:.4f} seconds")
    print(f"  PyTorch (CPU):  {torch_cpu_time:.4f} seconds")

    # PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch_a_gpu = torch_a.cuda()
        torch_b_gpu = torch_b.cuda()

        # Warm up
        for _ in range(10):
            _ = torch.matmul(torch_a_gpu, torch_b_gpu)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(iterations):
            torch_c_gpu = torch.matmul(torch_a_gpu, torch_b_gpu)
        torch.cuda.synchronize()
        torch_gpu_time = time.time() - start

        print(f"  PyTorch (GPU):  {torch_gpu_time:.4f} seconds")
        print(f"\n  Speedup (GPU vs CPU): {torch_cpu_time/torch_gpu_time:.2f}x")
    else:
        print(f"  PyTorch (GPU):  N/A (no GPU available)")


def when_to_use_which():
    """Guide on when to use NumPy vs PyTorch."""
    print("\n" + "=" * 60)
    print("When to Use NumPy vs PyTorch")
    print("=" * 60)

    print("\nUse NumPy when:")
    print("  ✓ Only need CPU computation")
    print("  ✓ Working with traditional numerical computing")
    print("  ✓ Integrating with SciPy, scikit-learn, pandas")
    print("  ✓ Don't need gradients/backpropagation")
    print("  ✓ Smaller arrays/simple operations")

    print("\nUse PyTorch when:")
    print("  ✓ Building deep learning models")
    print("  ✓ Need automatic differentiation")
    print("  ✓ Want GPU acceleration")
    print("  ✓ Working with neural networks")
    print("  ✓ Large-scale tensor computations")
    print("  ✓ Need dynamic computation graphs")

    print("\nBest practice:")
    print("  • Start with NumPy for data preprocessing")
    print("  • Convert to PyTorch for model training")
    print("  • Can easily convert back and forth")
    print("  • Use PyTorch end-to-end for deep learning pipelines")


def main():
    """Main function to run all demonstrations."""
    print("=" * 60)
    print("NumPy vs PyTorch")
    print("=" * 60)
    print(f"\nNumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Running on device: {device}")

    # Run all demonstrations
    similarities()
    key_differences()
    conversion_operations()
    api_comparison()
    performance_comparison()
    when_to_use_which()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. NumPy and PyTorch have similar APIs for basic operations")
    print("2. PyTorch adds GPU support and automatic differentiation")
    print("3. Easy conversion: torch.from_numpy() and .numpy()")
    print("4. from_numpy() and .numpy() share memory (be careful!)")
    print("5. Use NumPy for traditional computing, PyTorch for deep learning")
    print("6. GPU acceleration gives massive speedup for large operations")
    print("\nNext: 05_gpu_acceleration.py for deeper dive into GPU usage")
    print("=" * 60)


if __name__ == "__main__":
    main()
