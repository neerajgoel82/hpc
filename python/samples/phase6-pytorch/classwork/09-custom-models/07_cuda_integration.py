"""
CUDA Integration
Custom CUDA operations with PyTorch (educational level)

This module demonstrates:
- Using PyTorch with CUDA
- Understanding PyTorch-CUDA connection
- Custom CUDA kernels basics (conceptual)
- Performance optimization with CUDA
- CuPy integration for custom operations

Note: Full CUDA kernel implementation requires C++/CUDA compilation
Run: python 07_cuda_integration.py
"""

import torch
import torch.nn as nn
import time
import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not installed. Install with: pip install cupy-cuda11x")


def demonstrate_pytorch_cuda_basics():
    """Demonstrate basic PyTorch CUDA operations."""
    print("1. PyTorch CUDA Basics")
    print("-" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU.\n")
        return

    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")

    # Create tensors on GPU
    x_cpu = torch.randn(1000, 1000)
    x_gpu = x_cpu.cuda()  # or x_cpu.to('cuda')

    print(f"\nCPU tensor device: {x_cpu.device}")
    print(f"GPU tensor device: {x_gpu.device}")

    # Operations on GPU
    y_gpu = torch.matmul(x_gpu, x_gpu)
    print(f"Result device: {y_gpu.device}")

    # Memory info
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"\nGPU memory allocated: {allocated:.2f} MB")
    print(f"GPU memory reserved: {reserved:.2f} MB")
    print()


def demonstrate_performance_comparison():
    """Compare CPU vs GPU performance."""
    print("2. CPU vs GPU Performance")
    print("-" * 60)

    size = 5000

    if torch.cuda.is_available():
        # CPU benchmark
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)

        start = time.time()
        for _ in range(10):
            z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start

        # GPU benchmark
        x_gpu = x_cpu.cuda()
        y_gpu = y_cpu.cuda()

        # Warm-up
        _ = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(10):
            z_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"Matrix size: {size}x{size}")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("CUDA not available for benchmarking")

    print()


def demonstrate_cupy_integration():
    """Demonstrate CuPy integration for custom CUDA operations."""
    print("3. CuPy Integration")
    print("-" * 60)

    if not CUPY_AVAILABLE or not torch.cuda.is_available():
        print("CuPy or CUDA not available.")
        print("CuPy allows writing custom CUDA kernels in Python syntax\n")
        return

    # PyTorch tensor
    x_torch = torch.randn(100, 100, device="cuda")

    # Convert to CuPy (zero-copy)
    x_cupy = cp.from_dlpack(x_torch)

    # CuPy operations
    y_cupy = cp.exp(x_cupy)

    # Convert back to PyTorch (zero-copy)
    y_torch = torch.from_dlpack(y_cupy)

    print(f"PyTorch tensor device: {x_torch.device}")
    print(f"CuPy array device: {x_cupy.device}")
    print(f"Result tensor device: {y_torch.device}")
    print("\nCuPy enables custom CUDA operations in Python")
    print("Zero-copy conversion between PyTorch and CuPy")
    print()


def demonstrate_custom_cuda_kernel_concept():
    """Demonstrate custom CUDA kernel concepts."""
    print("4. Custom CUDA Kernel Concepts")
    print("-" * 60)

    print("Custom CUDA kernels in PyTorch require:")
    print()
    print("1. C++/CUDA code:")
    print("   - Write CUDA kernel in .cu file")
    print("   - Handle thread indexing and synchronization")
    print("   - Optimize memory access patterns")
    print()
    print("2. PyTorch C++ extension:")
    print("   - Create wrapper using torch::Tensor")
    print("   - Handle PyTorch integration")
    print("   - Build with setuptools")
    print()
    print("3. Python binding:")
    print("   - Load compiled extension")
    print("   - Use as regular PyTorch operation")
    print()
    print("Example CUDA kernel structure:")
    print("```cuda")
    print("__global__ void my_kernel(float* input, float* output, int size) {")
    print("    int idx = blockIdx.x * blockDim.x + threadIdx.x;")
    print("    if (idx < size) {")
    print("        output[idx] = input[idx] * 2.0f;")
    print("    }")
    print("}")
    print("```")
    print()


def demonstrate_torch_cuda_ops():
    """Demonstrate PyTorch's built-in CUDA-optimized operations."""
    print("5. PyTorch CUDA-Optimized Operations")
    print("-" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available.\n")
        return

    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")

    print("PyTorch provides CUDA-optimized implementations:")
    print()
    print("Matrix Operations:")
    print("  - torch.matmul() - Matrix multiplication")
    print("  - torch.mm() - Matrix multiply")
    print("  - torch.bmm() - Batch matrix multiply")

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        z = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"\nBenchmark: 100 matrix multiplications")
    print(f"Time: {elapsed:.4f}s")
    print(f"Throughput: {100/elapsed:.1f} ops/sec")

    print("\nConvolution Operations:")
    print("  - Uses cuDNN library")
    print("  - Highly optimized for CNNs")
    print("  - Multiple algorithms (FFT, Winograd, etc.)")

    print("\nActivation Functions:")
    print("  - ReLU, Sigmoid, Tanh (fused kernels)")
    print("  - Batch Normalization (optimized)")
    print()


def demonstrate_memory_management():
    """Demonstrate CUDA memory management."""
    print("6. CUDA Memory Management")
    print("-" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available.\n")
        return

    print("Memory management functions:")
    print()

    # Allocate tensor
    x = torch.randn(1000, 1000, device="cuda")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # Create more tensors
    tensors = [torch.randn(500, 500, device="cuda") for _ in range(10)]
    print(f"After allocations: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # Clear cache
    del tensors
    torch.cuda.empty_cache()
    print(f"After cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    print("\nMemory management tips:")
    print("  - torch.cuda.empty_cache() - Free unused memory")
    print("  - torch.cuda.memory_allocated() - Current usage")
    print("  - torch.cuda.memory_reserved() - Total reserved")
    print("  - torch.cuda.reset_peak_memory_stats() - Reset counters")
    print("  - Use context manager for automatic cleanup")
    print()


def demonstrate_cuda_streams():
    """Demonstrate CUDA streams for concurrent operations."""
    print("7. CUDA Streams (Advanced)")
    print("-" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available.\n")
        return

    print("CUDA Streams enable concurrent GPU operations:")
    print()

    # Default stream
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    print("Default stream: Sequential operations")

    # Multiple streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    print("\nWith streams:")
    with torch.cuda.stream(stream1):
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.matmul(a, a)

    with torch.cuda.stream(stream2):
        c = torch.randn(1000, 1000, device="cuda")
        d = torch.matmul(c, c)

    # Synchronize all streams
    torch.cuda.synchronize()
    print("Operations can run concurrently on different streams")

    print("\nUse cases:")
    print("  - Overlap computation and data transfer")
    print("  - Concurrent independent operations")
    print("  - Pipeline processing")
    print()


def demonstrate_best_practices():
    """Demonstrate best practices for PyTorch-CUDA integration."""
    print("8. Best Practices")
    print("-" * 60)

    print("PyTorch CUDA Best Practices:")
    print()
    print("1. Memory Transfer:")
    print("   - Minimize CPU-GPU transfers")
    print("   - Batch operations together")
    print("   - Use pinned memory for faster transfer")
    print()
    print("2. Kernel Launch:")
    print("   - Avoid small operations on GPU")
    print("   - Batch operations when possible")
    print("   - Use in-place operations carefully")
    print()
    print("3. cuDNN Optimization:")
    print("   - torch.backends.cudnn.benchmark = True")
    print("   - For fixed input sizes")
    print("   - Finds fastest algorithm")
    print()
    print("4. Mixed Precision:")
    print("   - Use torch.cuda.amp")
    print("   - Reduces memory usage")
    print("   - Faster on Tensor Cores")
    print()
    print("5. Profiling:")
    print("   - Use torch.profiler")
    print("   - Identify bottlenecks")
    print("   - Optimize hot spots")
    print()


def main():
    print("=" * 70)
    print("PYTORCH-CUDA INTEGRATION")
    print("=" * 70)
    print()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print()

    demonstrate_pytorch_cuda_basics()
    demonstrate_performance_comparison()
    demonstrate_cupy_integration()
    demonstrate_custom_cuda_kernel_concept()
    demonstrate_torch_cuda_ops()
    demonstrate_memory_management()
    demonstrate_cuda_streams()
    demonstrate_best_practices()

    print("=" * 70)
    print("\nFor advanced CUDA integration:")
    print("  - See PyTorch C++ extension tutorials")
    print("  - Check CUDA samples in /cuda/samples/")
    print("  - Learn CUDA programming fundamentals")
    print("=" * 70)


if __name__ == "__main__":
    main()
