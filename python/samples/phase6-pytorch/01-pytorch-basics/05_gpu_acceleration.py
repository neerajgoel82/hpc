"""
GPU Acceleration with PyTorch
==============================
Using CUDA, device management, CPU vs GPU performance.

Topics:
- CUDA availability and GPU detection
- Moving tensors between CPU and GPU
- Device management best practices
- Performance comparison CPU vs GPU
- Memory management on GPU

Run: python 05_gpu_acceleration.py
"""

import torch
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_gpu_availability():
    """Check GPU availability and get GPU information."""
    print("\n" + "=" * 60)
    print("GPU Availability and Information")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")

    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n--- GPU {i} ---")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"Compute Capability: {props.major}.{props.minor}")
            print(f"Multi-processors: {props.multi_processor_count}")

        print(f"\nCurrent GPU: {torch.cuda.current_device()}")
        print(f"Current GPU Name: {torch.cuda.get_device_name()}")
    else:
        print("\nNo GPU available. Running on CPU.")
        print("To use GPU:")
        print("  1. Install NVIDIA GPU driver")
        print("  2. Install CUDA toolkit")
        print("  3. Install PyTorch with CUDA support")
        print("  4. OR use Google Colab for free GPU access")


def device_management():
    """Demonstrate device management and moving tensors."""
    print("\n" + "=" * 60)
    print("Device Management")
    print("=" * 60)

    print("\n1. Creating tensors on different devices:")

    # Default device (CPU)
    x_cpu = torch.randn(3, 3)
    print(f"   Default tensor: {x_cpu.device}")

    # Explicit CPU
    x_cpu_explicit = torch.randn(3, 3, device='cpu')
    print(f"   Explicit CPU: {x_cpu_explicit.device}")

    # GPU (if available)
    if torch.cuda.is_available():
        x_gpu = torch.randn(3, 3, device='cuda')
        print(f"   Explicit GPU: {x_gpu.device}")

        # Specific GPU device
        x_gpu0 = torch.randn(3, 3, device='cuda:0')
        print(f"   GPU 0: {x_gpu0.device}")

    print("\n2. Moving tensors between devices:")
    x = torch.randn(3, 3)
    print(f"   Original: {x.device}")

    if torch.cuda.is_available():
        # Method 1: .to() method
        x_gpu = x.to('cuda')
        print(f"   After .to('cuda'): {x_gpu.device}")

        # Method 2: .cuda() method
        x_gpu2 = x.cuda()
        print(f"   After .cuda(): {x_gpu2.device}")

        # Move back to CPU
        x_cpu = x_gpu.cpu()
        print(f"   After .cpu(): {x_cpu.device}")

        # Method 3: Using device object
        device = torch.device('cuda')
        x_gpu3 = x.to(device)
        print(f"   After .to(device): {x_gpu3.device}")
    else:
        print("   (GPU not available, but methods are: .to('cuda'), .cuda())")

    print("\n3. Device-agnostic code:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Selected device: {device}")

    x = torch.randn(3, 3, device=device)
    y = torch.randn(3, 3, device=device)
    z = x + y
    print(f"   All tensors on: {z.device}")
    print("   This code works on both CPU and GPU!")


def tensor_operations_gpu():
    """Demonstrate tensor operations on GPU."""
    print("\n" + "=" * 60)
    print("Tensor Operations on GPU")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nGPU not available. Skipping GPU-specific operations.")
        print("(All operations shown would work on GPU if available)")
        return

    print("\n1. Basic operations on GPU:")
    a = torch.randn(3, 3, device='cuda')
    b = torch.randn(3, 3, device='cuda')

    print(f"   a device: {a.device}")
    print(f"   b device: {b.device}")

    c = a + b
    print(f"   c = a + b device: {c.device}")
    print("   Result stays on GPU automatically!")

    print("\n2. Matrix multiplication on GPU:")
    A = torch.randn(1000, 1000, device='cuda')
    B = torch.randn(1000, 1000, device='cuda')
    C = torch.matmul(A, B)
    print(f"   Result device: {C.device}")

    print("\n3. Moving data between devices:")
    x_cpu = torch.randn(3, 3)
    x_gpu = x_cpu.cuda()
    y_gpu = x_gpu * 2
    y_cpu = y_gpu.cpu()
    print(f"   Started on: {x_cpu.device}")
    print(f"   Computed on: {y_gpu.device}")
    print(f"   Ended on: {y_cpu.device}")


def performance_comparison():
    """Compare CPU vs GPU performance."""
    print("\n" + "=" * 60)
    print("CPU vs GPU Performance Comparison")
    print("=" * 60)

    sizes = [100, 500, 1000, 2000]

    print("\nMatrix Multiplication Benchmarks:")
    print(f"{'Size':<10} {'CPU Time':<15} {'GPU Time':<15} {'Speedup':<10}")
    print("-" * 50)

    for size in sizes:
        # CPU
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)

        start = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start

        # GPU
        if torch.cuda.is_available():
            a_gpu = a_cpu.cuda()
            b_gpu = b_cpu.cuda()

            # Warm up
            _ = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()

            start = time.time()
            c_gpu = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start

            speedup = cpu_time / gpu_time
            print(f"{size:<10} {cpu_time:<15.6f} {gpu_time:<15.6f} {speedup:<10.2f}x")
        else:
            print(f"{size:<10} {cpu_time:<15.6f} {'N/A':<15} {'N/A':<10}")

    if torch.cuda.is_available():
        print("\nObservations:")
        print("  • GPU advantage increases with larger matrices")
        print("  • Small operations may be faster on CPU (overhead)")
        print("  • Always include torch.cuda.synchronize() for accurate timing")
    else:
        print("\n(GPU not available - install CUDA PyTorch to see speedup)")


def memory_management():
    """Demonstrate GPU memory management."""
    print("\n" + "=" * 60)
    print("GPU Memory Management")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nGPU not available. Memory management concepts:")
        print("  • Check memory: torch.cuda.memory_allocated()")
        print("  • Clear cache: torch.cuda.empty_cache()")
        print("  • Delete tensors: del x or x = None")
        return

    print("\n1. Checking memory usage:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"   Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    print("\n2. Allocating memory:")
    x = torch.randn(1000, 1000, device='cuda')
    print(f"   Created 1000x1000 tensor")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    print("\n3. Freeing memory:")
    del x
    print(f"   After del x:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    torch.cuda.empty_cache()
    print(f"   After empty_cache():")
    print(f"   Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    print("\n4. Memory management tips:")
    print("   • Delete unused tensors: del x")
    print("   • Clear cache: torch.cuda.empty_cache()")
    print("   • Use smaller batch sizes if out of memory")
    print("   • Move tensors to CPU when not needed")
    print("   • Use mixed precision training (AMP)")


def common_pitfalls():
    """Demonstrate common pitfalls with GPU usage."""
    print("\n" + "=" * 60)
    print("Common Pitfalls and Solutions")
    print("=" * 60)

    print("\n1. Mixing CPU and GPU tensors:")
    print("   Problem: Cannot operate on tensors from different devices")
    print("   Example:")
    print("     x_cpu = torch.randn(3, 3)")
    print("     x_gpu = torch.randn(3, 3, device='cuda')")
    print("     # z = x_cpu + x_gpu  # ERROR!")
    print("\n   Solution: Move all tensors to same device")
    print("     z = x_cpu.to(device) + x_gpu")

    print("\n2. Forgetting to move model to GPU:")
    print("   Problem: Model on CPU, data on GPU (or vice versa)")
    print("   Solution:")
    print("     model = MyModel()")
    print("     model = model.to(device)")

    print("\n3. Not using torch.cuda.synchronize():")
    print("   Problem: Inaccurate timing measurements")
    print("   Solution:")
    print("     torch.cuda.synchronize()  # Wait for GPU operations")
    print("     # Then measure time")

    print("\n4. Out of memory errors:")
    print("   Solutions:")
    print("     • Reduce batch size")
    print("     • Delete unused tensors")
    print("     • Use gradient checkpointing")
    print("     • Use mixed precision training")

    if torch.cuda.is_available():
        print("\n5. Example: Proper device handling:")
        device = torch.device('cuda')

        # Create model and data on same device
        x = torch.randn(10, 5, device=device)
        y = torch.randn(10, 1, device=device)

        # Simple linear layer
        w = torch.randn(5, 1, device=device, requires_grad=True)
        b = torch.randn(1, device=device, requires_grad=True)

        # Forward pass (all on GPU)
        pred = x @ w + b
        loss = torch.mean((pred - y) ** 2)

        print(f"     All operations on: {pred.device}")
        print(f"     Loss: {loss.item():.4f}")


def best_practices():
    """Best practices for GPU usage in PyTorch."""
    print("\n" + "=" * 60)
    print("Best Practices for GPU Usage")
    print("=" * 60)

    print("\n1. Device-agnostic code:")
    print("   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
    print("   model.to(device)")
    print("   data.to(device)")

    print("\n2. Keep data on GPU:")
    print("   • Minimize CPU ↔ GPU transfers (slow!)")
    print("   • Process entire batches on GPU")
    print("   • Only move final results to CPU")

    print("\n3. Batch processing:")
    print("   • Larger batches better utilize GPU")
    print("   • Balance memory usage vs speed")

    print("\n4. Pin memory for DataLoader:")
    print("   DataLoader(dataset, pin_memory=True)")
    print("   • Faster CPU → GPU transfer")

    print("\n5. Mixed precision training:")
    print("   • Use torch.cuda.amp for automatic mixed precision")
    print("   • Saves memory and increases speed")

    print("\n6. Monitor GPU usage:")
    print("   • nvidia-smi (command line)")
    print("   • torch.cuda.memory_allocated()")
    print("   • TensorBoard profiling")

    print("\n7. Multiple GPUs:")
    print("   • torch.nn.DataParallel for simple multi-GPU")
    print("   • torch.nn.parallel.DistributedDataParallel for production")


def main():
    """Main function to run all demonstrations."""
    print("=" * 60)
    print("GPU Acceleration with PyTorch")
    print("=" * 60)
    print(f"\nRunning on device: {device}")

    # Run all demonstrations
    check_gpu_availability()
    device_management()
    tensor_operations_gpu()
    performance_comparison()
    memory_management()
    common_pitfalls()
    best_practices()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Check GPU with torch.cuda.is_available()")
    print("2. Move tensors with .to(device), .cuda(), or .cpu()")
    print("3. Write device-agnostic code using torch.device")
    print("4. GPU gives massive speedup for large operations")
    print("5. All tensors in operation must be on same device")
    print("6. Manage GPU memory carefully (del, empty_cache)")
    print("7. Minimize CPU ↔ GPU transfers")
    print("8. Use torch.cuda.synchronize() for accurate timing")
    print("\nNext: exercises.py to practice PyTorch basics!")
    print("=" * 60)


if __name__ == "__main__":
    main()
