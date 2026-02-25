"""
Memory Management
Efficient memory usage and management in PyTorch

This module demonstrates:
- In-place operations
- Cloning and copying tensors
- Memory sharing and storage
- Contiguous memory layout
- GPU memory management
- Best practices for memory efficiency

Run: python 06_memory_management.py
"""

import torch
import sys


def check_environment():
    """Display PyTorch environment information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()


def inplace_operations():
    """In-place operations and their implications."""
    print("1. In-place Operations")
    print("-" * 50)

    # Regular operation (creates new tensor)
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"Original x: {x}")
    print(f"Original x id: {id(x)}")

    y = x + 1
    print(f"y = x + 1: {y}")
    print(f"y id: {id(y)}")
    print(f"x unchanged: {x}")

    # In-place operation (modifies existing tensor)
    x.add_(1)
    print(f"\nAfter x.add_(1): {x}")
    print(f"x id unchanged: {id(x)}")

    # Common in-place operations
    a = torch.tensor([1.0, 2.0, 3.0])
    print(f"\nOriginal a: {a}")

    a.mul_(2)     # a *= 2
    print(f"After mul_(2): {a}")

    a.div_(2)     # a /= 2
    print(f"After div_(2): {a}")

    a.fill_(5)    # Fill with value
    print(f"After fill_(5): {a}")

    # In-place with slicing
    tensor = torch.ones(3, 3)
    tensor[0, :] += 10
    print(f"\nAfter in-place slice operation:\n{tensor}")

    # Warning about gradients
    print("\nWarning: In-place operations can cause issues with autograd!")
    print("Avoid in-place ops on tensors that require gradients.")
    print()


def cloning_and_copying():
    """Different ways to clone and copy tensors."""
    print("2. Cloning and Copying")
    print("-" * 50)

    original = torch.tensor([[1, 2], [3, 4]])
    print(f"Original:\n{original}")
    print(f"Original data_ptr: {original.data_ptr()}")

    # Clone (creates independent copy)
    cloned = original.clone()
    print(f"\nCloned:\n{cloned}")
    print(f"Cloned data_ptr: {cloned.data_ptr()}")
    print(f"Different memory: {cloned.data_ptr() != original.data_ptr()}")

    cloned[0, 0] = 999
    print(f"\nAfter modifying clone:")
    print(f"Original:\n{original}")
    print(f"Cloned:\n{cloned}")

    # Detach (creates view with no gradient)
    tensor_with_grad = torch.randn(2, 2, requires_grad=True)
    detached = tensor_with_grad.detach()
    print(f"\nOriginal requires_grad: {tensor_with_grad.requires_grad}")
    print(f"Detached requires_grad: {detached.requires_grad}")
    print(f"Shares memory: {detached.data_ptr() == tensor_with_grad.data_ptr()}")

    # Deep copy vs shallow copy
    import copy

    shallow = copy.copy(original)
    deep = copy.deepcopy(original)

    print(f"\nShallow copy data_ptr: {shallow.data_ptr()}")
    print(f"Deep copy data_ptr: {deep.data_ptr()}")
    print()


def memory_sharing():
    """Understanding memory sharing between tensors."""
    print("3. Memory Sharing")
    print("-" * 50)

    # Original tensor
    original = torch.arange(12).reshape(3, 4)
    print(f"Original:\n{original}")
    print(f"Data pointer: {original.data_ptr()}")

    # View shares memory
    viewed = original.view(4, 3)
    print(f"\nView data pointer: {viewed.data_ptr()}")
    print(f"Shares memory: {viewed.data_ptr() == original.data_ptr()}")

    # Slice shares memory
    sliced = original[1:3]
    print(f"Slice data pointer: {sliced.data_ptr()}")
    print(f"Shares memory (offset): {sliced.data_ptr() != original.data_ptr()}")

    # Modifying shared memory
    viewed[0, 0] = 999
    print(f"\nAfter modifying view:")
    print(f"Original:\n{original}")

    # Storage object
    print(f"\nOriginal storage size: {original.storage().size()}")
    print(f"View storage size: {viewed.storage().size()}")
    print(f"Same storage: {original.storage().data_ptr() == viewed.storage().data_ptr()}")

    # Check if tensors share storage
    print(f"\nTensors share storage: {original.storage().data_ptr() == viewed.storage().data_ptr()}")
    print()


def contiguous_memory_management():
    """Managing contiguous memory layout."""
    print("4. Contiguous Memory Management")
    print("-" * 50)

    # Contiguous tensor
    tensor = torch.arange(12).reshape(3, 4)
    print(f"Original is contiguous: {tensor.is_contiguous()}")
    print(f"Stride: {tensor.stride()}")

    # Transpose makes non-contiguous
    transposed = tensor.t()
    print(f"\nTransposed is contiguous: {transposed.is_contiguous()}")
    print(f"Stride: {transposed.stride()}")

    # Convert to contiguous
    made_contiguous = transposed.contiguous()
    print(f"\nMade contiguous: {made_contiguous.is_contiguous()}")
    print(f"Stride: {made_contiguous.stride()}")

    # Performance implications
    import time

    size = 1000
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # Contiguous operation
    start = time.time()
    for _ in range(100):
        _ = a + b
    contiguous_time = time.time() - start

    # Non-contiguous operation
    a_t = a.t()
    b_t = b.t()
    start = time.time()
    for _ in range(100):
        _ = a_t + b_t
    non_contiguous_time = time.time() - start

    print(f"\nContiguous operation time: {contiguous_time:.4f}s")
    print(f"Non-contiguous operation time: {non_contiguous_time:.4f}s")
    print(f"Slowdown: {non_contiguous_time / contiguous_time:.2f}x")
    print()


def gpu_memory_management():
    """Managing GPU memory."""
    print("5. GPU Memory Management")
    print("-" * 50)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory management")
        print()
        return

    # Check GPU memory
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Current memory usage
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    print(f"\nAllocated memory: {allocated:.2f} MB")
    print(f"Reserved memory: {reserved:.2f} MB")

    # Allocate GPU tensors
    tensor_gpu = torch.randn(1000, 1000, device='cuda')
    allocated_after = torch.cuda.memory_allocated() / 1e6
    print(f"\nAfter allocation (1000x1000):")
    print(f"Allocated memory: {allocated_after:.2f} MB")
    print(f"Increase: {allocated_after - allocated:.2f} MB")

    # Delete tensor
    del tensor_gpu
    torch.cuda.empty_cache()
    allocated_cleared = torch.cuda.memory_allocated() / 1e6
    print(f"\nAfter clearing:")
    print(f"Allocated memory: {allocated_cleared:.2f} MB")

    # Memory context manager
    print("\nUsing caching allocator:")
    with torch.cuda.device(0):
        x = torch.randn(500, 500, device='cuda')
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    # Clear cache
    torch.cuda.empty_cache()
    print()


def memory_profiling():
    """Profile memory usage."""
    print("6. Memory Profiling")
    print("-" * 50)

    # Get tensor size in bytes
    tensor = torch.randn(100, 100)
    element_size = tensor.element_size()
    num_elements = tensor.nelement()
    total_bytes = element_size * num_elements

    print(f"Tensor shape: {tensor.shape}")
    print(f"Element size: {element_size} bytes")
    print(f"Number of elements: {num_elements}")
    print(f"Total memory: {total_bytes} bytes ({total_bytes / 1e6:.2f} MB)")

    # Compare different dtypes
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64, torch.float16]
    print(f"\nMemory usage for (1000, 1000) tensor:")
    for dtype in dtypes:
        t = torch.zeros(1000, 1000, dtype=dtype)
        size = t.element_size() * t.nelement() / 1e6
        print(f"{str(dtype):20s}: {size:6.2f} MB")

    # Check Python object overhead
    small_tensor = torch.tensor([1.0])
    tensor_size = small_tensor.element_size() * small_tensor.nelement()
    python_size = sys.getsizeof(small_tensor)
    print(f"\nSmall tensor data size: {tensor_size} bytes")
    print(f"Python object size: {python_size} bytes")
    print(f"Overhead: {python_size - tensor_size} bytes")
    print()


def best_practices():
    """Memory management best practices."""
    print("7. Best Practices")
    print("-" * 50)

    print("Memory Management Best Practices:")
    print()
    print("1. In-place operations:")
    print("   - Use for memory efficiency: tensor.add_(1)")
    print("   - Avoid with autograd: can break gradient computation")
    print()
    print("2. Cloning:")
    print("   - Use .clone() for independent copies")
    print("   - Use .detach() to remove from computation graph")
    print()
    print("3. GPU memory:")
    print("   - Move tensors to CPU when not needed: tensor.cpu()")
    print("   - Clear cache: torch.cuda.empty_cache()")
    print("   - Delete large tensors: del tensor")
    print()
    print("4. Contiguous memory:")
    print("   - Call .contiguous() before certain operations")
    print("   - Impacts performance of element-wise operations")
    print()
    print("5. Data types:")
    print("   - Use float16 for memory savings (with caution)")
    print("   - Be aware of precision trade-offs")
    print()
    print("6. Avoid memory leaks:")
    print("   - Detach tensors from computation graph")
    print("   - Clear references to large tensors")
    print("   - Use context managers for GPU operations")
    print()

    # Example: Efficient batch processing
    print("Example: Efficient batch processing")
    print("-" * 50)

    def process_batch_inefficient(data):
        """Inefficient: Creates many intermediate tensors."""
        result = data + 1
        result = result * 2
        result = result - 3
        return result

    def process_batch_efficient(data):
        """Efficient: Uses in-place operations."""
        data = data.clone()  # Clone to avoid modifying input
        data.add_(1)
        data.mul_(2)
        data.sub_(3)
        return data

    data = torch.randn(1000, 1000)

    import time
    start = time.time()
    for _ in range(100):
        _ = process_batch_inefficient(data)
    inefficient_time = time.time() - start

    start = time.time()
    for _ in range(100):
        _ = process_batch_efficient(data)
    efficient_time = time.time() - start

    print(f"Inefficient time: {inefficient_time:.4f}s")
    print(f"Efficient time: {efficient_time:.4f}s")
    print(f"Speedup: {inefficient_time / efficient_time:.2f}x")
    print()


def main():
    print("=" * 60)
    print("MEMORY MANAGEMENT")
    print("=" * 60)
    print()

    check_environment()
    inplace_operations()
    cloning_and_copying()
    memory_sharing()
    contiguous_memory_management()
    gpu_memory_management()
    memory_profiling()
    best_practices()

    print("=" * 60)


if __name__ == "__main__":
    main()
