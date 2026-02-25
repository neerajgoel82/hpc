"""
Tensor Creation
Different ways to create tensors in PyTorch

This module demonstrates various tensor creation methods including:
- zeros, ones, empty tensors
- random tensors (randn, rand, randint)
- range tensors (arange, linspace)
- conversion from NumPy arrays
- tensor initialization with specific dtypes and devices

Run: python 01_tensor_creation.py
"""

import torch
import numpy as np


def check_environment():
    """Display PyTorch environment information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print()


def zeros_and_ones():
    """Create tensors filled with zeros and ones."""
    print("1. Zeros and Ones Tensors")
    print("-" * 50)

    # Create zero tensors
    zeros_1d = torch.zeros(5)
    zeros_2d = torch.zeros(3, 4)
    zeros_3d = torch.zeros(2, 3, 4)

    print(f"1D zeros: {zeros_1d}")
    print(f"2D zeros shape: {zeros_2d.shape}")
    print(f"3D zeros shape: {zeros_3d.shape}")

    # Create one tensors
    ones_2d = torch.ones(3, 3)
    print(f"\n2D ones:\n{ones_2d}")

    # Create with specific dtype
    ones_int = torch.ones(2, 3, dtype=torch.int32)
    ones_float64 = torch.ones(2, 3, dtype=torch.float64)
    print(f"\nInt32 ones dtype: {ones_int.dtype}")
    print(f"Float64 ones dtype: {ones_float64.dtype}")
    print()


def random_tensors():
    """Create tensors with random values."""
    print("2. Random Tensors")
    print("-" * 50)

    # Random from normal distribution (mean=0, std=1)
    randn_tensor = torch.randn(3, 3)
    print(f"Random normal (3x3):\n{randn_tensor}")

    # Random uniform [0, 1)
    rand_tensor = torch.rand(2, 4)
    print(f"\nRandom uniform (2x4):\n{rand_tensor}")

    # Random integers
    randint_tensor = torch.randint(0, 10, (3, 3))
    print(f"\nRandom integers [0, 10) (3x3):\n{randint_tensor}")

    # Set seed for reproducibility
    torch.manual_seed(42)
    seeded_tensor = torch.randn(2, 2)
    print(f"\nSeeded random (seed=42):\n{seeded_tensor}")
    print()


def range_tensors():
    """Create tensors with sequential values."""
    print("3. Range Tensors (arange, linspace)")
    print("-" * 50)

    # arange: similar to Python's range
    arange_tensor = torch.arange(10)
    print(f"arange(10): {arange_tensor}")

    arange_step = torch.arange(0, 10, 2)
    print(f"arange(0, 10, 2): {arange_step}")

    arange_float = torch.arange(0.0, 5.0, 0.5)
    print(f"arange(0.0, 5.0, 0.5): {arange_float}")

    # linspace: evenly spaced values
    linspace_tensor = torch.linspace(0, 10, steps=5)
    print(f"\nlinspace(0, 10, steps=5): {linspace_tensor}")

    linspace_tensor2 = torch.linspace(-1, 1, steps=11)
    print(f"linspace(-1, 1, steps=11): {linspace_tensor2}")
    print()


def from_numpy_conversion():
    """Convert between NumPy arrays and PyTorch tensors."""
    print("4. NumPy Conversion")
    print("-" * 50)

    # NumPy to PyTorch
    np_array = np.array([[1, 2, 3], [4, 5, 6]])
    tensor_from_np = torch.from_numpy(np_array)
    print(f"NumPy array:\n{np_array}")
    print(f"Tensor from NumPy:\n{tensor_from_np}")
    print(f"Tensor dtype: {tensor_from_np.dtype}")

    # Memory is shared between NumPy and PyTorch
    np_array[0, 0] = 100
    print(f"\nAfter modifying NumPy array:")
    print(f"NumPy array:\n{np_array}")
    print(f"Tensor (shared memory):\n{tensor_from_np}")

    # PyTorch to NumPy
    tensor = torch.tensor([[7, 8], [9, 10]])
    np_from_tensor = tensor.numpy()
    print(f"\nTensor:\n{tensor}")
    print(f"NumPy from tensor:\n{np_from_tensor}")
    print()


def device_handling():
    """Create tensors on different devices (CPU/GPU)."""
    print("5. Device Handling")
    print("-" * 50)

    # Check available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create tensor on specific device
    tensor_cpu = torch.ones(3, 3)
    print(f"CPU tensor device: {tensor_cpu.device}")

    if torch.cuda.is_available():
        # Create directly on GPU
        tensor_gpu = torch.ones(3, 3, device="cuda")
        print(f"GPU tensor device: {tensor_gpu.device}")

        # Move tensor to GPU
        tensor_moved = tensor_cpu.to(device)
        print(f"Moved tensor device: {tensor_moved.device}")

        # Move back to CPU
        tensor_back = tensor_moved.cpu()
        print(f"Back to CPU device: {tensor_back.device}")
    else:
        print("CUDA not available, using CPU only")
    print()


def other_creation_methods():
    """Other useful tensor creation methods."""
    print("6. Other Creation Methods")
    print("-" * 50)

    # Create empty tensor (uninitialized)
    empty_tensor = torch.empty(2, 3)
    print(f"Empty tensor (uninitialized):\n{empty_tensor}")

    # Create tensor from Python list
    list_tensor = torch.tensor([1, 2, 3, 4, 5])
    print(f"\nFrom list: {list_tensor}")

    # Create tensor like another tensor
    base_tensor = torch.randn(2, 3)
    zeros_like = torch.zeros_like(base_tensor)
    ones_like = torch.ones_like(base_tensor)
    rand_like = torch.rand_like(base_tensor)

    print(f"\nBase tensor:\n{base_tensor}")
    print(f"Zeros like (same shape):\n{zeros_like}")
    print(f"Ones like (same shape):\n{ones_like}")

    # Eye (identity matrix)
    eye_tensor = torch.eye(4)
    print(f"\nIdentity matrix (4x4):\n{eye_tensor}")

    # Full (filled with specific value)
    full_tensor = torch.full((3, 3), fill_value=7.5)
    print(f"\nFull tensor (filled with 7.5):\n{full_tensor}")
    print()


def main():
    print("=" * 60)
    print("TENSOR CREATION IN PYTORCH")
    print("=" * 60)
    print()

    check_environment()
    zeros_and_ones()
    random_tensors()
    range_tensors()
    from_numpy_conversion()
    device_handling()
    other_creation_methods()

    print("=" * 60)


if __name__ == "__main__":
    main()
