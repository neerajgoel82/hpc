"""
Tensor Indexing and Slicing
Advanced tensor indexing, slicing, and masking operations

This module demonstrates:
- Basic indexing and slicing (similar to NumPy)
- Advanced indexing techniques
- Boolean masking
- Fancy indexing
- Index selection and filtering

Run: python 02_indexing_slicing.py
"""

import torch


def check_environment():
    """Display PyTorch environment information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()


def basic_indexing():
    """Basic tensor indexing operations."""
    print("1. Basic Indexing")
    print("-" * 50)

    # Create a sample tensor
    tensor = torch.arange(20).reshape(4, 5)
    print(f"Original tensor (4x5):\n{tensor}")

    # Single element access
    element = tensor[1, 2]
    print(f"\nElement at [1, 2]: {element}")
    print(f"Type: {type(element)}, Value: {element.item()}")

    # Row access
    row = tensor[1]
    print(f"\nRow 1: {row}")

    # Column access
    column = tensor[:, 2]
    print(f"Column 2: {column}")

    # Multiple rows
    rows = tensor[1:3]
    print(f"\nRows 1-2:\n{rows}")

    # Last row
    last_row = tensor[-1]
    print(f"\nLast row: {last_row}")
    print()


def slicing_operations():
    """Advanced slicing operations."""
    print("2. Slicing Operations")
    print("-" * 50)

    tensor = torch.arange(24).reshape(4, 6)
    print(f"Original tensor (4x6):\n{tensor}")

    # Slice rows and columns
    slice1 = tensor[1:3, 2:5]
    print(f"\nRows 1-2, Columns 2-4:\n{slice1}")

    # Every other row
    slice2 = tensor[::2]
    print(f"\nEvery other row:\n{slice2}")

    # Every other element in each row
    slice3 = tensor[:, ::2]
    print(f"\nEvery other column:\n{slice3}")

    # Reverse rows
    reverse_rows = tensor[::-1]
    print(f"\nReversed rows:\n{reverse_rows}")

    # Reverse columns
    reverse_cols = tensor[:, ::-1]
    print(f"\nReversed columns:\n{reverse_cols}")

    # Complex slicing
    complex_slice = tensor[1:4:2, ::3]
    print(f"\nComplex slice (rows 1,3 and every 3rd column):\n{complex_slice}")
    print()


def boolean_masking():
    """Boolean indexing and masking."""
    print("3. Boolean Masking")
    print("-" * 50)

    tensor = torch.randn(4, 5)
    print(f"Original tensor:\n{tensor}")

    # Create boolean mask
    mask = tensor > 0
    print(f"\nBoolean mask (> 0):\n{mask}")

    # Apply mask
    positive_values = tensor[mask]
    print(f"\nPositive values: {positive_values}")

    # Conditional assignment
    tensor_copy = tensor.clone()
    tensor_copy[tensor_copy < 0] = 0
    print(f"\nNegative values set to 0:\n{tensor_copy}")

    # Multiple conditions
    mask_complex = (tensor > -0.5) & (tensor < 0.5)
    values_in_range = tensor[mask_complex]
    print(f"\nValues in range [-0.5, 0.5]: {values_in_range}")

    # Where operation
    result = torch.where(tensor > 0, tensor, torch.zeros_like(tensor))
    print(f"\ntorch.where (keep positive, zero negative):\n{result}")
    print()


def advanced_indexing():
    """Advanced indexing techniques."""
    print("4. Advanced Indexing")
    print("-" * 50)

    tensor = torch.arange(20).reshape(4, 5)
    print(f"Original tensor (4x5):\n{tensor}")

    # Index with list/tensor of indices
    row_indices = torch.tensor([0, 2, 3])
    selected_rows = tensor[row_indices]
    print(f"\nRows [0, 2, 3]:\n{selected_rows}")

    # Select specific elements
    row_idx = torch.tensor([0, 1, 2, 3])
    col_idx = torch.tensor([0, 2, 4, 1])
    elements = tensor[row_idx, col_idx]
    print(f"\nSpecific elements [0,0], [1,2], [2,4], [3,1]: {elements}")

    # Gather operation
    indices = torch.tensor([[0, 1], [2, 3]])
    gathered = torch.gather(tensor, 1, indices)
    print(f"\nGathered elements:\n{gathered}")

    # Index select
    selected = torch.index_select(tensor, 0, torch.tensor([1, 3]))
    print(f"\nIndex select (rows 1, 3):\n{selected}")
    print()


def nonzero_and_masking():
    """Find non-zero elements and advanced masking."""
    print("5. Non-zero and Masking")
    print("-" * 50)

    tensor = torch.tensor([[0, 1, 0], [2, 0, 3], [0, 0, 4]])
    print(f"Tensor:\n{tensor}")

    # Find non-zero elements
    nonzero_idx = torch.nonzero(tensor)
    print(f"\nNon-zero indices:\n{nonzero_idx}")

    # Get non-zero values
    nonzero_values = tensor[tensor != 0]
    print(f"Non-zero values: {nonzero_values}")

    # Masked select
    mask = tensor > 1
    masked = torch.masked_select(tensor, mask)
    print(f"\nMasked select (> 1): {masked}")

    # Masked fill
    tensor_filled = tensor.clone()
    tensor_filled.masked_fill_(tensor == 0, -1)
    print(f"\nMasked fill (zeros -> -1):\n{tensor_filled}")
    print()


def ellipsis_and_none():
    """Using ellipsis and None in indexing."""
    print("6. Ellipsis and None Indexing")
    print("-" * 50)

    tensor = torch.randn(2, 3, 4, 5)
    print(f"Original shape: {tensor.shape}")

    # Ellipsis (...)
    slice_ellipsis = tensor[..., 0]
    print(f"tensor[..., 0] shape: {slice_ellipsis.shape}")

    slice_ellipsis2 = tensor[0, ...]
    print(f"tensor[0, ...] shape: {slice_ellipsis2.shape}")

    # None (adds new dimension)
    expanded = tensor[None, ...]
    print(f"\ntensor[None, ...] shape: {expanded.shape}")

    expanded2 = tensor[:, None, :, :]
    print(f"tensor[:, None, :, :] shape: {expanded2.shape}")

    # Combine both
    result = tensor[0, ..., None]
    print(f"tensor[0, ..., None] shape: {result.shape}")
    print()


def practical_examples():
    """Practical indexing examples."""
    print("7. Practical Examples")
    print("-" * 50)

    # Batch data selection
    batch_data = torch.randn(32, 3, 28, 28)  # 32 images, 3 channels, 28x28
    print(f"Batch shape: {batch_data.shape}")

    # Select first 5 images
    first_five = batch_data[:5]
    print(f"First 5 images shape: {first_five.shape}")

    # Select all red channels
    red_channels = batch_data[:, 0, :, :]
    print(f"Red channels shape: {red_channels.shape}")

    # Center crop
    center_crop = batch_data[:, :, 7:21, 7:21]
    print(f"Center crop (14x14) shape: {center_crop.shape}")

    # Select samples based on condition
    labels = torch.randint(0, 10, (32,))
    class_3_indices = (labels == 3).nonzero(as_tuple=True)[0]
    class_3_samples = batch_data[class_3_indices]
    print(f"\nClass 3 samples: {len(class_3_indices)} out of 32")
    print(f"Class 3 data shape: {class_3_samples.shape}")
    print()


def main():
    print("=" * 60)
    print("TENSOR INDEXING AND SLICING")
    print("=" * 60)
    print()

    check_environment()
    basic_indexing()
    slicing_operations()
    boolean_masking()
    advanced_indexing()
    nonzero_and_masking()
    ellipsis_and_none()
    practical_examples()

    print("=" * 60)


if __name__ == "__main__":
    main()
