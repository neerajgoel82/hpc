"""
Reshaping and Views
Tensor reshaping, viewing, transposing, and dimension manipulation

This module demonstrates:
- reshape vs view vs contiguous
- transpose and permute
- squeeze and unsqueeze
- flatten and unflatten
- Understanding memory layout

Run: python 05_reshaping_views.py
"""

import torch


def check_environment():
    """Display PyTorch environment information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()


def reshape_vs_view():
    """Difference between reshape and view."""
    print("1. Reshape vs View")
    print("-" * 50)

    # Original tensor
    original = torch.arange(12)
    print(f"Original shape: {original.shape}")
    print(f"Original: {original}")

    # View (requires contiguous memory)
    viewed = original.view(3, 4)
    print(f"\nView (3, 4):\n{viewed}")
    print(f"Shares memory: {viewed.data_ptr() == original.data_ptr()}")

    # Reshape (may copy if not contiguous)
    reshaped = original.reshape(4, 3)
    print(f"\nReshape (4, 3):\n{reshaped}")

    # Modifying view affects original
    viewed[0, 0] = 999
    print(f"\nAfter modifying view:")
    print(f"Original: {original}")
    print(f"View:\n{viewed}")

    # View on non-contiguous tensor fails
    transposed = original.view(3, 4).t()
    print(f"\nTransposed is contiguous: {transposed.is_contiguous()}")
    try:
        bad_view = transposed.view(12)
        print("View succeeded (unexpected)")
    except RuntimeError:
        print("View failed on non-contiguous tensor (expected)")
        # Use reshape instead
        good_reshape = transposed.reshape(12)
        print(f"Reshape succeeded: {good_reshape.shape}")
    print()


def transpose_and_permute():
    """Transposing and permuting dimensions."""
    print("2. Transpose and Permute")
    print("-" * 50)

    tensor = torch.arange(24).reshape(2, 3, 4)
    print(f"Original shape: {tensor.shape}")
    print(f"Original:\n{tensor}")

    # Transpose (swap two dimensions)
    transposed = tensor.transpose(0, 1)
    print(f"\nTranspose dims 0,1: {transposed.shape}")
    print(f"Result:\n{transposed}")

    # t() for 2D tensors
    matrix = torch.arange(6).reshape(2, 3)
    transposed_2d = matrix.t()
    print(f"\nMatrix:\n{matrix}")
    print(f"Transposed (t()):\n{transposed_2d}")

    # Permute (rearrange all dimensions)
    permuted = tensor.permute(2, 0, 1)
    print(f"\nPermute (2, 0, 1): {tensor.shape} -> {permuted.shape}")

    # Example: Convert from NCHW to NHWC format
    batch = torch.randn(8, 3, 224, 224)  # Batch, Channels, Height, Width
    nhwc = batch.permute(0, 2, 3, 1)     # Batch, Height, Width, Channels
    print(f"\nNCHW to NHWC: {batch.shape} -> {nhwc.shape}")
    print()


def squeeze_and_unsqueeze():
    """Adding and removing dimensions."""
    print("3. Squeeze and Unsqueeze")
    print("-" * 50)

    # Unsqueeze (add dimension)
    tensor = torch.arange(6).reshape(2, 3)
    print(f"Original shape: {tensor.shape}")
    print(f"Original:\n{tensor}")

    # Add dimension at different positions
    unsqueezed_0 = tensor.unsqueeze(0)
    unsqueezed_1 = tensor.unsqueeze(1)
    unsqueezed_2 = tensor.unsqueeze(2)

    print(f"\nUnsqueeze(0): {unsqueezed_0.shape}")
    print(f"Unsqueeze(1): {unsqueezed_1.shape}")
    print(f"Unsqueeze(2): {unsqueezed_2.shape}")

    # Negative indexing
    unsqueezed_neg = tensor.unsqueeze(-1)
    print(f"Unsqueeze(-1): {unsqueezed_neg.shape}")

    # Squeeze (remove dimensions of size 1)
    tensor_with_ones = torch.randn(1, 3, 1, 4, 1)
    print(f"\nTensor with 1s: {tensor_with_ones.shape}")

    squeezed_all = tensor_with_ones.squeeze()
    print(f"Squeeze all: {squeezed_all.shape}")

    squeezed_specific = tensor_with_ones.squeeze(0)
    print(f"Squeeze dim 0: {squeezed_specific.shape}")

    squeezed_specific2 = tensor_with_ones.squeeze(2)
    print(f"Squeeze dim 2: {squeezed_specific2.shape}")
    print()


def flatten_and_unflatten():
    """Flattening and unflattening tensors."""
    print("4. Flatten and Unflatten")
    print("-" * 50)

    tensor = torch.arange(24).reshape(2, 3, 4)
    print(f"Original shape: {tensor.shape}")

    # Flatten all dimensions
    flattened = tensor.flatten()
    print(f"\nFlatten all: {flattened.shape}")
    print(f"Flattened: {flattened}")

    # Flatten starting from a dimension
    flat_from_1 = tensor.flatten(start_dim=1)
    print(f"\nFlatten from dim 1: {flat_from_1.shape}")
    print(f"Result:\n{flat_from_1}")

    # Flatten a range of dimensions
    flat_range = tensor.flatten(start_dim=1, end_dim=2)
    print(f"\nFlatten dims 1-2: {flat_range.shape}")

    # Unflatten (inverse of flatten)
    unflattened = flat_from_1.unflatten(1, (3, 4))
    print(f"\nUnflatten (3, 4): {unflattened.shape}")
    print(f"Matches original: {torch.equal(tensor, unflattened)}")

    # Practical example: Batch of images
    batch_images = torch.randn(32, 3, 28, 28)
    flattened_images = batch_images.flatten(start_dim=1)
    print(f"\nBatch images: {batch_images.shape}")
    print(f"Flattened (keep batch): {flattened_images.shape}")
    print()


def contiguous_memory():
    """Understanding contiguous memory layout."""
    print("5. Contiguous Memory")
    print("-" * 50)

    # Contiguous tensor
    tensor = torch.arange(6).reshape(2, 3)
    print(f"Original tensor:\n{tensor}")
    print(f"Is contiguous: {tensor.is_contiguous()}")
    print(f"Stride: {tensor.stride()}")

    # Non-contiguous after transpose
    transposed = tensor.t()
    print(f"\nTransposed:\n{transposed}")
    print(f"Is contiguous: {transposed.is_contiguous()}")
    print(f"Stride: {transposed.stride()}")

    # Make contiguous
    made_contiguous = transposed.contiguous()
    print(f"\nMade contiguous:")
    print(f"Is contiguous: {made_contiguous.is_contiguous()}")
    print(f"Stride: {made_contiguous.stride()}")

    # Stride explanation
    tensor_3d = torch.randn(2, 3, 4)
    print(f"\n3D tensor shape: {tensor_3d.shape}")
    print(f"Stride: {tensor_3d.stride()}")
    print("Stride (2, 3, 4) means:")
    print("  - Move 12 elements to go to next batch")
    print("  - Move 4 elements to go to next row")
    print("  - Move 1 element to go to next column")
    print()


def advanced_operations():
    """Advanced reshaping operations."""
    print("6. Advanced Reshaping")
    print("-" * 50)

    # Expand (broadcast to larger size)
    tensor = torch.tensor([[1], [2], [3]])
    expanded = tensor.expand(3, 4)
    print(f"Original shape: {tensor.shape}")
    print(f"Expanded shape: {expanded.shape}")
    print(f"Expanded:\n{expanded}")
    print(f"Shares memory: {expanded.data_ptr() == tensor.data_ptr()}")

    # Repeat (actually duplicate data)
    repeated = tensor.repeat(2, 3)
    print(f"\nRepeated (2, 3): {repeated.shape}")
    print(f"Repeated:\n{repeated}")

    # Chunk (split into chunks)
    tensor_large = torch.arange(12)
    chunks = torch.chunk(tensor_large, 3)
    print(f"\nOriginal: {tensor_large}")
    print(f"Chunks (3): {[c for c in chunks]}")

    # Split (split by sizes)
    splits = torch.split(tensor_large, [3, 4, 5])
    print(f"Split by [3, 4, 5]: {[s for s in splits]}")

    # Stack (combine tensors along new dimension)
    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([4, 5, 6])
    stacked = torch.stack([t1, t2])
    print(f"\nStack along dim 0: {stacked.shape}")
    print(f"Stacked:\n{stacked}")

    stacked_dim1 = torch.stack([t1, t2], dim=1)
    print(f"\nStack along dim 1: {stacked_dim1.shape}")
    print(f"Stacked:\n{stacked_dim1}")

    # Cat (concatenate along existing dimension)
    concatenated = torch.cat([t1, t2])
    print(f"\nConcatenate: {concatenated.shape}")
    print(f"Concatenated: {concatenated}")
    print()


def practical_examples():
    """Practical reshaping examples."""
    print("7. Practical Examples")
    print("-" * 50)

    # Example 1: Prepare batch for neural network
    images = torch.randn(10, 3, 64, 64)  # 10 images, 3 channels, 64x64
    flattened = images.flatten(start_dim=1)
    print("Example 1: Flatten images for fully connected layer")
    print(f"Original: {images.shape}")
    print(f"Flattened: {flattened.shape}")

    # Example 2: Add batch dimension
    single_image = torch.randn(3, 224, 224)
    batched = single_image.unsqueeze(0)
    print(f"\nExample 2: Add batch dimension")
    print(f"Single image: {single_image.shape}")
    print(f"Batched: {batched.shape}")

    # Example 3: Reshape for RNN (seq_len, batch, features)
    batch_first = torch.randn(32, 10, 512)  # Batch, sequence, features
    seq_first = batch_first.transpose(0, 1)
    print(f"\nExample 3: Batch-first to sequence-first")
    print(f"Batch first: {batch_first.shape}")
    print(f"Sequence first: {seq_first.shape}")

    # Example 4: Patch extraction
    image = torch.randn(1, 3, 224, 224)
    patches = image.unfold(2, 16, 16).unfold(3, 16, 16)
    print(f"\nExample 4: Extract 16x16 patches")
    print(f"Image: {image.shape}")
    print(f"Patches: {patches.shape}")
    print()


def main():
    print("=" * 60)
    print("RESHAPING AND VIEWS")
    print("=" * 60)
    print()

    check_environment()
    reshape_vs_view()
    transpose_and_permute()
    squeeze_and_unsqueeze()
    flatten_and_unflatten()
    contiguous_memory()
    advanced_operations()
    practical_examples()

    print("=" * 60)


if __name__ == "__main__":
    main()
