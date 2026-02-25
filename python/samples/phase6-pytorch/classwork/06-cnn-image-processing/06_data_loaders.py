"""
Data Loaders - Efficient Data Loading with DataLoader

This module demonstrates:
- DataLoader configuration and usage
- Batching and shuffling strategies
- Multi-worker data loading
- Pin memory for GPU training
- Custom collate functions
- Data sampling strategies
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
import time


def demonstrate_basic_dataloader():
    """Demonstrate basic DataLoader usage"""
    print("=" * 60)
    print("Basic DataLoader")
    print("=" * 60)

    # Load MNIST
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Batch size: 64")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Shuffle: True")

    # Get a batch
    images, labels = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")


def demonstrate_batching():
    """Demonstrate different batch sizes"""
    print("\n" + "=" * 60)
    print("Batching Strategies")
    print("=" * 60)

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

    batch_sizes = [32, 64, 128, 256]

    print("\nBatch size effects:")
    for batch_size in batch_sizes:
        dataloader = DataLoader(dataset, batch_size=batch_size)
        print(f"  Batch size {batch_size}: {len(dataloader)} batches")

    # Last batch handling
    print("\n" + "-" * 60)
    print("Last batch handling:")

    # drop_last=False (default)
    dataloader_keep = DataLoader(dataset, batch_size=100, drop_last=False)
    last_batch_size = len(dataset) % 100
    print(f"  drop_last=False: Keep last batch (size={last_batch_size})")

    # drop_last=True
    dataloader_drop = DataLoader(dataset, batch_size=100, drop_last=True)
    print(f"  drop_last=True: Drop last batch")
    print(f"  Batches: {len(dataloader_keep)} vs {len(dataloader_drop)}")


def demonstrate_shuffling():
    """Demonstrate shuffling effects"""
    print("\n" + "=" * 60)
    print("Shuffling")
    print("=" * 60)

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

    # Without shuffling
    dataloader_no_shuffle = DataLoader(dataset, batch_size=10, shuffle=False)
    _, labels1 = next(iter(dataloader_no_shuffle))
    _, labels2 = next(iter(dataloader_no_shuffle))

    print("\nWithout shuffling (shuffle=False):")
    print(f"  First batch labels: {labels1.tolist()}")
    print(f"  Second batch labels: {labels2.tolist()}")

    # With shuffling
    dataloader_shuffle = DataLoader(dataset, batch_size=10, shuffle=True)
    _, labels1 = next(iter(dataloader_shuffle))
    _, labels2 = next(iter(dataloader_shuffle))

    print("\nWith shuffling (shuffle=True):")
    print(f"  First batch labels: {labels1.tolist()}")
    print(f"  Second batch labels: {labels2.tolist()}")
    print("\n  Note: Labels are more randomly distributed")


def demonstrate_num_workers():
    """Demonstrate multi-worker data loading"""
    print("\n" + "=" * 60)
    print("Multi-Worker Data Loading")
    print("=" * 60)

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

    # Test different numbers of workers
    num_workers_list = [0, 2, 4]

    print("\nLoading time comparison (100 batches):")
    for num_workers in num_workers_list:
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=num_workers
        )

        start_time = time.time()
        for i, (images, labels) in enumerate(dataloader):
            if i >= 100:
                break
        elapsed = time.time() - start_time

        print(f"  num_workers={num_workers}: {elapsed:.3f}s")

    print("\n  Optimal num_workers: typically 2-4 for CPU, 4-8 for GPU")
    print("  Too many workers can slow down due to overhead")


def demonstrate_pin_memory():
    """Demonstrate pin_memory for GPU training"""
    print("\n" + "=" * 60)
    print("Pin Memory (GPU Optimization)")
    print("=" * 60)

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

    # Without pin_memory
    dataloader_normal = DataLoader(dataset, batch_size=64, pin_memory=False)

    # With pin_memory
    dataloader_pinned = DataLoader(dataset, batch_size=64, pin_memory=True)

    print("\npin_memory=False (default):")
    print("  - Data loaded to CPU pageable memory")
    print("  - Slower transfer to GPU")

    print("\npin_memory=True:")
    print("  - Data loaded to CPU pinned memory")
    print("  - Faster transfer to GPU")
    print("  - Uses more memory")
    print("  - Recommended when using GPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCurrent device: {device}")

    if torch.cuda.is_available():
        print("  Use pin_memory=True for faster GPU training")


def demonstrate_custom_collate():
    """Demonstrate custom collate function"""
    print("\n" + "=" * 60)
    print("Custom Collate Function")
    print("=" * 60)

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

    # Default collate
    dataloader_default = DataLoader(dataset, batch_size=4)
    images, labels = next(iter(dataloader_default))
    print(f"\nDefault collate:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")

    # Custom collate that adds metadata
    def custom_collate(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])

        # Add batch metadata
        metadata = {
            'batch_size': len(batch),
            'unique_labels': labels.unique().tolist(),
            'mean_pixel_value': images.mean().item()
        }

        return images, labels, metadata

    dataloader_custom = DataLoader(dataset, batch_size=4, collate_fn=custom_collate)
    images, labels, metadata = next(iter(dataloader_custom))

    print(f"\nCustom collate:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Metadata: {metadata}")


def demonstrate_samplers():
    """Demonstrate different sampling strategies"""
    print("\n" + "=" * 60)
    print("Sampling Strategies")
    print("=" * 60)

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

    # Subset sampling
    print("\n1. Subset Sampling:")
    indices = list(range(0, 1000))
    subset = Subset(dataset, indices)
    subset_loader = DataLoader(subset, batch_size=100)
    print(f"  Using first 1000 samples: {len(subset)} samples, {len(subset_loader)} batches")

    # Weighted sampling for imbalanced datasets
    print("\n2. Weighted Random Sampling (for imbalanced data):")

    # Get labels
    labels = [label for _, label in dataset]
    label_counts = np.bincount(labels)

    # Create weights (inverse frequency)
    weights = 1.0 / label_counts[labels]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    weighted_loader = DataLoader(dataset, batch_size=100, sampler=sampler)

    print(f"  Original class distribution: {label_counts}")
    print(f"  Sampler ensures balanced batches")

    # Get a batch and check distribution
    _, batch_labels = next(iter(weighted_loader))
    batch_counts = np.bincount(batch_labels.numpy(), minlength=10)
    print(f"  Batch distribution: {batch_counts}")


def demonstrate_iteration_patterns():
    """Demonstrate different iteration patterns"""
    print("\n" + "=" * 60)
    print("Iteration Patterns")
    print("=" * 60)

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("\n1. Standard iteration:")
    print("   for images, labels in dataloader:")
    print("       # process batch")

    print("\n2. Enumerate with index:")
    print("   for batch_idx, (images, labels) in enumerate(dataloader):")
    print("       # batch_idx available")

    print("\n3. Manual iteration:")
    dataloader_iter = iter(dataloader)
    images, labels = next(dataloader_iter)
    print(f"   images, labels = next(iter(dataloader))")
    print(f"   Got batch: {images.shape}")

    print("\n4. Limited iteration:")
    count = 0
    for images, labels in dataloader:
        count += 1
        if count >= 10:
            break
    print(f"   Processed {count} batches (early stopping)")


def demonstrate_dataloader_best_practices():
    """Demonstrate DataLoader best practices"""
    print("\n" + "=" * 60)
    print("DataLoader Best Practices")
    print("=" * 60)

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

    print("\nTraining configuration:")
    train_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,           # Shuffle for training
        num_workers=4,          # Multi-worker loading
        pin_memory=True,        # Faster GPU transfer
        drop_last=False,        # Keep all data
        persistent_workers=True # Keep workers alive between epochs
    )
    print(f"  batch_size=128")
    print(f"  shuffle=True (randomize training data)")
    print(f"  num_workers=4 (parallel loading)")
    print(f"  pin_memory=True (GPU optimization)")

    print("\nValidation configuration:")
    val_loader = DataLoader(
        dataset,
        batch_size=256,         # Larger batch for validation
        shuffle=False,          # No shuffle for validation
        num_workers=4,
        pin_memory=True
    )
    print(f"  batch_size=256 (larger for inference)")
    print(f"  shuffle=False (deterministic order)")

    print("\nTips:")
    print("  - Training: shuffle=True, smaller batches")
    print("  - Validation: shuffle=False, larger batches")
    print("  - Tune num_workers based on your system")
    print("  - Use pin_memory=True when training on GPU")
    print("  - Consider drop_last=True for batch normalization")


def main():
    """Main demonstration function"""
    print("\n" + "=" * 60)
    print("DATALOADER: EFFICIENT DATA LOADING")
    print("=" * 60)

    # Basic usage
    demonstrate_basic_dataloader()

    # Batching
    demonstrate_batching()

    # Shuffling
    demonstrate_shuffling()

    # Multi-worker loading
    demonstrate_num_workers()

    # Pin memory
    demonstrate_pin_memory()

    # Custom collate
    demonstrate_custom_collate()

    # Samplers
    demonstrate_samplers()

    # Iteration patterns
    demonstrate_iteration_patterns()

    # Best practices
    demonstrate_dataloader_best_practices()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. DataLoader handles batching and shuffling automatically")
    print("2. num_workers enables parallel data loading")
    print("3. pin_memory speeds up GPU data transfer")
    print("4. Custom collate functions allow flexible batching")
    print("5. Samplers enable balanced/weighted sampling")
    print("6. Different configs for training vs validation")
    print("=" * 60)


if __name__ == "__main__":
    main()
