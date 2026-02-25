"""
TorchVision Datasets - Built-in Image Datasets

This module demonstrates:
- Built-in datasets in torchvision
- MNIST, Fashion-MNIST, CIFAR-10/100
- ImageNet and other large datasets
- Dataset properties and exploration
- Standard transforms for different datasets
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def explore_mnist():
    """Explore MNIST dataset"""
    print("=" * 60)
    print("MNIST Dataset")
    print("=" * 60)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    print(f"\nDataset: MNIST (handwritten digits)")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: 10 (digits 0-9)")
    print(f"Image size: 28x28 grayscale")

    # Get a sample
    img, label = train_dataset[0]
    print(f"\nSample image shape: {img.shape}")
    print(f"Sample label: {label}")
    print(f"Pixel value range: [{img.min():.3f}, {img.max():.3f}]")

    return train_dataset, test_dataset


def explore_fashion_mnist():
    """Explore Fashion-MNIST dataset"""
    print("\n" + "=" * 60)
    print("Fashion-MNIST Dataset")
    print("=" * 60)

    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(f"\nDataset: Fashion-MNIST (clothing items)")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Classes: 10")
    print(f"Class names: {class_names}")
    print(f"Image size: 28x28 grayscale")

    # Class distribution
    labels = [label for _, label in train_dataset]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nClass distribution:")
    for cls, count in zip(class_names, counts):
        print(f"  {cls}: {count}")

    return train_dataset, class_names


def explore_cifar10():
    """Explore CIFAR-10 dataset"""
    print("\n" + "=" * 60)
    print("CIFAR-10 Dataset")
    print("=" * 60)

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print(f"\nDataset: CIFAR-10 (natural images)")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: 10")
    print(f"Class names: {class_names}")
    print(f"Image size: 32x32 RGB (3 channels)")

    # Get a sample
    img, label = train_dataset[0]
    print(f"\nSample image shape: {img.shape}")
    print(f"Sample label: {label} ({class_names[label]})")

    return train_dataset, test_dataset, class_names


def explore_cifar100():
    """Explore CIFAR-100 dataset"""
    print("\n" + "=" * 60)
    print("CIFAR-100 Dataset")
    print("=" * 60)

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    print(f"\nDataset: CIFAR-100 (100 fine-grained classes)")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Classes: 100 (grouped into 20 superclasses)")
    print(f"Image size: 32x32 RGB")
    print(f"Samples per class: ~500 training, ~100 test")

    # Show first 10 class names
    print(f"\nFirst 10 classes: {train_dataset.classes[:10]}")

    return train_dataset


def explore_svhn():
    """Explore SVHN (Street View House Numbers) dataset"""
    print("\n" + "=" * 60)
    print("SVHN Dataset")
    print("=" * 60)

    transform = transforms.ToTensor()

    train_dataset = datasets.SVHN(
        root='./data',
        split='train',
        download=True,
        transform=transform
    )

    test_dataset = datasets.SVHN(
        root='./data',
        split='test',
        download=True,
        transform=transform
    )

    print(f"\nDataset: SVHN (Street View House Numbers)")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: 10 (digits 0-9)")
    print(f"Image size: 32x32 RGB")
    print(f"Source: Real-world images from Google Street View")

    return train_dataset, test_dataset


def demonstrate_transforms():
    """Demonstrate standard transforms for datasets"""
    print("\n" + "=" * 60)
    print("Standard Transforms")
    print("=" * 60)

    # MNIST transforms
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    print("\nMNIST transforms:")
    print("  - ToTensor: Convert PIL Image to tensor")
    print("  - Normalize: mean=0.1307, std=0.3081")

    # CIFAR-10 transforms (training)
    cifar_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    print("\nCIFAR-10 training transforms:")
    print("  - RandomCrop: Random 32x32 crop with padding=4")
    print("  - RandomHorizontalFlip: 50% chance to flip")
    print("  - ToTensor: Convert to tensor")
    print("  - Normalize: per-channel mean and std")

    # CIFAR-10 transforms (test)
    cifar_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    print("\nCIFAR-10 test transforms:")
    print("  - ToTensor: Convert to tensor")
    print("  - Normalize: same as training")


def compare_datasets():
    """Compare different datasets"""
    print("\n" + "=" * 60)
    print("Dataset Comparison")
    print("=" * 60)

    datasets_info = [
        ("MNIST", 60000, 10000, "28x28x1", 10, "Handwritten digits"),
        ("Fashion-MNIST", 60000, 10000, "28x28x1", 10, "Clothing items"),
        ("CIFAR-10", 50000, 10000, "32x32x3", 10, "Natural images"),
        ("CIFAR-100", 50000, 10000, "32x32x3", 100, "100 fine classes"),
        ("SVHN", 73257, 26032, "32x32x3", 10, "House numbers"),
        ("ImageNet", 1281167, 50000, "varies", 1000, "Natural images"),
    ]

    print(f"\n{'Dataset':<15} {'Train':<8} {'Test':<8} {'Size':<10} {'Classes':<8} Description")
    print("-" * 80)
    for name, train, test, size, classes, desc in datasets_info:
        print(f"{name:<15} {train:<8} {test:<8} {size:<10} {classes:<8} {desc}")


def visualize_datasets():
    """Visualize samples from different datasets"""
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    fig.suptitle('Samples from Different Datasets', fontsize=16)

    # MNIST
    mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    for i in range(6):
        img, label = mnist_dataset[i * 1000]
        axes[0, i].imshow(img.squeeze(), cmap='gray')
        axes[0, i].set_title(f'MNIST: {label}')
        axes[0, i].axis('off')

    # Fashion-MNIST
    fashion_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    fashion_classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal']
    for i in range(6):
        img, label = fashion_dataset[i * 1000]
        axes[1, i].imshow(img.squeeze(), cmap='gray')
        axes[1, i].set_title(f'F-MNIST: {fashion_classes[label] if label < 6 else label}')
        axes[1, i].axis('off')

    # CIFAR-10
    cifar_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    cifar_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog']
    for i in range(6):
        img, label = cifar_dataset[i * 1000]
        img_np = img.permute(1, 2, 0).numpy()
        axes[2, i].imshow(img_np)
        axes[2, i].set_title(f'CIFAR: {cifar_classes[label] if label < 6 else label}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('/tmp/dataset_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to /tmp/dataset_comparison.png")
    plt.close()


def demonstrate_dataloaders():
    """Demonstrate DataLoader with different datasets"""
    print("\n" + "=" * 60)
    print("DataLoader Usage")
    print("=" * 60)

    mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

    # Create DataLoader
    dataloader = DataLoader(
        mnist_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True  # Faster data transfer to GPU
    )

    print(f"\nDataLoader configuration:")
    print(f"  Dataset: MNIST")
    print(f"  Batch size: 32")
    print(f"  Shuffle: True")
    print(f"  Num workers: 2")
    print(f"  Total batches: {len(dataloader)}")

    # Get one batch
    images, labels = next(iter(dataloader))
    print(f"\nBatch properties:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels in batch: {labels.unique().tolist()}")


def main():
    """Main demonstration function"""
    print("\n" + "=" * 60)
    print("TORCHVISION BUILT-IN DATASETS")
    print("=" * 60)

    # Explore different datasets
    explore_mnist()
    explore_fashion_mnist()
    cifar10_train, cifar10_test, cifar_classes = explore_cifar10()
    explore_cifar100()
    explore_svhn()

    # Demonstrate transforms
    demonstrate_transforms()

    # Compare datasets
    compare_datasets()

    # Visualize datasets
    visualize_datasets()

    # Demonstrate DataLoaders
    demonstrate_dataloaders()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. TorchVision provides many built-in datasets")
    print("2. All datasets follow the same interface")
    print("3. Transforms preprocess images consistently")
    print("4. Choose dataset based on problem complexity")
    print("5. MNIST: simple, Fashion-MNIST: harder, CIFAR-10: realistic")
    print("6. Use DataLoader for efficient batching")
    print("=" * 60)


if __name__ == "__main__":
    main()
