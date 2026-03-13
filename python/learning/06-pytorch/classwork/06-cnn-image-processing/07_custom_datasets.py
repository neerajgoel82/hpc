"""
Custom Datasets - Creating Custom Dataset Classes

This module demonstrates:
- Implementing custom Dataset class
- Loading images from directories
- Custom transforms and augmentation
- Handling different data formats
- Dataset composition and wrappers
- Memory-mapped datasets for large data
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json


class SimpleImageDataset(Dataset):
    """
    Simple custom dataset for image classification

    Expected directory structure:
    root/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img3.jpg
            img4.jpg
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Scan directory for images
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        if not os.path.exists(root_dir):
            # Create dummy data for demonstration
            self._create_dummy_data()
        else:
            self._load_data()

    def _create_dummy_data(self):
        """Create dummy data for demonstration"""
        # Create synthetic images
        for class_idx in range(3):
            for img_idx in range(10):
                # Create random image
                img = torch.randn(3, 32, 32)
                self.images.append(img)
                self.labels.append(class_idx)

        self.class_to_idx = {'class_0': 0, 'class_1': 1, 'class_2': 2}

    def _load_data(self):
        """Load data from directory"""
        classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for class_name in classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(self.images[idx], str):
            # Load image from file
            image = Image.open(self.images[idx]).convert('RGB')
        else:
            # Use pre-loaded image
            image = self.images[idx]

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class ImageLabelDataset(Dataset):
    """
    Dataset that loads images and labels from separate files

    Format:
    - images: list of image paths or numpy array
    - labels: list/array of corresponding labels
    """

    def __init__(self, image_paths, labels, transform=None):
        assert len(image_paths) == len(labels), "Images and labels must have same length"

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            # Assume numpy array
            image = Image.fromarray(image_path.astype('uint8'))

        if self.transform:
            image = self.transform(image)

        return image, label


class MemoryMappedDataset(Dataset):
    """
    Memory-mapped dataset for large datasets

    Uses numpy memmap to avoid loading entire dataset into memory
    """

    def __init__(self, data_file, label_file, shape, transform=None):
        self.data = np.memmap(data_file, dtype='uint8', mode='r', shape=shape)
        self.labels = np.load(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # Convert to PIL Image
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


class MultiModalDataset(Dataset):
    """
    Dataset that returns multiple data modalities

    Example: image + metadata
    """

    def __init__(self, images, labels, metadata, transform=None):
        self.images = images
        self.labels = labels
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        meta = self.metadata[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, meta


class CachedDataset(Dataset):
    """
    Dataset that caches transformed images in memory

    Useful when transforms are expensive
    """

    def __init__(self, base_dataset, cache_size=1000):
        self.base_dataset = base_dataset
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        item = self.base_dataset[idx]

        if len(self.cache) < self.cache_size:
            self.cache[idx] = item

        return item


def demonstrate_simple_dataset():
    """Demonstrate simple custom dataset"""
    print("=" * 60)
    print("Simple Custom Dataset")
    print("=" * 60)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = SimpleImageDataset(root_dir='./data/custom', transform=transform)

    print(f"\nDataset length: {len(dataset)}")
    print(f"Classes: {dataset.class_to_idx}")

    # Get a sample
    image, label = dataset[0]
    print(f"\nSample:")
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label}")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, labels = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels}")


def demonstrate_image_label_dataset():
    """Demonstrate image-label dataset"""
    print("\n" + "=" * 60)
    print("Image-Label Dataset")
    print("=" * 60)

    # Create synthetic data
    num_samples = 100
    image_data = np.random.randint(0, 255, size=(num_samples, 32, 32, 3))
    labels = np.random.randint(0, 10, size=num_samples)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = ImageLabelDataset(image_data, labels, transform=transform)

    print(f"\nDataset length: {len(dataset)}")

    image, label = dataset[0]
    print(f"Sample:")
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label}")


def demonstrate_multimodal_dataset():
    """Demonstrate multi-modal dataset"""
    print("\n" + "=" * 60)
    print("Multi-Modal Dataset")
    print("=" * 60)

    # Create data with metadata
    num_samples = 50
    images = [torch.randn(3, 32, 32) for _ in range(num_samples)]
    labels = [np.random.randint(0, 10) for _ in range(num_samples)]
    metadata = [
        {'age': np.random.randint(20, 60), 'gender': np.random.choice(['M', 'F'])}
        for _ in range(num_samples)
    ]

    dataset = MultiModalDataset(images, labels, metadata)

    print(f"\nDataset length: {len(dataset)}")

    image, label, meta = dataset[0]
    print(f"\nSample:")
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label}")
    print(f"  Metadata: {meta}")


def demonstrate_dataset_composition():
    """Demonstrate composing datasets"""
    print("\n" + "=" * 60)
    print("Dataset Composition")
    print("=" * 60)

    # Create two datasets
    dataset1 = SimpleImageDataset('./data/custom1', transform=transforms.ToTensor())
    dataset2 = SimpleImageDataset('./data/custom2', transform=transforms.ToTensor())

    # Concatenate datasets
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset([dataset1, dataset2])

    print(f"\nDataset 1 length: {len(dataset1)}")
    print(f"Dataset 2 length: {len(dataset2)}")
    print(f"Combined length: {len(combined_dataset)}")

    # Subset dataset
    from torch.utils.data import Subset
    indices = list(range(0, 10))
    subset = Subset(combined_dataset, indices)
    print(f"Subset length: {len(subset)}")


def demonstrate_custom_transforms():
    """Demonstrate custom transform functions"""
    print("\n" + "=" * 60)
    print("Custom Transforms")
    print("=" * 60)

    class AddGaussianNoise:
        """Add Gaussian noise to image"""
        def __init__(self, mean=0., std=0.1):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return tensor + noise

    class Cutout:
        """Randomly mask out one or more patches"""
        def __init__(self, n_holes=1, length=8):
            self.n_holes = n_holes
            self.length = length

        def __call__(self, img):
            h, w = img.shape[1], img.shape[2]

            mask = torch.ones((h, w))
            for _ in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1:y2, x1:x2] = 0.

            mask = mask.expand_as(img)
            return img * mask

    # Use custom transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.1),
        Cutout(n_holes=1, length=8)
    ])

    print("\nCustom transform pipeline:")
    print("  1. ToTensor")
    print("  2. AddGaussianNoise (mean=0, std=0.1)")
    print("  3. Cutout (1 hole, size=8x8)")

    dataset = SimpleImageDataset('./data/custom', transform=transform)
    image, label = dataset[0]
    print(f"\nTransformed image shape: {image.shape}")


def demonstrate_lazy_loading():
    """Demonstrate lazy loading for large datasets"""
    print("\n" + "=" * 60)
    print("Lazy Loading Strategy")
    print("=" * 60)

    class LazyLoadDataset(Dataset):
        """Only load data when accessed"""

        def __init__(self, data_dir):
            self.data_dir = data_dir
            # Only store paths, not actual data
            self.file_paths = self._scan_directory()

        def _scan_directory(self):
            # Scan but don't load
            return ['path1.jpg', 'path2.jpg', 'path3.jpg']  # Example

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            # Load data only when needed
            path = self.file_paths[idx]
            # image = Image.open(path)  # Lazy load
            image = torch.randn(3, 32, 32)  # Dummy for demo
            label = 0
            return image, label

    dataset = LazyLoadDataset('./data')
    print(f"\nLazy loading dataset:")
    print(f"  Dataset length: {len(dataset)}")
    print(f"  Memory usage: Only file paths stored")
    print(f"  Data loaded: Only when __getitem__ is called")


def main():
    """Main demonstration function"""
    print("\n" + "=" * 60)
    print("CUSTOM DATASET IMPLEMENTATION")
    print("=" * 60)

    # Simple dataset
    demonstrate_simple_dataset()

    # Image-label dataset
    demonstrate_image_label_dataset()

    # Multi-modal dataset
    demonstrate_multimodal_dataset()

    # Dataset composition
    demonstrate_dataset_composition()

    # Custom transforms
    demonstrate_custom_transforms()

    # Lazy loading
    demonstrate_lazy_loading()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Custom Dataset needs __init__, __len__, __getitem__")
    print("2. Lazy loading saves memory for large datasets")
    print("3. Transforms can be customized for specific needs")
    print("4. Datasets can be concatenated and subsetted")
    print("5. Multi-modal data: return tuples/dicts")
    print("6. Memory-mapped data for huge datasets")
    print("=" * 60)


if __name__ == "__main__":
    main()
