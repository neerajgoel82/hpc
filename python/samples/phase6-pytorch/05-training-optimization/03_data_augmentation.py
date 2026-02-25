"""
Data Augmentation in PyTorch
===========================
Demonstrates image augmentation techniques to improve model generalization.

Benefits of Data Augmentation:
- Increases effective training data size
- Reduces overfitting
- Improves model robustness
- Better generalization to unseen data

Common techniques:
- RandomHorizontalFlip
- RandomRotation
- RandomCrop
- ColorJitter
- RandomAffine
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def generate_synthetic_images(num_samples=500, num_classes=10):
    """Generate synthetic 32x32 RGB images."""
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (num_samples,))
    return images, labels


def create_basic_transform():
    """Basic transform: only normalization."""
    return transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def create_augmentation_transform():
    """Full augmentation pipeline."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def apply_augmentations(images, transform):
    """Apply transformations to a batch of images."""
    augmented = []
    for img in images:
        augmented.append(transform(img))
    return torch.stack(augmented)


def train_model(model, optimizer, X_train, y_train, X_val, y_val,
                transform=None, num_epochs=25, batch_size=32):
    """Train model with optional data augmentation."""
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    val_accuracies = []

    num_samples = X_train.size(0)

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0

        indices = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i+batch_size, num_samples)]
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]

            # Apply augmentation during training
            if transform is not None:
                batch_X = apply_augmentations(batch_X, transform)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / (num_samples // batch_size)
        train_losses.append(avg_train_loss)

        # Validation (no augmentation)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val).float().mean().item() * 100
            val_accuracies.append(accuracy)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Acc: {accuracy:.2f}%")

    return train_losses, val_losses, val_accuracies


def visualize_augmentations():
    """Show examples of different augmentation techniques."""
    print("\n" + "="*60)
    print("Visualizing Data Augmentation Techniques")
    print("="*60)

    # Create a sample image
    torch.manual_seed(42)
    original_image = torch.randn(3, 32, 32)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    # Different augmentation techniques
    augmentations = {
        'Original': transforms.Compose([]),
        'Horizontal Flip': transforms.RandomHorizontalFlip(p=1.0),
        'Rotation (30°)': transforms.RandomRotation(degrees=30),
        'Color Jitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        'Random Crop': transforms.Compose([
            transforms.RandomCrop(size=32, padding=4)
        ]),
        'Combined': transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3)
        ])
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (name, transform) in enumerate(augmentations.items()):
        if name == 'Original':
            img = original_image
        else:
            torch.manual_seed(42 + idx)  # Different seed for variety
            img = transform(original_image)

        # Convert to displayable format
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        axes[idx].imshow(img_np)
        axes[idx].set_title(name)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('data_augmentation_examples.png', dpi=150, bbox_inches='tight')
    print("Augmentation examples saved as 'data_augmentation_examples.png'")
    plt.show()


def compare_with_without_augmentation():
    """Compare training with and without data augmentation."""
    print("\n" + "="*60)
    print("Comparing Training: With vs Without Augmentation")
    print("="*60)

    # Set seed
    torch.manual_seed(42)

    # Generate data
    print("\nGenerating synthetic image data...")
    X_train, y_train = generate_synthetic_images(num_samples=500)
    X_val, y_val = generate_synthetic_images(num_samples=100)

    # Normalize validation data
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    X_val = torch.stack([normalize(img) for img in X_val])

    # Train WITHOUT augmentation
    print("\n" + "-"*60)
    print("Training WITHOUT Data Augmentation")
    print("-"*60)

    model_no_aug = SimpleCNN()
    optimizer_no_aug = optim.Adam(model_no_aug.parameters(), lr=0.001)

    basic_transform = create_basic_transform()
    train_loss_no_aug, val_loss_no_aug, val_acc_no_aug = train_model(
        model_no_aug, optimizer_no_aug, X_train, y_train, X_val, y_val,
        transform=basic_transform
    )

    # Train WITH augmentation
    print("\n" + "-"*60)
    print("Training WITH Data Augmentation")
    print("-"*60)

    torch.manual_seed(42)
    model_with_aug = SimpleCNN()
    optimizer_with_aug = optim.Adam(model_with_aug.parameters(), lr=0.001)

    aug_transform = create_augmentation_transform()
    train_loss_with_aug, val_loss_with_aug, val_acc_with_aug = train_model(
        model_with_aug, optimizer_with_aug, X_train, y_train, X_val, y_val,
        transform=aug_transform
    )

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Without Augmentation - Final Val Acc: {val_acc_no_aug[-1]:.2f}%")
    print(f"With Augmentation    - Final Val Acc: {val_acc_with_aug[-1]:.2f}%")
    print(f"\nImprovement: {val_acc_with_aug[-1] - val_acc_no_aug[-1]:.2f}% accuracy")

    return (train_loss_no_aug, val_loss_no_aug, val_acc_no_aug,
            train_loss_with_aug, val_loss_with_aug, val_acc_with_aug)


def visualize_training_comparison(results):
    """Visualize training comparison."""
    train_loss_no_aug, val_loss_no_aug, val_acc_no_aug, \
    train_loss_with_aug, val_loss_with_aug, val_acc_with_aug = results

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_loss_no_aug) + 1)

    # Validation Loss
    ax1 = axes[0]
    ax1.plot(epochs, val_loss_no_aug, 'b-', label='Without Augmentation', linewidth=2)
    ax1.plot(epochs, val_loss_with_aug, 'r-', label='With Augmentation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Validation Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, val_acc_no_aug, 'b-', label='Without Augmentation', linewidth=2)
    ax2.plot(epochs, val_acc_with_aug, 'r-', label='With Augmentation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('augmentation_training_comparison.png', dpi=150, bbox_inches='tight')
    print("\nTraining comparison saved as 'augmentation_training_comparison.png'")
    plt.show()


def main():
    """Main demonstration function."""
    print("="*60)
    print("DATA AUGMENTATION IN PYTORCH")
    print("="*60)

    # Visualize augmentation techniques
    visualize_augmentations()

    # Compare training with and without augmentation
    results = compare_with_without_augmentation()

    # Visualize training comparison
    visualize_training_comparison(results)

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("1. Data augmentation increases effective training data")
    print("2. Reduces overfitting by showing model varied examples")
    print("3. Apply only during training, not validation/testing")
    print("4. Common: flip, rotation, crop, color jitter")
    print("5. Can improve accuracy by 2-10% depending on dataset")
    print("6. Domain-specific augmentations work best")
    print("7. Balance: too much augmentation can hurt performance")
    print("="*60)


if __name__ == "__main__":
    main()
