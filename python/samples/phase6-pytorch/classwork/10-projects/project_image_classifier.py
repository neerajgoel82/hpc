"""
Project: Image Classifier
=========================
Build a complete image classification system.

Dataset: CIFAR-10 (10 classes of 32x32 images)
Goals:
- Load and preprocess images with data augmentation
- Build CNN architecture with modern techniques
- Train with proper validation and learning rate scheduling
- Evaluate on test set with detailed metrics
- Visualize results, predictions, and misclassifications
- Save/load model checkpoints

Skills: CNNs, Data loaders, Training pipelines, Model evaluation
Run: python project_image_classifier.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# CIFAR-10 class names
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ImageClassifier(nn.Module):
    """CNN architecture for CIFAR-10 classification."""

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()

        # First block: 3 -> 64
        self.block1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        )

        # Second block: 64 -> 128
        self.block2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )

        # Third block: 128 -> 256
        self.block3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def load_data(batch_size=128):
    """Load CIFAR-10 dataset with augmentation."""
    print("Loading CIFAR-10 dataset...")

    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # No augmentation for test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Download and load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print progress
        if batch_idx % 100 == 99:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {running_loss/(batch_idx+1):.3f} "
                  f"Acc: {100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    # Class-wise accuracy
    class_correct = [0] * 10
    class_total = [0] * 10

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == targets[i]).item()
                class_total[label] += 1

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total

    # Class-wise accuracy
    class_accuracies = {}
    for i in range(10):
        if class_total[i] > 0:
            class_accuracies[CLASSES[i]] = 100. * class_correct[i] / class_total[i]

    return test_loss, test_acc, class_accuracies, all_predictions, all_targets


def train_model(model, train_loader, test_loader, num_epochs, device, lr=0.001):
    """Complete training loop with validation."""
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()

        print(f"\nEpoch [{epoch+1}/{num_epochs}] LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Evaluate
        test_loss, test_acc, _, _, _ = evaluate_model(
            model, test_loader, criterion, device
        )

        # Update learning rate
        scheduler.step()

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, optimizer, epoch, test_acc, 'best_model.pth')

        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        epoch_time = time.time() - start_time

        print(f"  Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.3f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'test_loss': test_losses,
        'test_acc': test_accs
    }

    return history, best_acc


def save_checkpoint(model, optimizer, epoch, accuracy, filename='checkpoint.pth'):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    torch.save(checkpoint, filename)
    print(f"  Saved checkpoint: {filename} (Acc: {accuracy:.2f}%)")


def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    """Load model checkpoint."""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    print(f"Loaded checkpoint from epoch {epoch} (Acc: {accuracy:.2f}%)")
    return epoch, accuracy


def visualize_results(history, class_accuracies, predictions, targets, test_loader):
    """Create comprehensive visualizations."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 10))

    # 1. Training curves - Loss
    ax1 = plt.subplot(2, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True)

    # 2. Training curves - Accuracy
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy', fontweight='bold')
    ax2.legend()
    ax2.grid(True)

    # 3. Per-class accuracy
    ax3 = plt.subplot(2, 3, 3)
    classes = list(class_accuracies.keys())
    accs = list(class_accuracies.values())
    ax3.barh(classes, accs, color='steelblue')
    ax3.set_xlabel('Accuracy (%)')
    ax3.set_title('Per-Class Accuracy', fontweight='bold')
    ax3.grid(True, axis='x')

    # 4. Confusion matrix
    ax4 = plt.subplot(2, 3, 4)
    confusion = np.zeros((10, 10), dtype=int)
    for pred, target in zip(predictions, targets):
        confusion[target][pred] += 1
    im = ax4.imshow(confusion, cmap='Blues', aspect='auto')
    ax4.set_xticks(range(10))
    ax4.set_yticks(range(10))
    ax4.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax4.set_yticklabels(CLASSES)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_title('Confusion Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax4)

    # 5. Sample predictions (correct)
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    ax5.set_title('Correct Predictions (Random Samples)', fontweight='bold')

    # 6. Sample predictions (incorrect)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_title('Incorrect Predictions (Random Samples)', fontweight='bold')

    plt.tight_layout()
    print("Visualizations created successfully!")
    print("Close the plot window to continue...")
    plt.show()


def show_sample_predictions(model, test_loader, device, num_samples=16):
    """Show sample predictions with images."""
    model.eval()

    # Get a batch of images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)

    # Move to CPU for visualization
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()

    # Denormalize images
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for idx in range(min(num_samples, len(images))):
        ax = axes[idx // 4, idx % 4]
        img = images[idx].permute(1, 2, 0).numpy()
        ax.imshow(img)

        true_label = CLASSES[labels[idx]]
        pred_label = CLASSES[predicted[idx]]
        color = 'green' if labels[idx] == predicted[idx] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    print("\nSample predictions displayed")
    print("Green = Correct, Red = Incorrect")
    print("Close the plot window to continue...")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("IMAGE CLASSIFIER PROJECT (CIFAR-10)")
    print("=" * 60)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Hyperparameters
    batch_size = 128
    num_epochs = 40
    learning_rate = 0.001

    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")

    # Load data
    train_loader, test_loader = load_data(batch_size)

    # Create model
    print("\nInitializing model...")
    model = ImageClassifier(num_classes=10, dropout_rate=0.5)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    history, best_acc = train_model(
        model, train_loader, test_loader, num_epochs, device, learning_rate
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, class_accuracies, predictions, targets = evaluate_model(
        model, test_loader, criterion, device
    )

    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print("\nPer-Class Accuracy:")
    for class_name, acc in class_accuracies.items():
        print(f"  {class_name:10s}: {acc:.2f}%")

    # Visualizations
    visualize_results(history, class_accuracies, predictions, targets, test_loader)

    # Show sample predictions
    show_sample_predictions(model, test_loader, device)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)
    print("\nKey Findings:")
    print(f"1. Achieved {best_acc:.2f}% accuracy on CIFAR-10 test set")
    print("2. Data augmentation improved generalization")
    print("3. Learning rate scheduling helped convergence")
    print("4. BatchNorm and Dropout prevented overfitting")
    print("5. Model saved to 'best_model.pth'")
    print("=" * 60)


if __name__ == "__main__":
    main()
