"""
LeNet-5 Architecture - Classic CNN Implementation

This module demonstrates:
- LeNet-5 architecture implementation
- First successful CNN architecture (1998)
- Conv -> Pool -> Conv -> Pool -> FC structure
- Training on MNIST dataset
- Model evaluation and visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time


class LeNet5(nn.Module):
    """
    LeNet-5 architecture (Yann LeCun, 1998)

    Architecture:
    Input (1x32x32) -> Conv1 (6 filters, 5x5) -> AvgPool (2x2)
    -> Conv2 (16 filters, 5x5) -> AvgPool (2x2)
    -> Conv3 (120 filters, 5x5) -> FC (84) -> FC (10)
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Layer 1: Conv -> Activation -> Pool
        x = self.conv1(x)
        x = torch.tanh(x)  # Original used tanh
        x = self.pool1(x)

        # Layer 2: Conv -> Activation -> Pool
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.pool2(x)

        # Layer 3: Conv -> Activation
        x = self.conv3(x)
        x = torch.tanh(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)

        return x


class ModernLeNet(nn.Module):
    """
    Modernized LeNet with ReLU and MaxPool

    Improvements:
    - ReLU instead of tanh (faster training)
    - MaxPool instead of AvgPool (better feature extraction)
    - Dropout for regularization
    - Batch normalization
    """

    def __init__(self, num_classes=10):
        super(ModernLeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)

        self.fc1 = nn.Linear(120, 84)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def analyze_architecture():
    """Analyze LeNet architecture and parameters"""
    print("=" * 60)
    print("LeNet-5 Architecture Analysis")
    print("=" * 60)

    model = LeNet5()

    print("\nOriginal LeNet-5 (1998):")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Layer-by-layer analysis
    print("\n" + "-" * 60)
    print("Layer-by-layer output shapes:")
    print("-" * 60)

    x = torch.randn(1, 1, 32, 32)
    print(f"Input: {x.shape}")

    x = model.conv1(x)
    print(f"After Conv1: {x.shape}")
    x = torch.tanh(x)
    x = model.pool1(x)
    print(f"After Pool1: {x.shape}")

    x = model.conv2(x)
    print(f"After Conv2: {x.shape}")
    x = torch.tanh(x)
    x = model.pool2(x)
    print(f"After Pool2: {x.shape}")

    x = model.conv3(x)
    print(f"After Conv3: {x.shape}")
    x = torch.tanh(x)

    x = x.view(x.size(0), -1)
    print(f"After Flatten: {x.shape}")

    x = model.fc1(x)
    print(f"After FC1: {x.shape}")
    x = model.fc2(x)
    print(f"Output: {x.shape}")


def prepare_data(batch_size=64):
    """Prepare MNIST dataset with appropriate transforms"""
    # LeNet expects 32x32 images, MNIST is 28x28
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy


def train_lenet(epochs=5):
    """Train LeNet on MNIST"""
    print("\n" + "=" * 60)
    print("Training LeNet-5 on MNIST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Prepare data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = prepare_data(batch_size=64)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    model = ModernLeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        epoch_time = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch}/{epochs} ({epoch_time:.2f}s):")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    return model, history


def visualize_predictions(model, test_loader, device):
    """Visualize model predictions"""
    model.eval()

    # Get a batch
    data, targets = next(iter(test_loader))
    data, targets = data.to(device), targets.to(device)

    with torch.no_grad():
        outputs = model(data)
        predictions = outputs.argmax(dim=1)

    # Plot first 12 images
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle('LeNet-5 Predictions on MNIST', fontsize=16)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(data):
            break

        img = data[idx].cpu().squeeze()
        true_label = targets[idx].item()
        pred_label = predictions[idx].item()

        ax.imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('/tmp/lenet_predictions.png', dpi=150, bbox_inches='tight')
    print("\nPredictions saved to /tmp/lenet_predictions.png")
    plt.close()


def main():
    """Main demonstration function"""
    print("\n" + "=" * 60)
    print("LENET-5: CLASSIC CNN ARCHITECTURE")
    print("=" * 60)

    # Analyze architecture
    analyze_architecture()

    # Train model (use fewer epochs for quick demo)
    model, history = train_lenet(epochs=3)

    # Visualize predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = prepare_data()
    visualize_predictions(model, test_loader, device)

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. LeNet-5 pioneered the CNN architecture (1998)")
    print("2. Conv-Pool-Conv-Pool-FC structure is still used")
    print("3. Modern improvements: ReLU, MaxPool, BatchNorm, Dropout")
    print("4. Achieves ~99% accuracy on MNIST")
    print("5. Foundation for modern CNNs (AlexNet, VGG, ResNet)")
    print("=" * 60)


if __name__ == "__main__":
    main()
