"""
MLP Classification
Multi-layer perceptron for classification on Fashion-MNIST

This module demonstrates:
- Complete MLP implementation
- Training on real dataset (Fashion-MNIST)
- Data loading and preprocessing
- Evaluation metrics
- Confusion matrix
- Model performance analysis

Run: python 07_mlp_classification.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time


class MLP(nn.Module):
    """Multi-Layer Perceptron for image classification"""

    def __init__(self, input_size: int = 784, hidden_sizes: list = None,
                 num_classes: int = 10, dropout_rate: float = 0.2):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten image
        x = x.view(x.size(0), -1)
        return self.network(x)


def get_fashion_mnist_loaders(batch_size: int = 128):
    """Load Fashion-MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def compute_confusion_matrix(model, dataloader, device, num_classes: int = 10):
    """Compute confusion matrix"""
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)

            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix


def demonstrate_mlp_training():
    """Train MLP on Fashion-MNIST"""
    print("\n1. MLP Training on Fashion-MNIST")
    print("-" * 60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading Fashion-MNIST dataset...")
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=128)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    model = MLP(input_size=784, hidden_sizes=[256, 128], num_classes=10,
                dropout_rate=0.2)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 60)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")

    return model, test_loader, device


def demonstrate_model_evaluation(model, test_loader, device):
    """Detailed model evaluation"""
    print("\n2. Model Evaluation")
    print("-" * 60)

    # Class names
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    # Compute confusion matrix
    print("\nComputing confusion matrix...")
    cm = compute_confusion_matrix(model, test_loader, device, num_classes=10)

    print("\nConfusion Matrix:")
    print("(Rows: True labels, Columns: Predicted labels)")
    print(cm)

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_correct = cm[i, i].item()
        class_total = cm[i].sum().item()
        class_acc = 100.0 * class_correct / class_total if class_total > 0 else 0
        print(f"  {class_name:15s}: {class_acc:.2f}% ({class_correct}/{class_total})")


def demonstrate_model_comparison():
    """Compare different MLP architectures"""
    print("\n3. Architecture Comparison")
    print("-" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=128)

    architectures = [
        ('Small (128)', [128]),
        ('Medium (256, 128)', [256, 128]),
        ('Large (512, 256, 128)', [512, 256, 128]),
    ]

    print("\nTraining different architectures for 3 epochs each...")

    for name, hidden_sizes in architectures:
        print(f"\n{name}:")
        model = MLP(input_size=784, hidden_sizes=hidden_sizes,
                    num_classes=10, dropout_rate=0.2)
        model.to(device)

        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train for 3 epochs
        for epoch in range(3):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )

        # Final test performance
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"  Final Test Accuracy: {test_acc:.2f}%")


def demonstrate_inference():
    """Inference on new samples"""
    print("\n4. Inference")
    print("-" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_fashion_mnist_loaders(batch_size=1)

    # Load a trained model (use small model for demo)
    model = MLP(input_size=784, hidden_sizes=[128], num_classes=10)
    model.to(device)

    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    # Quick training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader, _ = get_fashion_mnist_loaders(batch_size=128)
    train_epoch(model, train_loader, criterion, optimizer, device)

    # Inference
    model.eval()
    print("\nSample predictions:")

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= 5:  # Show 5 examples
                break

            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(1).item()
            confidence = probabilities[0, predicted_class].item()

            print(f"  True: {class_names[target.item()]:15s} | "
                  f"Predicted: {class_names[predicted_class]:15s} | "
                  f"Confidence: {confidence:.2%}")


def main():
    print("=" * 60)
    print("MLP Classification on Fashion-MNIST")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Main training
    model, test_loader, device = demonstrate_mlp_training()

    # Evaluation
    demonstrate_model_evaluation(model, test_loader, device)

    # Compare architectures
    demonstrate_model_comparison()

    # Inference
    demonstrate_inference()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Fashion-MNIST Classification with MLP:")
    print("- 10 classes (clothing items)")
    print("- 28x28 grayscale images (784 features)")
    print("- Typical accuracy: 85-90% with simple MLP")
    print("- Can achieve 90%+ with deeper networks/regularization")
    print("=" * 60)


if __name__ == "__main__":
    main()
