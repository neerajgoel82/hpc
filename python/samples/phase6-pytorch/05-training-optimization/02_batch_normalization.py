"""
Batch Normalization in PyTorch
=============================
Demonstrates batch normalization and its benefits for training.

Benefits of Batch Normalization:
- Faster training (higher learning rates)
- Less sensitive to initialization
- Acts as regularization (reduces overfitting)
- Reduces internal covariate shift

How it works:
1. Normalize inputs: (x - mean) / sqrt(variance + epsilon)
2. Scale and shift: gamma * normalized + beta
3. Use running statistics during inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time


# Network WITHOUT Batch Normalization
class NetworkWithoutBN(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], num_classes=10):
        super(NetworkWithoutBN, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Network WITH Batch Normalization
class NetworkWithBN(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], num_classes=10):
        super(NetworkWithBN, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Add BatchNorm
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def generate_mnist_like_data(num_samples=5000, input_size=784, num_classes=10):
    """Generate synthetic data similar to MNIST."""
    X = torch.randn(num_samples, input_size) * 0.5  # Mean ~0, std ~0.5
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def train_model(model, optimizer, X_train, y_train, X_val, y_val,
                num_epochs=30, batch_size=64):
    """Train model and track metrics."""
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    val_accuracies = []

    num_samples = X_train.size(0)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0

        indices = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i+batch_size, num_samples)]
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / (num_samples // batch_size)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

            # Calculate accuracy
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val).float().mean().item() * 100
            val_accuracies.append(accuracy)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {accuracy:.2f}%")

    return train_losses, val_losses, val_accuracies


def compare_with_without_bn():
    """Compare training with and without batch normalization."""
    print("\n" + "="*60)
    print("Comparing Training: With vs Without Batch Normalization")
    print("="*60)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Generate data
    print("\nGenerating synthetic data...")
    X_train, y_train = generate_mnist_like_data(num_samples=5000)
    X_val, y_val = generate_mnist_like_data(num_samples=1000)

    # Train WITHOUT BatchNorm
    print("\n" + "-"*60)
    print("Training WITHOUT Batch Normalization")
    print("-"*60)

    model_no_bn = NetworkWithoutBN()
    optimizer_no_bn = optim.SGD(model_no_bn.parameters(), lr=0.01, momentum=0.9)

    start_time = time.time()
    train_loss_no_bn, val_loss_no_bn, val_acc_no_bn = train_model(
        model_no_bn, optimizer_no_bn, X_train, y_train, X_val, y_val
    )
    time_no_bn = time.time() - start_time

    # Train WITH BatchNorm
    print("\n" + "-"*60)
    print("Training WITH Batch Normalization")
    print("-"*60)

    torch.manual_seed(42)  # Reset seed for fair comparison
    model_with_bn = NetworkWithBN()
    optimizer_with_bn = optim.SGD(model_with_bn.parameters(), lr=0.01, momentum=0.9)

    start_time = time.time()
    train_loss_with_bn, val_loss_with_bn, val_acc_with_bn = train_model(
        model_with_bn, optimizer_with_bn, X_train, y_train, X_val, y_val
    )
    time_with_bn = time.time() - start_time

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Without BN - Final Val Acc: {val_acc_no_bn[-1]:.2f}%, Time: {time_no_bn:.2f}s")
    print(f"With BN    - Final Val Acc: {val_acc_with_bn[-1]:.2f}%, Time: {time_with_bn:.2f}s")
    print(f"\nImprovement: {val_acc_with_bn[-1] - val_acc_no_bn[-1]:.2f}% accuracy")

    return (train_loss_no_bn, val_loss_no_bn, val_acc_no_bn,
            train_loss_with_bn, val_loss_with_bn, val_acc_with_bn)


def visualize_comparison(results):
    """Visualize training comparison."""
    train_loss_no_bn, val_loss_no_bn, val_acc_no_bn, \
    train_loss_with_bn, val_loss_with_bn, val_acc_with_bn = results

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(train_loss_no_bn) + 1)

    # Training Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_loss_no_bn, 'b-', label='Without BN', linewidth=2)
    ax1.plot(epochs, train_loss_with_bn, 'r-', label='With BN', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Validation Loss
    ax2 = axes[1]
    ax2.plot(epochs, val_loss_no_bn, 'b-', label='Without BN', linewidth=2)
    ax2.plot(epochs, val_loss_with_bn, 'r-', label='With BN', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Validation Accuracy
    ax3 = axes[2]
    ax3.plot(epochs, val_acc_no_bn, 'b-', label='Without BN', linewidth=2)
    ax3.plot(epochs, val_acc_with_bn, 'r-', label='With BN', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Accuracy (%)')
    ax3.set_title('Validation Accuracy Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('batch_normalization_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'batch_normalization_comparison.png'")
    plt.show()


def demonstrate_bn_behavior():
    """Demonstrate how batch normalization behaves."""
    print("\n" + "="*60)
    print("Batch Normalization Behavior Demo")
    print("="*60)

    # Create a simple BatchNorm layer
    bn = nn.BatchNorm1d(num_features=3)

    # Create input with different scales
    x = torch.tensor([[1.0, 100.0, 0.1],
                      [2.0, 200.0, 0.2],
                      [3.0, 300.0, 0.3],
                      [4.0, 400.0, 0.4]])

    print("\nInput (features with different scales):")
    print(x)
    print(f"Mean per feature: {x.mean(dim=0)}")
    print(f"Std per feature: {x.std(dim=0)}")

    # Apply batch normalization
    bn.eval()  # Use evaluation mode (uses running stats)
    with torch.no_grad():
        # First pass to initialize running stats
        _ = bn(x)

    bn.train()  # Switch to training mode
    output = bn(x)

    print("\nOutput after BatchNorm:")
    print(output)
    print(f"Mean per feature: {output.mean(dim=0)}")
    print(f"Std per feature: {output.std(dim=0)}")
    print("\nNotice: All features now have similar scale (mean~0, std~1)")


def main():
    """Main demonstration function."""
    print("="*60)
    print("BATCH NORMALIZATION IN PYTORCH")
    print("="*60)

    # Demonstrate BN behavior
    demonstrate_bn_behavior()

    # Compare with and without BN
    results = compare_with_without_bn()

    # Visualize results
    visualize_comparison(results)

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("1. BatchNorm normalizes layer inputs to mean=0, std=1")
    print("2. Allows higher learning rates -> faster convergence")
    print("3. Reduces sensitivity to weight initialization")
    print("4. Acts as regularization (slight dropout effect)")
    print("5. Use after Linear/Conv layers, before activation")
    print("6. Different behavior in train vs eval mode")
    print("7. Typically improves accuracy by 2-5%")
    print("="*60)


if __name__ == "__main__":
    main()
