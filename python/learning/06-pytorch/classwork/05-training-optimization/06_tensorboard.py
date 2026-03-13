"""
TensorBoard Integration in PyTorch
=================================
Demonstrates using TensorBoard to visualize training progress.

TensorBoard Features:
- Track loss and metrics over time
- Visualize model architecture
- Display images and embeddings
- Monitor gradients and weights
- Compare multiple runs

To view TensorBoard:
1. Run this script to generate logs
2. In terminal: tensorboard --logdir=runs
3. Open browser to: http://localhost:6006
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class ConvNet(nn.Module):
    """Simple CNN for demonstration."""
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def generate_mnist_like_data(num_samples=1000, num_classes=10):
    """Generate synthetic MNIST-like images (28x28)."""
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, num_classes, (num_samples,))
    return images, labels


def train_with_tensorboard(writer, run_name, learning_rate=0.001):
    """Train model and log everything to TensorBoard."""
    print(f"\nTraining: {run_name}")
    print("-" * 60)

    # Model setup
    model = ConvNet()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Generate data
    X_train, y_train = generate_mnist_like_data(num_samples=800)
    X_val, y_val = generate_mnist_like_data(num_samples=200)

    # Log model graph
    writer.add_graph(model, X_train[:1])

    # Training parameters
    num_epochs = 30
    batch_size = 32

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        indices = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            batch_indices = indices[i:min(i+batch_size, X_train.size(0))]
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_loss = epoch_loss / (X_train.size(0) // batch_size)
        train_acc = 100.0 * correct / total

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

            _, predicted = torch.max(val_outputs, 1)
            val_acc = 100.0 * (predicted == y_val).sum().item() / y_val.size(0)

        # Log scalars
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss.item(), epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Log histograms of weights and gradients every 10 epochs
        if (epoch + 1) % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Weights/{name}', param.data, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        # Log sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Log a few validation images with predictions
            sample_images = X_val[:8]
            with torch.no_grad():
                sample_outputs = model(sample_images)
                _, sample_preds = torch.max(sample_outputs, 1)

            # Create grid of images
            img_grid = torch.cat([img for img in sample_images], dim=2)
            writer.add_image(f'Predictions/Epoch_{epoch+1}', img_grid, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%")

    return train_loss, val_acc


def compare_hyperparameters():
    """Compare different hyperparameters using TensorBoard."""
    print("\n" + "="*60)
    print("Comparing Multiple Training Runs")
    print("="*60)
    print("\nThis will create logs for TensorBoard comparison")

    # Create parent log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/comparison_{timestamp}'

    # Different learning rates
    learning_rates = [0.0001, 0.001, 0.01]

    results = []

    for lr in learning_rates:
        run_name = f'lr_{lr}'
        writer = SummaryWriter(f'{log_dir}/{run_name}')

        # Add hyperparameters to TensorBoard
        writer.add_text('Hyperparameters', f'Learning Rate: {lr}')

        # Train
        train_loss, val_acc = train_with_tensorboard(writer, run_name, learning_rate=lr)

        # Log final metrics as hyperparameters
        writer.add_hparams(
            {'lr': lr},
            {'hparam/final_loss': train_loss, 'hparam/final_accuracy': val_acc}
        )

        writer.close()
        results.append((lr, train_loss, val_acc))

    print("\n" + "="*60)
    print("TRAINING COMPLETE - Results Summary")
    print("="*60)
    for lr, loss, acc in results:
        print(f"LR={lr:.4f}: Final Loss={loss:.4f}, Val Acc={acc:.2f}%")

    print("\n" + "="*60)
    print("TO VIEW RESULTS IN TENSORBOARD:")
    print("="*60)
    print(f"1. Run: tensorboard --logdir={log_dir}")
    print("2. Open browser to: http://localhost:6006")
    print("3. Compare runs in Scalars, Images, and Graphs tabs")
    print("="*60)

    return log_dir


def demonstrate_advanced_features():
    """Demonstrate advanced TensorBoard features."""
    print("\n" + "="*60)
    print("Advanced TensorBoard Features")
    print("="*60)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/advanced_{timestamp}')

    # 1. Text logging
    writer.add_text('Experiment/Description',
                   'Demonstrating advanced TensorBoard features')
    writer.add_text('Experiment/Config',
                   'Model: ConvNet, Optimizer: Adam, Batch Size: 32')

    # 2. Custom scalars
    for i in range(100):
        # Log multiple related metrics together
        writer.add_scalars('Metrics/train_vs_val', {
            'train_loss': np.sin(i/10.0) + 2,
            'val_loss': np.sin(i/10.0) + 2.2,
        }, i)

    # 3. Embedding visualization (for high-dimensional data)
    features = torch.randn(100, 50)  # 100 samples, 50 features
    labels = torch.randint(0, 10, (100,))
    writer.add_embedding(features, metadata=labels.tolist(), tag='feature_embedding')

    # 4. PR curve (Precision-Recall)
    num_samples = 100
    predictions = torch.rand(num_samples)
    targets = torch.randint(0, 2, (num_samples,))
    writer.add_pr_curve('pr_curve', targets, predictions, 0)

    # 5. Custom matplotlib figures
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label='sin(x)')
    ax.plot(x, np.cos(x), label='cos(x)')
    ax.legend()
    ax.set_title('Custom Plot')
    writer.add_figure('custom_plot', fig, 0)
    plt.close(fig)

    writer.close()

    print("\nAdvanced features logged to TensorBoard!")
    print("Features demonstrated:")
    print("  - Text logging")
    print("  - Grouped scalars")
    print("  - Embedding visualization")
    print("  - PR curves")
    print("  - Custom matplotlib figures")


def demonstrate_basic_usage():
    """Demonstrate basic TensorBoard usage."""
    print("\n" + "="*60)
    print("Basic TensorBoard Usage")
    print("="*60)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/basic_{timestamp}')

    # Simple scalar logging
    for epoch in range(50):
        # Simulate training metrics
        train_loss = 2.0 - 1.8 * (epoch / 50.0) + 0.1 * np.random.random()
        val_loss = 2.0 - 1.6 * (epoch / 50.0) + 0.15 * np.random.random()
        accuracy = 30 + 65 * (epoch / 50.0) + 3 * np.random.random()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy', accuracy, epoch)

    # Log some images
    dummy_images = torch.randn(16, 1, 28, 28)
    img_grid = torch.cat([img for img in dummy_images[:8]], dim=2)
    writer.add_image('Sample_Images', img_grid, 0)

    writer.close()

    print("\nBasic metrics logged!")
    print(f"Run: tensorboard --logdir=runs/basic_{timestamp}")


def main():
    """Main demonstration function."""
    print("="*60)
    print("TENSORBOARD INTEGRATION IN PYTORCH")
    print("="*60)

    # Basic usage
    demonstrate_basic_usage()

    # Compare hyperparameters
    log_dir = compare_hyperparameters()

    # Advanced features
    demonstrate_advanced_features()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("1. TensorBoard visualizes training in real-time")
    print("2. Log scalars: loss, accuracy, learning rate")
    print("3. Log images: input data, predictions, activations")
    print("4. Log histograms: weights and gradients")
    print("5. Log model graph for architecture visualization")
    print("6. Compare multiple runs with different hyperparameters")
    print("7. Use SummaryWriter for all logging operations")
    print("8. Essential for debugging and monitoring training")
    print("="*60)

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("Run TensorBoard to view the logged data:")
    print(f"  tensorboard --logdir=runs")
    print("\nThen open your browser to:")
    print("  http://localhost:6006")
    print("="*60)


if __name__ == "__main__":
    main()
