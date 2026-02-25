"""
Early Stopping in PyTorch
========================
Demonstrates early stopping to prevent overfitting.

What is Early Stopping?
- Stop training when validation performance stops improving
- Prevents overfitting by not training too long
- Saves best model based on validation metric

Key parameters:
- patience: How many epochs to wait for improvement
- min_delta: Minimum change to qualify as improvement
- mode: 'min' for loss, 'max' for accuracy

Benefits:
- Prevents overfitting automatically
- Saves training time
- No need to guess optimal number of epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""

    def __init__(self, patience=5, min_delta=0, mode='min', verbose=True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, score, model):
        """Check if should stop training."""
        # Convert to loss-like metric (lower is better)
        if self.mode == 'max':
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"Initial best score: {score:.4f}")
        elif score > self.best_score - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
        else:
            if self.verbose:
                print(f"Score improved: {self.best_score:.4f} -> {score:.4f}")
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_model(self, model):
        """Load the best model weights."""
        if self.best_model is not None:
            model.load_state_dict(self.best_model)


class SimpleNet(nn.Module):
    """Simple network that can easily overfit."""
    def __init__(self, input_size=20, hidden_size=100, num_classes=3):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def generate_data(num_train=500, num_val=200, input_size=20, num_classes=3):
    """Generate synthetic data that's easy to overfit."""
    torch.manual_seed(42)
    X_train = torch.randn(num_train, input_size)
    y_train = torch.randint(0, num_classes, (num_train,))

    X_val = torch.randn(num_val, input_size)
    y_val = torch.randint(0, num_classes, (num_val,))

    return X_train, y_train, X_val, y_val


def train_with_early_stopping(model, optimizer, X_train, y_train, X_val, y_val,
                               early_stopper, max_epochs=100, batch_size=32):
    """Train with early stopping."""
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    stopped_epoch = max_epochs

    num_samples = X_train.size(0)

    for epoch in range(max_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

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
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_train_loss = epoch_loss / (num_samples // batch_size)
        train_acc = 100.0 * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

            _, predicted = torch.max(val_outputs, 1)
            val_acc = 100.0 * (predicted == y_val).sum().item() / y_val.size(0)
            val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch [{epoch+1}/{max_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%")

        # Check early stopping
        early_stopper(val_loss.item(), model)
        if early_stopper.early_stop:
            stopped_epoch = epoch + 1
            print(f"\nStopped at epoch {stopped_epoch}")
            break

    # Load best model
    early_stopper.load_best_model(model)

    return train_losses, val_losses, train_accs, val_accs, stopped_epoch


def train_without_early_stopping(model, optimizer, X_train, y_train, X_val, y_val,
                                 max_epochs=100, batch_size=32):
    """Train without early stopping."""
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    num_samples = X_train.size(0)

    for epoch in range(max_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

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
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_train_loss = epoch_loss / (num_samples // batch_size)
        train_acc = 100.0 * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

            _, predicted = torch.max(val_outputs, 1)
            val_acc = 100.0 * (predicted == y_val).sum().item() / y_val.size(0)
            val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch [{epoch+1}/{max_epochs}], "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%")

    return train_losses, val_losses, train_accs, val_accs


def compare_with_without_early_stopping():
    """Compare training with and without early stopping."""
    print("\n" + "="*60)
    print("Comparing Training: With vs Without Early Stopping")
    print("="*60)

    # Generate data
    X_train, y_train, X_val, y_val = generate_data()

    # Train WITH early stopping
    print("\n" + "-"*60)
    print("Training WITH Early Stopping (patience=7)")
    print("-"*60)

    torch.manual_seed(42)
    model_with_es = SimpleNet()
    optimizer_with_es = optim.Adam(model_with_es.parameters(), lr=0.001)
    early_stopper = EarlyStopping(patience=7, verbose=True)

    train_loss_with, val_loss_with, train_acc_with, val_acc_with, stopped_epoch = \
        train_with_early_stopping(model_with_es, optimizer_with_es,
                                 X_train, y_train, X_val, y_val, early_stopper)

    # Train WITHOUT early stopping
    print("\n" + "-"*60)
    print("Training WITHOUT Early Stopping (full 100 epochs)")
    print("-"*60)

    torch.manual_seed(42)
    model_without_es = SimpleNet()
    optimizer_without_es = optim.Adam(model_without_es.parameters(), lr=0.001)

    train_loss_without, val_loss_without, train_acc_without, val_acc_without = \
        train_without_early_stopping(model_without_es, optimizer_without_es,
                                    X_train, y_train, X_val, y_val)

    # Results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"WITH Early Stopping:")
    print(f"  - Stopped at epoch: {stopped_epoch}")
    print(f"  - Best val accuracy: {max(val_acc_with):.2f}%")
    print(f"  - Final val loss: {val_loss_with[-1]:.4f}")

    print(f"\nWITHOUT Early Stopping:")
    print(f"  - Trained for: 100 epochs")
    print(f"  - Best val accuracy: {max(val_acc_without):.2f}%")
    print(f"  - Final val loss: {val_loss_without[-1]:.4f}")

    print(f"\nTime saved: {100 - stopped_epoch} epochs")

    return (train_loss_with, val_loss_with, val_acc_with,
            train_loss_without, val_loss_without, val_acc_without, stopped_epoch)


def visualize_comparison(results):
    """Visualize training with and without early stopping."""
    train_loss_with, val_loss_with, val_acc_with, \
    train_loss_without, val_loss_without, val_acc_without, stopped_epoch = results

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Validation Loss
    ax1 = axes[0]
    epochs_with = range(1, len(val_loss_with) + 1)
    epochs_without = range(1, len(val_loss_without) + 1)

    ax1.plot(epochs_without, val_loss_without, 'b-', alpha=0.6,
             label='Without Early Stop', linewidth=2)
    ax1.plot(epochs_with, val_loss_with, 'r-', label='With Early Stop', linewidth=2)
    ax1.axvline(x=stopped_epoch, color='r', linestyle='--', alpha=0.5,
                label=f'Stopped at epoch {stopped_epoch}')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss: Early Stopping Prevents Overfitting')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Validation Accuracy
    ax2 = axes[1]
    ax2.plot(epochs_without, val_acc_without, 'b-', alpha=0.6,
             label='Without Early Stop', linewidth=2)
    ax2.plot(epochs_with, val_acc_with, 'r-', label='With Early Stop', linewidth=2)
    ax2.axvline(x=stopped_epoch, color='r', linestyle='--', alpha=0.5,
                label=f'Stopped at epoch {stopped_epoch}')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('early_stopping_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'early_stopping_comparison.png'")
    plt.show()


def main():
    """Main demonstration function."""
    print("="*60)
    print("EARLY STOPPING IN PYTORCH")
    print("="*60)

    results = compare_with_without_early_stopping()
    visualize_comparison(results)

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("1. Early stopping prevents overfitting automatically")
    print("2. Monitors validation metric, stops when no improvement")
    print("3. patience: how many epochs to wait (typically 5-10)")
    print("4. Saves best model, not final model")
    print("5. Saves training time and computational resources")
    print("6. Essential when you don't know optimal epoch count")
    print("7. Use with validation set, not training set")
    print("="*60)


if __name__ == "__main__":
    main()
