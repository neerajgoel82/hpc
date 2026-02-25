"""
Model Checkpointing in PyTorch
=============================
Demonstrates saving and loading model checkpoints during training.

Why Checkpointing?
- Save progress during long training runs
- Resume training after interruption
- Save best model based on validation performance
- Experiment with different training stages

What to Save:
- Model state_dict (weights and biases)
- Optimizer state_dict (for resuming)
- Epoch number
- Loss and metrics
- Random states (for reproducibility)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from pathlib import Path


class SimpleNet(nn.Module):
    """Simple neural network for demonstration."""
    def __init__(self, input_size=20, hidden_size=64, num_classes=3):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save a complete checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'random_state': torch.get_rng_state()
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load a checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        Dictionary with epoch, loss, and accuracy
    """
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore random state for reproducibility
    torch.set_rng_state(checkpoint['random_state'])

    print(f"Checkpoint loaded: {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    print(f"  Accuracy: {checkpoint['accuracy']:.2f}%")

    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'accuracy': checkpoint['accuracy']
    }


def save_best_model(model, filepath):
    """Save only the model weights (lightweight)."""
    torch.save(model.state_dict(), filepath)
    print(f"Best model saved: {filepath}")


def load_model_weights(filepath, model):
    """Load only model weights (for inference)."""
    model.load_state_dict(torch.load(filepath))
    print(f"Model weights loaded: {filepath}")


def generate_data(num_samples=1000, input_size=20, num_classes=3):
    """Generate synthetic data."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def train_with_checkpointing(checkpoint_dir='checkpoints', resume_from=None):
    """Train model with periodic checkpointing."""
    print("\n" + "="*60)
    print("Training with Checkpointing")
    print("="*60)

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(exist_ok=True)

    # Setup
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Generate data
    X_train, y_train = generate_data(num_samples=800)
    X_val, y_val = generate_data(num_samples=200)

    # Training parameters
    start_epoch = 0
    num_epochs = 50
    batch_size = 32
    best_val_acc = 0.0

    train_losses = []
    val_losses = []
    val_accuracies = []

    # Resume from checkpoint if specified
    if resume_from is not None and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint_info = load_checkpoint(resume_from, model, optimizer)
        start_epoch = checkpoint_info['epoch'] + 1
        best_val_acc = checkpoint_info['accuracy']
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0

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

        avg_train_loss = epoch_loss / (X_train.size(0) // batch_size)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

            _, predicted = torch.max(val_outputs, 1)
            val_acc = 100.0 * (predicted == y_val).sum().item() / y_val.size(0)
            val_accuracies.append(val_acc)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss.item(),
                          val_acc, checkpoint_path)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_best_model(model, best_model_path)

    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
    save_checkpoint(model, optimizer, num_epochs-1, val_losses[-1],
                   val_accuracies[-1], final_checkpoint_path)

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    return train_losses, val_losses, val_accuracies


def demonstrate_checkpoint_loading():
    """Demonstrate loading different types of checkpoints."""
    print("\n" + "="*60)
    print("Demonstrating Checkpoint Loading")
    print("="*60)

    checkpoint_dir = 'checkpoints'

    # Create and train a model
    print("\n1. Training initial model...")
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    X_train, y_train = generate_data(num_samples=500)

    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'demo_checkpoint.pth')
    save_checkpoint(model, optimizer, 5, loss.item(), 85.5, checkpoint_path)

    # Load full checkpoint
    print("\n2. Loading full checkpoint (for resuming training)...")
    new_model = SimpleNet()
    new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
    load_checkpoint(checkpoint_path, new_model, new_optimizer)

    # Save and load model weights only
    print("\n3. Saving/loading model weights only (for inference)...")
    weights_path = os.path.join(checkpoint_dir, 'model_weights.pth')
    save_best_model(model, weights_path)

    inference_model = SimpleNet()
    load_model_weights(weights_path, inference_model)


def demonstrate_interrupted_training():
    """Demonstrate resuming training after interruption."""
    print("\n" + "="*60)
    print("Simulating Interrupted Training")
    print("="*60)

    checkpoint_dir = 'checkpoints_interrupted'
    Path(checkpoint_dir).mkdir(exist_ok=True)

    # First training run (simulate interruption at epoch 20)
    print("\n1. Starting initial training (will 'interrupt' at epoch 20)...")

    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    X_train, y_train = generate_data(num_samples=500)
    X_val, y_val = generate_data(num_samples=100)

    # Train for 20 epochs
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                _, predicted = torch.max(val_outputs, 1)
                val_acc = 100.0 * (predicted == y_val).sum().item() / y_val.size(0)
                print(f"Epoch {epoch+1}, Val Acc: {val_acc:.2f}%")

    # Save checkpoint at "interruption point"
    interrupt_checkpoint = os.path.join(checkpoint_dir, 'interrupted_at_epoch_20.pth')
    save_checkpoint(model, optimizer, 19, loss.item(), val_acc, interrupt_checkpoint)
    print("\n>>> Training 'interrupted' and checkpoint saved! <<<")

    # Resume training
    print("\n2. Resuming training from checkpoint...")

    resumed_model = SimpleNet()
    resumed_optimizer = optim.Adam(resumed_model.parameters(), lr=0.001)

    checkpoint_info = load_checkpoint(interrupt_checkpoint, resumed_model, resumed_optimizer)

    # Continue training from epoch 21 to 30
    start_epoch = checkpoint_info['epoch'] + 1
    for epoch in range(start_epoch, 30):
        resumed_model.train()
        resumed_optimizer.zero_grad()
        outputs = resumed_model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        resumed_optimizer.step()

        if (epoch + 1) % 10 == 0:
            resumed_model.eval()
            with torch.no_grad():
                val_outputs = resumed_model(X_val)
                _, predicted = torch.max(val_outputs, 1)
                val_acc = 100.0 * (predicted == y_val).sum().item() / y_val.size(0)
                print(f"Epoch {epoch+1}, Val Acc: {val_acc:.2f}%")

    print("\n>>> Training successfully resumed and completed! <<<")


def main():
    """Main demonstration function."""
    print("="*60)
    print("MODEL CHECKPOINTING IN PYTORCH")
    print("="*60)

    # Train with checkpointing
    train_with_checkpointing(checkpoint_dir='checkpoints_demo')

    # Demonstrate loading
    demonstrate_checkpoint_loading()

    # Demonstrate interrupted training
    demonstrate_interrupted_training()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("1. Save checkpoints periodically during training")
    print("2. Save full checkpoint: model + optimizer + epoch + metrics")
    print("3. Save best model based on validation performance")
    print("4. Load checkpoint to resume interrupted training")
    print("5. Save only weights for inference (smaller file)")
    print("6. Include random state for reproducibility")
    print("7. Use meaningful filenames (epoch number, date, metric)")
    print("="*60)


if __name__ == "__main__":
    main()
