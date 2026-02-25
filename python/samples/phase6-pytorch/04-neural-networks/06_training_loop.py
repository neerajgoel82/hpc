"""
Training Loop
Complete training pipeline

This module demonstrates:
- Complete training loop structure
- Training vs validation
- Batch processing
- Loss tracking
- Model checkpointing
- Early stopping

Run: python 06_training_loop.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple


class SimpleClassifier(nn.Module):
    """Simple classifier for demonstration"""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_dummy_dataset(n_samples: int = 1000, input_size: int = 20,
                        num_classes: int = 3) -> Tuple[TensorDataset, TensorDataset]:
    """Create dummy classification dataset"""
    # Training data
    X_train = torch.randn(n_samples, input_size)
    y_train = torch.randint(0, num_classes, (n_samples,))
    train_dataset = TensorDataset(X_train, y_train)

    # Validation data
    X_val = torch.randn(n_samples // 5, input_size)
    y_val = torch.randint(0, num_classes, (n_samples // 5,))
    val_dataset = TensorDataset(X_val, y_val)

    return train_dataset, val_dataset


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
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

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def demonstrate_basic_training_loop():
    """Basic training loop"""
    print("\n1. Basic Training Loop")
    print("-" * 60)

    # Create dataset
    train_dataset, val_dataset = create_dummy_dataset(n_samples=500)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Create model
    model = SimpleClassifier(input_size=20, hidden_size=50, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")


def demonstrate_with_checkpointing():
    """Training with model checkpointing"""
    print("\n2. Training with Checkpointing")
    print("-" * 60)

    train_dataset, val_dataset = create_dummy_dataset(n_samples=500)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = SimpleClassifier(input_size=20, hidden_size=50, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = float('inf')
    checkpoint_path = '/tmp/best_model.pth'

    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"  Saved checkpoint! Best val loss: {best_val_loss:.4f}")


def demonstrate_early_stopping():
    """Training with early stopping"""
    print("\n3. Training with Early Stopping")
    print("-" * 60)

    train_dataset, val_dataset = create_dummy_dataset(n_samples=500)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = SimpleClassifier(input_size=20, hidden_size=50, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(20):  # Max epochs
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  Improvement! Reset patience counter.")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break


def demonstrate_gradient_accumulation():
    """Gradient accumulation for larger effective batch size"""
    print("\n4. Gradient Accumulation")
    print("-" * 60)

    train_dataset, _ = create_dummy_dataset(n_samples=500)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = SimpleClassifier(input_size=20, hidden_size=50, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    accumulation_steps = 4  # Effective batch size = 16 * 4 = 64

    model.train()
    optimizer.zero_grad()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        loss = loss / accumulation_steps  # Normalize loss
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Step {(batch_idx + 1) // accumulation_steps}: "
                  f"Updated weights after {accumulation_steps} batches")

        if batch_idx >= 7:  # Show just a few steps
            break

    print(f"\nEffective batch size: {16 * accumulation_steps}")


def demonstrate_learning_rate_scheduling():
    """Learning rate scheduling"""
    print("\n5. Learning Rate Scheduling")
    print("-" * 60)

    train_dataset, val_dataset = create_dummy_dataset(n_samples=500)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = SimpleClassifier(input_size=20, hidden_size=50, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Higher initial LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(8):
        train_loss, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, _ = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        # Step scheduler
        scheduler.step(val_loss)


def main():
    print("=" * 60)
    print("Training Loop")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    demonstrate_basic_training_loop()
    demonstrate_with_checkpointing()
    demonstrate_early_stopping()
    demonstrate_gradient_accumulation()
    demonstrate_learning_rate_scheduling()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Training loop components:")
    print("1. model.train() / model.eval()")
    print("2. optimizer.zero_grad()")
    print("3. Forward pass")
    print("4. Loss computation")
    print("5. loss.backward()")
    print("6. optimizer.step()")
    print("7. Track metrics and save checkpoints")
    print("=" * 60)


if __name__ == "__main__":
    main()
