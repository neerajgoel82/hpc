"""
Neural Network Exercises
Build and train neural networks

Complete these exercises to practice neural network concepts.
Each exercise has a solution provided.

Run: python exercises.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# EXERCISE 1: Create a simple 2-layer neural network
# ============================================================================

def exercise1():
    """
    Create a 2-layer neural network with:
    - Input size: 10
    - Hidden size: 20
    - Output size: 1
    - ReLU activation
    """
    print("\nExercise 1: Simple 2-Layer Network")
    print("-" * 60)

    # YOUR CODE HERE
    class TwoLayerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Test
    model = TwoLayerNet()
    x = torch.randn(5, 10)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print("Expected: 241 parameters (10*20 + 20 + 20*1 + 1)")


# ============================================================================
# EXERCISE 2: Initialize weights with He initialization
# ============================================================================

def exercise2():
    """
    Create a network and initialize all linear layers with He initialization.
    """
    print("\nExercise 2: He Initialization")
    print("-" * 60)

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(50, 100)
            self.fc2 = nn.Linear(100, 50)
            self.fc3 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # YOUR CODE HERE: Initialize with He initialization
    model = Network()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.bias)

    # Test
    print("Weight statistics after He initialization:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}: mean={param.mean().item():.6f}, "
                  f"std={param.std().item():.6f}")


# ============================================================================
# EXERCISE 3: Compare different activation functions
# ============================================================================

def exercise3():
    """
    Create networks with different activation functions and compare their outputs.
    """
    print("\nExercise 3: Activation Functions")
    print("-" * 60)

    class NetworkWithActivation(nn.Module):
        def __init__(self, activation):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
            self.activation = activation

        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.fc2(x)
            return x

    # YOUR CODE HERE: Test ReLU, Tanh, and Sigmoid
    x = torch.randn(3, 10)

    activations = {
        'ReLU': nn.ReLU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'LeakyReLU': nn.LeakyReLU(),
    }

    for name, activation in activations.items():
        model = NetworkWithActivation(activation)
        output = model(x)
        print(f"{name:12s}: Output range [{output.min().item():.2f}, "
              f"{output.max().item():.2f}]")


# ============================================================================
# EXERCISE 4: Implement a custom loss function
# ============================================================================

def exercise4():
    """
    Implement a custom loss function: Mean Absolute Percentage Error (MAPE)
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    """
    print("\nExercise 4: Custom Loss Function")
    print("-" * 60)

    # YOUR CODE HERE
    class MAPELoss(nn.Module):
        def __init__(self, epsilon=1e-8):
            super().__init__()
            self.epsilon = epsilon

        def forward(self, predictions, targets):
            return torch.mean(
                torch.abs((targets - predictions) / (targets + self.epsilon))
            ) * 100

    # Test
    predictions = torch.tensor([9.0, 8.5, 11.0, 7.0])
    targets = torch.tensor([10.0, 9.0, 10.0, 8.0])

    mape_loss = MAPELoss()
    loss = mape_loss(predictions, targets)
    print(f"Predictions: {predictions}")
    print(f"Targets: {targets}")
    print(f"MAPE Loss: {loss.item():.2f}%")


# ============================================================================
# EXERCISE 5: Train a network with different optimizers
# ============================================================================

def exercise5():
    """
    Train the same network with SGD, Adam, and AdamW. Compare convergence.
    """
    print("\nExercise 5: Optimizer Comparison")
    print("-" * 60)

    # Create simple dataset
    X = torch.randn(100, 5)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # YOUR CODE HERE: Train with different optimizers
    optimizers_config = [
        ('SGD', lambda p: optim.SGD(p, lr=0.01)),
        ('Adam', lambda p: optim.Adam(p, lr=0.001)),
        ('AdamW', lambda p: optim.AdamW(p, lr=0.001, weight_decay=0.01)),
    ]

    for name, opt_fn in optimizers_config:
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        optimizer = opt_fn(model.parameters())
        criterion = nn.MSELoss()

        # Train for 10 epochs
        losses = []
        for epoch in range(10):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        print(f"{name:10s}: Initial loss={losses[0]:.4f}, "
              f"Final loss={losses[-1]:.4f}")


# ============================================================================
# EXERCISE 6: Build a complete training loop with validation
# ============================================================================

def exercise6():
    """
    Build a complete training loop with train/validation split.
    """
    print("\nExercise 6: Complete Training Loop")
    print("-" * 60)

    # Create datasets
    X_train = torch.randn(200, 10)
    y_train = torch.randint(0, 3, (200,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    X_val = torch.randn(50, 10)
    y_val = torch.randint(0, 3, (50,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # YOUR CODE HERE: Implement training loop
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 3)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        # Training
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / len(val_dataset)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")


# ============================================================================
# EXERCISE 7: Implement a network with dropout
# ============================================================================

def exercise7():
    """
    Create a network with dropout and compare training vs evaluation mode.
    """
    print("\nExercise 7: Dropout")
    print("-" * 60)

    # YOUR CODE HERE
    class NetworkWithDropout(nn.Module):
        def __init__(self, dropout_rate=0.5):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(50, 2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = NetworkWithDropout(dropout_rate=0.5)
    x = torch.randn(5, 10)

    # Training mode
    model.train()
    out1 = model(x)
    out2 = model(x)
    print(f"Training mode - outputs differ: {not torch.allclose(out1, out2)}")
    print(f"  Output 1: {out1[0]}")
    print(f"  Output 2: {out2[0]}")

    # Eval mode
    model.eval()
    out3 = model(x)
    out4 = model(x)
    print(f"\nEval mode - outputs identical: {torch.allclose(out3, out4)}")
    print(f"  Output 3: {out3[0]}")
    print(f"  Output 4: {out4[0]}")


# ============================================================================
# EXERCISE 8: Build MLP with batch normalization
# ============================================================================

def exercise8():
    """
    Create a network with batch normalization and compare with/without.
    """
    print("\nExercise 8: Batch Normalization")
    print("-" * 60)

    # YOUR CODE HERE
    class MLPWithBatchNorm(nn.Module):
        def __init__(self, use_bn=True):
            super().__init__()
            self.fc1 = nn.Linear(20, 50)
            self.bn1 = nn.BatchNorm1d(50) if use_bn else nn.Identity()
            self.fc2 = nn.Linear(50, 10)
            self.bn2 = nn.BatchNorm1d(10) if use_bn else nn.Identity()

        def forward(self, x):
            x = self.bn1(F.relu(self.fc1(x)))
            x = self.bn2(F.relu(self.fc2(x)))
            return x

    x = torch.randn(32, 20)

    # Without batch norm
    model_no_bn = MLPWithBatchNorm(use_bn=False)
    out_no_bn = model_no_bn(x)
    print(f"Without BatchNorm:")
    print(f"  Mean: {out_no_bn.mean().item():.4f}")
    print(f"  Std:  {out_no_bn.std().item():.4f}")

    # With batch norm
    model_with_bn = MLPWithBatchNorm(use_bn=True)
    model_with_bn.train()
    out_with_bn = model_with_bn(x)
    print(f"\nWith BatchNorm:")
    print(f"  Mean: {out_with_bn.mean().item():.4f}")
    print(f"  Std:  {out_with_bn.std().item():.4f}")


# ============================================================================
# EXERCISE 9: Implement gradient clipping
# ============================================================================

def exercise9():
    """
    Train a network with gradient clipping to prevent exploding gradients.
    """
    print("\nExercise 9: Gradient Clipping")
    print("-" * 60)

    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    X = torch.randn(20, 5)
    y = torch.randn(20, 1) * 100  # Large targets to create large gradients

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # YOUR CODE HERE: Implement gradient clipping
    max_norm = 1.0

    for epoch in range(3):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()

        # Compute gradient norm before clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Compute gradient norm after clipping
        clipped_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                clipped_norm += p.grad.data.norm(2).item() ** 2
        clipped_norm = clipped_norm ** 0.5

        optimizer.step()

        print(f"Epoch {epoch+1}: Grad norm before={total_norm:.4f}, "
              f"after={clipped_norm:.4f}, loss={loss.item():.4f}")


# ============================================================================
# EXERCISE 10: Build a classifier for MNIST
# ============================================================================

def exercise10():
    """
    Build and train a simple MLP for MNIST digit classification.
    """
    print("\nExercise 10: MNIST Classification")
    print("-" * 60)

    try:
        from torchvision import datasets, transforms

        # YOUR CODE HERE
        class MNISTClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 10)
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x

        # Load data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # Create and train model
        model = MNISTClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Training for 2 epochs...")
        for epoch in range(2):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                if batch_idx >= 50:  # Train on subset for speed
                    break

            avg_loss = total_loss / min(50, len(train_loader))
            accuracy = 100.0 * correct / total
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

    except ImportError:
        print("torchvision not installed. Skipping MNIST exercise.")
        print("Install with: pip install torchvision")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Neural Network Exercises")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    exercises = [
        exercise1, exercise2, exercise3, exercise4, exercise5,
        exercise6, exercise7, exercise8, exercise9, exercise10
    ]

    for i, exercise in enumerate(exercises, 1):
        try:
            exercise()
        except Exception as e:
            print(f"\nExercise {i} failed with error: {e}")

    print("\n" + "=" * 60)
    print("All Exercises Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
