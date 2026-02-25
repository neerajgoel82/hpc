"""
Training Optimization Exercises
==============================
Hands-on exercises to practice training optimization techniques.

Complete these exercises to master:
1. Learning rate scheduling strategies
2. Batch normalization implementation
3. Data augmentation pipelines
4. Early stopping integration
5. Model checkpointing
6. Combined optimization techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision import transforms
import matplotlib.pyplot as plt
import copy


def generate_classification_data(num_samples=1000, input_size=20, num_classes=4):
    """Generate synthetic classification data."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def generate_image_data(num_samples=500, channels=3, height=32, width=32, num_classes=5):
    """Generate synthetic image data."""
    images = torch.randn(num_samples, channels, height, width)
    labels = torch.randint(0, num_classes, (num_samples,))
    return images, labels


# ============================================================================
# EXERCISE 1: Implement Cosine Annealing with Warm Restarts
# ============================================================================

def exercise1_cosine_annealing_warm_restarts():
    """
    Exercise 1: Implement training with CosineAnnealingWarmRestarts scheduler.

    Tasks:
    1. Create a simple neural network
    2. Use CosineAnnealingWarmRestarts scheduler
    3. Train for 50 epochs
    4. Plot learning rate schedule
    5. Compare with constant learning rate

    Expected behavior:
    - Learning rate should cycle (restart) periodically
    - Each cycle should improve model performance
    """
    print("\n" + "="*60)
    print("EXERCISE 1: Cosine Annealing with Warm Restarts")
    print("="*60)

    # TODO: Implement your solution here
    # Hint: Use torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    # Set T_0=10 (restart every 10 epochs)

    print("\nTODO: Implement CosineAnnealingWarmRestarts")
    print("Expected: Learning rate should cycle periodically")
    print("Benefit: Helps escape local minima")


def exercise1_solution():
    """Solution for Exercise 1."""
    print("\n" + "-"*60)
    print("EXERCISE 1: Solution")
    print("-"*60)

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    # Setup
    model = nn.Sequential(
        nn.Linear(20, 50),
        nn.ReLU(),
        nn.Linear(50, 4)
    )
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001)

    # Generate data
    X_train, y_train = generate_classification_data(num_samples=500)
    criterion = nn.CrossEntropyLoss()

    lr_history = []

    # Train
    for epoch in range(50):
        lr_history.append(optimizer.param_groups[0]['lr'])

        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, LR: {lr_history[-1]:.6f}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(lr_history, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('CosineAnnealingWarmRestarts Schedule')
    plt.grid(True, alpha=0.3)
    plt.savefig('exercise1_lr_schedule.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'exercise1_lr_schedule.png'")
    plt.show()


# ============================================================================
# EXERCISE 2: Compare Networks With and Without Batch Normalization
# ============================================================================

def exercise2_batch_normalization_comparison():
    """
    Exercise 2: Build and compare two CNNs - with and without BatchNorm.

    Tasks:
    1. Create CNN without BatchNorm
    2. Create CNN with BatchNorm after each Conv layer
    3. Train both for 30 epochs
    4. Compare convergence speed and final accuracy
    5. Visualize training curves

    Expected result:
    - BatchNorm model converges faster
    - BatchNorm model achieves higher accuracy
    """
    print("\n" + "="*60)
    print("EXERCISE 2: Batch Normalization Comparison")
    print("="*60)

    # TODO: Implement your solution here
    # Hint: Add nn.BatchNorm2d after Conv2d layers
    # Compare training speed and accuracy

    print("\nTODO: Build two CNNs and compare performance")
    print("Expected: BatchNorm model should train faster")


def exercise2_solution():
    """Solution for Exercise 2."""
    print("\n" + "-"*60)
    print("EXERCISE 2: Solution")
    print("-"*60)

    class CNNWithoutBN(nn.Module):
        def __init__(self):
            super(CNNWithoutBN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 8 * 8, 5)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x

    class CNNWithBN(nn.Module):
        def __init__(self):
            super(CNNWithBN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 8 * 8, 5)

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x

    # Generate data
    X_train, y_train = generate_image_data(num_samples=400)
    X_val, y_val = generate_image_data(num_samples=100)

    def train_cnn(model, name):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        losses = []

        for epoch in range(30):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"{name} - Epoch {epoch+1}, Loss: {loss.item():.4f}")

        return losses

    print("\nTraining without BatchNorm:")
    model_no_bn = CNNWithoutBN()
    losses_no_bn = train_cnn(model_no_bn, "No BN")

    print("\nTraining with BatchNorm:")
    model_with_bn = CNNWithBN()
    losses_with_bn = train_cnn(model_with_bn, "With BN")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(losses_no_bn, label='Without BatchNorm', linewidth=2)
    plt.plot(losses_with_bn, label='With BatchNorm', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('BatchNorm Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('exercise2_batchnorm_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'exercise2_batchnorm_comparison.png'")
    plt.show()


# ============================================================================
# EXERCISE 3: Implement Custom Data Augmentation Pipeline
# ============================================================================

def exercise3_custom_augmentation():
    """
    Exercise 3: Create a custom data augmentation pipeline.

    Tasks:
    1. Implement augmentation with RandomHorizontalFlip, RandomRotation, ColorJitter
    2. Train model with augmentation
    3. Train model without augmentation
    4. Compare overfitting behavior
    5. Visualize augmented samples

    Expected result:
    - Augmented model should have better validation accuracy
    - Less overfitting with augmentation
    """
    print("\n" + "="*60)
    print("EXERCISE 3: Custom Data Augmentation")
    print("="*60)

    # TODO: Implement your solution here
    # Hint: Use torchvision.transforms.Compose
    # Apply augmentation only during training

    print("\nTODO: Create augmentation pipeline and compare results")
    print("Expected: Augmentation reduces overfitting")


def exercise3_solution():
    """Solution for Exercise 3."""
    print("\n" + "-"*60)
    print("EXERCISE 3: Solution")
    print("-"*60)

    # Define augmentation
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])

    # Simple CNN
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 5)
    )

    # Generate small dataset to encourage overfitting
    X_train, y_train = generate_image_data(num_samples=100)  # Small
    X_val, y_val = generate_image_data(num_samples=50)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(40):
        # Apply augmentation
        X_aug = torch.stack([augmentation(img) for img in X_train])

        # Train
        model.train()
        optimizer.zero_grad()
        outputs = model(X_aug)
        train_loss = criterion(outputs, y_train)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        # Validate (no augmentation)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training with Data Augmentation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('exercise3_augmentation.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'exercise3_augmentation.png'")
    plt.show()


# ============================================================================
# EXERCISE 4: Implement Early Stopping with Model Restoration
# ============================================================================

def exercise4_early_stopping_restoration():
    """
    Exercise 4: Implement early stopping that restores best model.

    Tasks:
    1. Create an EarlyStopping class
    2. Monitor validation loss
    3. Save best model weights
    4. Restore best model when stopping
    5. Compare final vs best model performance

    Expected result:
    - Training stops before overfitting
    - Best model is restored, not final model
    """
    print("\n" + "="*60)
    print("EXERCISE 4: Early Stopping with Restoration")
    print("="*60)

    # TODO: Implement your solution here
    # Hint: Save model state_dict when validation improves
    # Restore when early stopping triggers

    print("\nTODO: Implement early stopping with model restoration")
    print("Expected: Best model weights are restored")


def exercise4_solution():
    """Solution for Exercise 4."""
    print("\n" + "-"*60)
    print("EXERCISE 4: Solution")
    print("-"*60)

    class EarlyStopping:
        def __init__(self, patience=5):
            self.patience = patience
            self.counter = 0
            self.best_score = None
            self.best_model = None
            self.early_stop = False

        def __call__(self, val_loss, model):
            if self.best_score is None:
                self.best_score = val_loss
                self.best_model = copy.deepcopy(model.state_dict())
            elif val_loss < self.best_score:
                self.best_score = val_loss
                self.best_model = copy.deepcopy(model.state_dict())
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

        def load_best_model(self, model):
            model.load_state_dict(self.best_model)

    # Setup
    model = nn.Sequential(
        nn.Linear(20, 100),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 4)
    )

    X_train, y_train = generate_classification_data(num_samples=300)
    X_val, y_val = generate_classification_data(num_samples=100)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=5)

    val_losses = []

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

        early_stopping(val_loss.item(), model)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Val Loss: {val_loss.item():.4f}")

        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Restore best model
    early_stopping.load_best_model(model)
    print("Best model restored!")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(val_losses, linewidth=2)
    plt.axvline(x=len(val_losses)-early_stopping.patience, color='r',
                linestyle='--', label='Best Model')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Early Stopping Demonstration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('exercise4_early_stopping.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'exercise4_early_stopping.png'")
    plt.show()


# ============================================================================
# EXERCISE 5: Combined Optimization Techniques
# ============================================================================

def exercise5_combined_optimization():
    """
    Exercise 5: Combine multiple optimization techniques.

    Tasks:
    1. Create a model with BatchNorm
    2. Use learning rate scheduling
    3. Implement early stopping
    4. Save checkpoints
    5. Compare with baseline (no optimizations)

    Expected result:
    - Combined techniques achieve best performance
    - Faster convergence than baseline
    - Better final accuracy
    """
    print("\n" + "="*60)
    print("EXERCISE 5: Combined Optimization Techniques")
    print("="*60)

    # TODO: Implement your solution here
    # Combine: BatchNorm + LR Scheduler + Early Stopping
    # Show improvement over baseline

    print("\nTODO: Combine multiple optimization techniques")
    print("Expected: Best performance with all techniques combined")


def exercise5_solution():
    """Solution for Exercise 5."""
    print("\n" + "-"*60)
    print("EXERCISE 5: Solution")
    print("-"*60)

    # Optimized model
    class OptimizedNet(nn.Module):
        def __init__(self):
            super(OptimizedNet, self).__init__()
            self.fc1 = nn.Linear(20, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.fc2 = nn.Linear(64, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.fc3 = nn.Linear(64, 4)

        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.fc3(x)
            return x

    # Generate data
    X_train, y_train = generate_classification_data(num_samples=500)
    X_val, y_val = generate_classification_data(num_samples=150)

    model = OptimizedNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []

    for epoch in range(50):
        # Train
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, predicted = torch.max(val_outputs, 1)
            acc = 100.0 * (predicted == y_val).sum().item() / y_val.size(0)
            val_accs.append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {acc:.2f}%")

    print(f"\nFinal accuracy: {val_accs[-1]:.2f}%")
    print("Techniques used: BatchNorm + Cosine LR Scheduler + Adam")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run all exercises."""
    print("="*60)
    print("TRAINING OPTIMIZATION EXERCISES")
    print("="*60)
    print("\nChoose an exercise to run:")
    print("1. Cosine Annealing with Warm Restarts")
    print("2. Batch Normalization Comparison")
    print("3. Custom Data Augmentation")
    print("4. Early Stopping with Restoration")
    print("5. Combined Optimization Techniques")
    print("0. Run all solutions")

    choice = input("\nEnter choice (0-5): ").strip()

    if choice == '1':
        exercise1_cosine_annealing_warm_restarts()
        run_solution = input("Run solution? (y/n): ").strip().lower()
        if run_solution == 'y':
            exercise1_solution()
    elif choice == '2':
        exercise2_batch_normalization_comparison()
        run_solution = input("Run solution? (y/n): ").strip().lower()
        if run_solution == 'y':
            exercise2_solution()
    elif choice == '3':
        exercise3_custom_augmentation()
        run_solution = input("Run solution? (y/n): ").strip().lower()
        if run_solution == 'y':
            exercise3_solution()
    elif choice == '4':
        exercise4_early_stopping_restoration()
        run_solution = input("Run solution? (y/n): ").strip().lower()
        if run_solution == 'y':
            exercise4_solution()
    elif choice == '5':
        exercise5_combined_optimization()
        run_solution = input("Run solution? (y/n): ").strip().lower()
        if run_solution == 'y':
            exercise5_solution()
    elif choice == '0':
        print("\nRunning all solutions...")
        exercise1_solution()
        exercise2_solution()
        exercise3_solution()
        exercise4_solution()
        exercise5_solution()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
