"""
Learning Rate Scheduling in PyTorch
==================================
Demonstrates different learning rate scheduling strategies to improve training.

Learning rate scheduling helps:
- Start with large steps for fast convergence
- Reduce learning rate to fine-tune
- Escape local minima
- Improve final model accuracy

Key schedulers:
- StepLR: Reduce LR by gamma every step_size epochs
- ExponentialLR: Reduce LR exponentially
- CosineAnnealingLR: Cosine annealing schedule
- ReduceLROnPlateau: Reduce when metric plateaus
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np


# Simple neural network for demonstration
class SimpleNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=50, num_classes=3):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_dummy_data(num_samples=1000, input_size=10, num_classes=3):
    """Generate dummy classification data."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def train_with_scheduler(model, optimizer, scheduler, X_train, y_train,
                         num_epochs=50, batch_size=32):
    """Train model and track learning rates."""
    criterion = nn.CrossEntropyLoss()
    num_samples = X_train.size(0)
    lr_history = []
    loss_history = []

    for epoch in range(num_epochs):
        # Track current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # Training
        model.train()
        epoch_loss = 0.0

        # Mini-batch training
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

        avg_loss = epoch_loss / (num_samples // batch_size)
        loss_history.append(avg_loss)

        # Step the scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_loss)
        else:
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

    return lr_history, loss_history


def demonstrate_step_lr():
    """StepLR: Multiply LR by gamma every step_size epochs."""
    print("\n" + "="*60)
    print("StepLR Scheduler Demo")
    print("="*60)
    print("Reduces learning rate by factor gamma every step_size epochs")
    print("Good for: Training in stages with clear phases\n")

    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    X_train, y_train = generate_dummy_data()
    lr_hist, loss_hist = train_with_scheduler(model, optimizer, scheduler,
                                              X_train, y_train)

    return lr_hist, loss_hist, "StepLR (step=10, gamma=0.5)"


def demonstrate_exponential_lr():
    """ExponentialLR: Multiply LR by gamma every epoch."""
    print("\n" + "="*60)
    print("ExponentialLR Scheduler Demo")
    print("="*60)
    print("Reduces learning rate exponentially: lr = lr * gamma")
    print("Good for: Smooth continuous decay\n")

    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    X_train, y_train = generate_dummy_data()
    lr_hist, loss_hist = train_with_scheduler(model, optimizer, scheduler,
                                              X_train, y_train)

    return lr_hist, loss_hist, "ExponentialLR (gamma=0.95)"


def demonstrate_cosine_annealing():
    """CosineAnnealingLR: Cosine annealing schedule."""
    print("\n" + "="*60)
    print("CosineAnnealingLR Scheduler Demo")
    print("="*60)
    print("Follows cosine curve from max to min learning rate")
    print("Good for: Better convergence, avoiding sharp drops\n")

    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.001)

    X_train, y_train = generate_dummy_data()
    lr_hist, loss_hist = train_with_scheduler(model, optimizer, scheduler,
                                              X_train, y_train)

    return lr_hist, loss_hist, "CosineAnnealingLR (T_max=50)"


def demonstrate_reduce_on_plateau():
    """ReduceLROnPlateau: Reduce when metric plateaus."""
    print("\n" + "="*60)
    print("ReduceLROnPlateau Scheduler Demo")
    print("="*60)
    print("Reduces LR when validation metric stops improving")
    print("Good for: Adaptive learning based on performance\n")

    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=5, verbose=False)

    X_train, y_train = generate_dummy_data()
    lr_hist, loss_hist = train_with_scheduler(model, optimizer, scheduler,
                                              X_train, y_train)

    return lr_hist, loss_hist, "ReduceLROnPlateau"


def compare_no_scheduler():
    """Train without scheduler for comparison."""
    print("\n" + "="*60)
    print("No Scheduler (Baseline)")
    print("="*60)
    print("Constant learning rate throughout training\n")

    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Use a dummy scheduler that does nothing
    class DummyScheduler:
        def step(self, *args): pass

    scheduler = DummyScheduler()

    X_train, y_train = generate_dummy_data()
    lr_hist, loss_hist = train_with_scheduler(model, optimizer, scheduler,
                                              X_train, y_train)

    return lr_hist, loss_hist, "No Scheduler (Constant LR)"


def visualize_schedulers():
    """Visualize all schedulers and their effects."""
    print("\n" + "="*60)
    print("Visualizing All Schedulers")
    print("="*60)

    results = []

    # Run all scheduler demos
    results.append(compare_no_scheduler())
    results.append(demonstrate_step_lr())
    results.append(demonstrate_exponential_lr())
    results.append(demonstrate_cosine_annealing())
    results.append(demonstrate_reduce_on_plateau())

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot learning rates
    ax1 = axes[0]
    for lr_hist, _, label in results:
        ax1.plot(lr_hist, label=label, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Learning Rate Schedules Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot training losses
    ax2 = axes[1]
    for _, loss_hist, label in results:
        ax2.plot(loss_hist, label=label, linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_rate_schedulers_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'learning_rate_schedulers_comparison.png'")
    plt.show()


def main():
    """Main demonstration function."""
    print("="*60)
    print("LEARNING RATE SCHEDULING IN PYTORCH")
    print("="*60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run all demonstrations and visualize
    visualize_schedulers()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("1. StepLR: Good for staged training with clear phases")
    print("2. ExponentialLR: Smooth decay, predictable behavior")
    print("3. CosineAnnealingLR: Better convergence, avoids sharp drops")
    print("4. ReduceLROnPlateau: Adaptive, responds to training dynamics")
    print("5. Schedulers often improve final accuracy by 1-5%")
    print("6. Start with higher LR, decay for fine-tuning")
    print("="*60)


if __name__ == "__main__":
    main()
