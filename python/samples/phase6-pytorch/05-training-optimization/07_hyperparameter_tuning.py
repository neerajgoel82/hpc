"""
Hyperparameter Tuning in PyTorch
===============================
Demonstrates systematic hyperparameter optimization techniques.

Tuning Strategies:
1. Grid Search: Try all combinations (exhaustive but expensive)
2. Random Search: Try random combinations (often better than grid)
3. Bayesian Optimization: Smart search (advanced, not shown here)

Common Hyperparameters to Tune:
- Learning rate
- Batch size
- Number of layers
- Hidden layer size
- Dropout rate
- Weight decay
- Optimizer choice
"""

import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import matplotlib.pyplot as plt
from time import time


class ConfigurableNet(nn.Module):
    """Neural network with configurable architecture."""
    def __init__(self, input_size=20, hidden_size=64, num_layers=2,
                 dropout=0.2, num_classes=3):
        super(ConfigurableNet, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def generate_data(num_samples=1000, input_size=20, num_classes=3):
    """Generate synthetic classification data."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def train_model(model, optimizer, X_train, y_train, X_val, y_val,
                num_epochs=20, batch_size=32):
    """Train model and return validation accuracy."""
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
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

    # Final validation accuracy
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        _, predicted = torch.max(val_outputs, 1)
        val_acc = 100.0 * (predicted == y_val).sum().item() / y_val.size(0)

    return val_acc


def grid_search(X_train, y_train, X_val, y_val):
    """Perform grid search over hyperparameters."""
    print("\n" + "="*60)
    print("Grid Search")
    print("="*60)
    print("Testing all combinations of hyperparameters...")

    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'hidden_size': [32, 64, 128],
        'dropout': [0.1, 0.3],
        'num_layers': [2, 3]
    }

    # Calculate total combinations
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Total combinations to test: {total_combinations}")

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    best_acc = 0
    best_params = None

    start_time = time()

    for idx, params in enumerate(combinations, 1):
        print(f"\n[{idx}/{total_combinations}] Testing: {params}")

        # Create model with these parameters
        model = ConfigurableNet(
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )

        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        # Train and evaluate
        val_acc = train_model(model, optimizer, X_train, y_train, X_val, y_val,
                             num_epochs=15, batch_size=32)

        results.append({**params, 'val_acc': val_acc})

        print(f"  Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params

    elapsed_time = time() - start_time

    print("\n" + "="*60)
    print("GRID SEARCH RESULTS")
    print("="*60)
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Best parameters: {best_params}")

    return results, best_params


def random_search(X_train, y_train, X_val, y_val, n_iterations=20):
    """Perform random search over hyperparameters."""
    print("\n" + "="*60)
    print("Random Search")
    print("="*60)
    print(f"Testing {n_iterations} random combinations...")

    # Define hyperparameter ranges
    param_space = {
        'learning_rate': (0.0001, 0.01, 'log'),  # log scale
        'hidden_size': (32, 128, 'int'),         # integer
        'dropout': (0.0, 0.5, 'uniform'),        # uniform
        'num_layers': (2, 4, 'int')              # integer
    }

    def sample_params():
        """Sample random parameters from the space."""
        params = {}
        for name, (low, high, scale) in param_space.items():
            if scale == 'log':
                params[name] = 10 ** np.random.uniform(np.log10(low), np.log10(high))
            elif scale == 'int':
                params[name] = np.random.randint(low, high + 1)
            else:  # uniform
                params[name] = np.random.uniform(low, high)
        return params

    results = []
    best_acc = 0
    best_params = None

    start_time = time()

    for idx in range(n_iterations):
        params = sample_params()
        print(f"\n[{idx+1}/{n_iterations}] Testing: ")
        print(f"  LR={params['learning_rate']:.6f}, "
              f"Hidden={params['hidden_size']}, "
              f"Dropout={params['dropout']:.2f}, "
              f"Layers={params['num_layers']}")

        # Create model
        model = ConfigurableNet(
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )

        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        # Train and evaluate
        val_acc = train_model(model, optimizer, X_train, y_train, X_val, y_val,
                             num_epochs=15, batch_size=32)

        results.append({**params, 'val_acc': val_acc})

        print(f"  Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params

    elapsed_time = time() - start_time

    print("\n" + "="*60)
    print("RANDOM SEARCH RESULTS")
    print("="*60)
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Best parameters:")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    return results, best_params


def visualize_results(grid_results, random_results):
    """Visualize hyperparameter tuning results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Grid Search: Learning Rate vs Accuracy
    ax1 = axes[0, 0]
    lrs = [r['learning_rate'] for r in grid_results]
    accs = [r['val_acc'] for r in grid_results]
    ax1.scatter(lrs, accs, alpha=0.6, s=100)
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Grid Search: Learning Rate Effect')
    ax1.grid(True, alpha=0.3)

    # Grid Search: Hidden Size vs Accuracy
    ax2 = axes[0, 1]
    hidden_sizes = [r['hidden_size'] for r in grid_results]
    ax2.scatter(hidden_sizes, accs, alpha=0.6, s=100)
    ax2.set_xlabel('Hidden Size')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Grid Search: Hidden Size Effect')
    ax2.grid(True, alpha=0.3)

    # Random Search: Learning Rate vs Accuracy
    ax3 = axes[1, 0]
    rand_lrs = [r['learning_rate'] for r in random_results]
    rand_accs = [r['val_acc'] for r in random_results]
    ax3.scatter(rand_lrs, rand_accs, alpha=0.6, s=100, c='orange')
    ax3.set_xscale('log')
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Validation Accuracy (%)')
    ax3.set_title('Random Search: Learning Rate Effect')
    ax3.grid(True, alpha=0.3)

    # Comparison: Grid vs Random Search Performance
    ax4 = axes[1, 1]
    grid_sorted = sorted([r['val_acc'] for r in grid_results], reverse=True)
    random_sorted = sorted([r['val_acc'] for r in random_results], reverse=True)

    ax4.plot(range(len(grid_sorted)), grid_sorted, 'b-', label='Grid Search',
             linewidth=2, marker='o')
    ax4.plot(range(len(random_sorted)), random_sorted, 'r-', label='Random Search',
             linewidth=2, marker='s')
    ax4.set_xlabel('Configuration Rank')
    ax4.set_ylabel('Validation Accuracy (%)')
    ax4.set_title('Performance Comparison: Grid vs Random')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hyperparameter_tuning_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'hyperparameter_tuning_results.png'")
    plt.show()


def demonstrate_learning_rate_sensitivity():
    """Show how sensitive model is to learning rate."""
    print("\n" + "="*60)
    print("Learning Rate Sensitivity Analysis")
    print("="*60)

    X_train, y_train = generate_data(num_samples=500)
    X_val, y_val = generate_data(num_samples=100)

    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    results = []

    for lr in learning_rates:
        print(f"\nTesting LR = {lr}")
        model = ConfigurableNet(hidden_size=64, num_layers=2, dropout=0.2)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        val_acc = train_model(model, optimizer, X_train, y_train, X_val, y_val,
                             num_epochs=20)
        results.append((lr, val_acc))
        print(f"  Accuracy: {val_acc:.2f}%")

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    lrs, accs = zip(*results)
    ax.plot(lrs, accs, 'bo-', linewidth=2, markersize=10)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Learning Rate Sensitivity')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0.001, color='r', linestyle='--', label='Typical Good Value')
    ax.legend()

    plt.tight_layout()
    plt.savefig('learning_rate_sensitivity.png', dpi=150, bbox_inches='tight')
    print("\nLearning rate sensitivity plot saved!")
    plt.show()


def main():
    """Main demonstration function."""
    print("="*60)
    print("HYPERPARAMETER TUNING IN PYTORCH")
    print("="*60)

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate data
    print("\nGenerating training and validation data...")
    X_train, y_train = generate_data(num_samples=600)
    X_val, y_val = generate_data(num_samples=150)

    # Demonstrate learning rate sensitivity
    demonstrate_learning_rate_sensitivity()

    # Grid search (smaller grid for demo)
    grid_results, best_grid_params = grid_search(X_train, y_train, X_val, y_val)

    # Random search
    random_results, best_random_params = random_search(
        X_train, y_train, X_val, y_val, n_iterations=20
    )

    # Visualize results
    visualize_results(grid_results, random_results)

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("1. Grid Search: Exhaustive but expensive")
    print("2. Random Search: Often finds good solutions faster")
    print("3. Learning rate is usually most important hyperparameter")
    print("4. Start with coarse search, then fine-tune")
    print("5. Use validation set for tuning, not test set")
    print("6. Random search scales better to many hyperparameters")
    print("7. Consider automated tools: Optuna, Ray Tune")
    print("8. Balance search thoroughness vs computational cost")
    print("="*60)


if __name__ == "__main__":
    main()
