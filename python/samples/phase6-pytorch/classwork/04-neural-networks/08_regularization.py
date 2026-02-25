"""
Regularization
Dropout, weight decay, and batch normalization

This module demonstrates:
- Dropout for regularization
- Weight decay (L2 regularization)
- Batch normalization
- Layer normalization
- Comparing regularization techniques
- Overfitting prevention

Run: python 08_regularization.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLPWithDropout(nn.Module):
    """MLP with dropout layers"""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class MLPWithBatchNorm(nn.Module):
    """MLP with batch normalization"""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class MLPWithLayerNorm(nn.Module):
    """MLP with layer normalization"""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class MLPCombined(nn.Module):
    """MLP with combined regularization techniques"""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


def create_overfitting_dataset(n_train: int = 100, n_test: int = 500,
                               input_size: int = 20, num_classes: int = 3):
    """Create small training set to demonstrate overfitting"""
    X_train = torch.randn(n_train, input_size)
    y_train = torch.randint(0, num_classes, (n_train,))
    train_dataset = TensorDataset(X_train, y_train)

    X_test = torch.randn(n_test, input_size)
    y_test = torch.randint(0, num_classes, (n_test,))
    test_dataset = TensorDataset(X_test, y_test)

    return train_dataset, test_dataset


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion,
                      device, epochs: int = 20):
    """Train and track train/test performance"""
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        # Training
        model.train()
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_acc = 100.0 * correct / total
        train_accs.append(train_acc)

        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_acc = 100.0 * correct / total
        test_accs.append(test_acc)

    return train_accs, test_accs


def demonstrate_dropout():
    """Demonstrate dropout regularization"""
    print("\n1. Dropout Regularization")
    print("-" * 60)

    model = MLPWithDropout(input_size=10, hidden_size=20, num_classes=3,
                           dropout_rate=0.5)

    x = torch.randn(5, 10)

    # Training mode: dropout is active
    model.train()
    out1 = model(x)
    out2 = model(x)
    print(f"Training mode (dropout active):")
    print(f"  Output 1: {out1[0]}")
    print(f"  Output 2: {out2[0]}")
    print(f"  Outputs differ: {not torch.allclose(out1, out2)}")

    # Eval mode: dropout is disabled
    model.eval()
    out3 = model(x)
    out4 = model(x)
    print(f"\nEvaluation mode (dropout disabled):")
    print(f"  Output 3: {out3[0]}")
    print(f"  Output 4: {out4[0]}")
    print(f"  Outputs identical: {torch.allclose(out3, out4)}")

    print("\nDropout randomly zeros activations during training")
    print("This prevents co-adaptation of neurons")


def demonstrate_weight_decay():
    """Demonstrate weight decay (L2 regularization)"""
    print("\n2. Weight Decay (L2 Regularization)")
    print("-" * 60)

    model = nn.Linear(10, 5)

    # Without weight decay
    optimizer_no_decay = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0)
    print(f"Without weight decay: {optimizer_no_decay.defaults['weight_decay']}")

    # With weight decay
    optimizer_with_decay = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    print(f"With weight decay: {optimizer_with_decay.defaults['weight_decay']}")

    print("\nWeight decay penalizes large weights")
    print("Equivalent to L2 regularization: loss + lambda * ||weights||^2")


def demonstrate_batch_norm():
    """Demonstrate batch normalization"""
    print("\n3. Batch Normalization")
    print("-" * 60)

    batch_norm = nn.BatchNorm1d(num_features=10)

    # Training mode
    batch_norm.train()
    x = torch.randn(32, 10)  # batch_size=32, features=10
    out = batch_norm(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"\nInput statistics:")
    print(f"  Mean: {x.mean(0)[:3]}...")
    print(f"  Std:  {x.std(0)[:3]}...")
    print(f"\nOutput statistics (normalized):")
    print(f"  Mean: {out.mean(0)[:3]}...")
    print(f"  Std:  {out.std(0)[:3]}...")

    print("\nBatch norm normalizes activations to mean=0, std=1")
    print("Reduces internal covariate shift")


def demonstrate_layer_norm():
    """Demonstrate layer normalization"""
    print("\n4. Layer Normalization")
    print("-" * 60)

    # Batch norm normalizes across batch dimension
    batch_norm = nn.BatchNorm1d(num_features=10)

    # Layer norm normalizes across feature dimension
    layer_norm = nn.LayerNorm(normalized_shape=10)

    x = torch.randn(32, 10)

    out_bn = batch_norm(x)
    out_ln = layer_norm(x)

    print(f"Input shape: {x.shape}")
    print(f"\nBatch Norm (normalizes across batch):")
    print(f"  Mean per feature: {out_bn.mean(0)[:3]}...")
    print(f"  Std per feature:  {out_bn.std(0)[:3]}...")

    print(f"\nLayer Norm (normalizes across features):")
    print(f"  Mean per sample: {out_ln.mean(1)[:3]}...")
    print(f"  Std per sample:  {out_ln.std(1)[:3]}...")

    print("\nLayer norm is better for variable batch sizes")
    print("Commonly used in transformers and RNNs")


def demonstrate_overfitting_prevention():
    """Compare regularization techniques for preventing overfitting"""
    print("\n5. Preventing Overfitting")
    print("-" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create small dataset to induce overfitting
    train_dataset, test_dataset = create_overfitting_dataset(
        n_train=100, n_test=500, input_size=20, num_classes=3
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(f"Small training set: {len(train_dataset)} samples")
    print(f"Large test set: {len(test_dataset)} samples")

    models = {
        'No Regularization': nn.Sequential(
            nn.Linear(20, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        ),
        'With Dropout': MLPWithDropout(20, 100, 3, dropout_rate=0.5),
        'With BatchNorm': MLPWithBatchNorm(20, 100, 3),
    }

    print("\nTraining models for 20 epochs...")
    criterion = nn.CrossEntropyLoss()

    for name, model in models.items():
        model.to(device)

        # Use weight decay for all models
        if 'No Regularization' in name:
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

        train_accs, test_accs = train_and_evaluate(
            model, train_loader, test_loader, optimizer, criterion, device, epochs=20
        )

        print(f"\n{name}:")
        print(f"  Train Acc: {train_accs[-1]:.2f}%")
        print(f"  Test Acc:  {test_accs[-1]:.2f}%")
        print(f"  Gap:       {train_accs[-1] - test_accs[-1]:.2f}%")


def demonstrate_combined_regularization():
    """Combine multiple regularization techniques"""
    print("\n6. Combined Regularization")
    print("-" * 60)

    model = MLPCombined(input_size=20, hidden_size=100, num_classes=3,
                        dropout_rate=0.3)

    print("Model with combined regularization:")
    print("- Batch Normalization")
    print("- Dropout (rate=0.3)")
    print("- Weight Decay (via optimizer)")

    print(f"\nModel structure:")
    print(model)

    print("\nCombining techniques often works best:")
    print("- BatchNorm: stabilizes training")
    print("- Dropout: prevents co-adaptation")
    print("- Weight decay: prevents large weights")


def main():
    print("=" * 60)
    print("Regularization Techniques")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    demonstrate_dropout()
    demonstrate_weight_decay()
    demonstrate_batch_norm()
    demonstrate_layer_norm()
    demonstrate_overfitting_prevention()
    demonstrate_combined_regularization()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Regularization techniques:")
    print("1. Dropout: Randomly zeros activations (p=0.2-0.5)")
    print("2. Weight Decay: L2 penalty on weights (0.01-0.0001)")
    print("3. Batch Norm: Normalizes activations across batch")
    print("4. Layer Norm: Normalizes activations across features")
    print("5. Combine multiple techniques for best results")
    print("=" * 60)


if __name__ == "__main__":
    main()
