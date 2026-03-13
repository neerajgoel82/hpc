"""
Fine-Tuning
Unfreezing and fine-tuning pretrained models

This module demonstrates:
- Fine-tuning strategies (gradual unfreezing)
- Layer-specific learning rates
- Discriminative learning rates
- Progressive fine-tuning
- When to use fine-tuning vs feature extraction

Run: python 03_fine_tuning.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models


def check_environment():
    """Display PyTorch environment information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()


def basic_fine_tuning():
    """Basic fine-tuning: unfreeze all layers."""
    print("1. Basic Fine-Tuning")
    print("-" * 50)

    # Load pretrained model
    model = models.resnet18(pretrained=True)

    # Replace final layer
    num_features = model.fc.in_features
    num_classes = 10
    model.fc = nn.Linear(num_features, num_classes)

    # All layers are trainable (default)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\nAll layers will be fine-tuned")
    print("Use a small learning rate to avoid destroying pretrained weights!")
    print()

    return model


def selective_fine_tuning():
    """Selectively unfreeze only later layers."""
    print("2. Selective Fine-Tuning")
    print("-" * 50)

    model = models.resnet18(pretrained=True)

    # Freeze early layers
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    # Count trainable parameters
    print("Trainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,} parameters")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal: {total:,}")
    print(f"Trainable: {trainable:,}")
    print(f"Percentage: {100 * trainable / total:.2f}%")
    print()

    return model


def discriminative_learning_rates():
    """Use different learning rates for different layers."""
    print("3. Discriminative Learning Rates")
    print("-" * 50)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)

    # Group parameters by layer
    base_params = []
    layer4_params = []
    fc_params = list(model.fc.parameters())

    for name, param in model.named_parameters():
        if 'layer4' in name:
            layer4_params.append(param)
        elif 'fc' not in name:
            base_params.append(param)

    # Different learning rates
    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-5},      # Very small for early layers
        {'params': layer4_params, 'lr': 1e-4},    # Medium for layer4
        {'params': fc_params, 'lr': 1e-3}         # Large for new FC layer
    ])

    print("Learning rate schedule:")
    print(f"  Early layers (layer1-3): 1e-5")
    print(f"  Layer 4: 1e-4")
    print(f"  Final FC layer: 1e-3")
    print("\nThis allows:")
    print("  - Preserve pretrained features in early layers")
    print("  - Moderate adaptation in middle layers")
    print("  - Fast learning in final layer")
    print()

    return model, optimizer


def progressive_fine_tuning():
    """Demonstrate progressive unfreezing strategy."""
    print("4. Progressive Fine-Tuning")
    print("-" * 50)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)

    print("Progressive unfreezing strategy:")
    print("\nPhase 1: Train only final layer")
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze final layer
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable1:,}")

    print("\nPhase 2: Unfreeze layer4")
    for param in model.layer4.parameters():
        param.requires_grad = True

    trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable2:,}")

    print("\nPhase 3: Unfreeze all layers")
    for param in model.parameters():
        param.requires_grad = True

    trainable3 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable3:,}")

    print("\nBenefits:")
    print("  - Stable training")
    print("  - Better final performance")
    print("  - Reduced overfitting risk")
    print()

    return model


def create_synthetic_data(num_samples=100, num_classes=10):
    """Create synthetic dataset for demonstration."""
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return train_loader


def train_with_fine_tuning(model, train_loader, optimizer, epochs=2):
    """Train with fine-tuning."""
    print("5. Training with Fine-Tuning")
    print("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    print(f"Training on: {device}")
    print(f"Epochs: {epochs}")
    print()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")

    print()
    return model


def compare_strategies():
    """Compare feature extraction vs fine-tuning."""
    print("6. Feature Extraction vs Fine-Tuning")
    print("-" * 50)

    print("Feature Extraction:")
    print("  Pros:")
    print("    + Faster training")
    print("    + Less memory usage")
    print("    + Good for small datasets")
    print("    + Less risk of overfitting")
    print("  Cons:")
    print("    - May not adapt to very different domains")
    print("    - Limited performance ceiling")
    print("  Use when:")
    print("    - Small dataset (< 1000 images)")
    print("    - Similar to ImageNet domain")
    print("    - Quick prototyping")

    print("\nFine-Tuning:")
    print("  Pros:")
    print("    + Better performance on target task")
    print("    + Adapts to new domain")
    print("    + More flexible")
    print("  Cons:")
    print("    - Slower training")
    print("    - More memory usage")
    print("    - Risk of overfitting on small data")
    print("  Use when:")
    print("    - Larger dataset (> 1000 images)")
    print("    - Different from ImageNet domain")
    print("    - Need best performance")
    print()


def learning_rate_scheduling():
    """Demonstrate learning rate scheduling for fine-tuning."""
    print("7. Learning Rate Scheduling")
    print("-" * 50)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    print("Learning rate scheduler: ReduceLROnPlateau")
    print("  Initial LR: 1e-3")
    print("  Factor: 0.5")
    print("  Patience: 3 epochs")

    print("\nOther useful schedulers:")
    print("  - CosineAnnealingLR: Smooth decay")
    print("  - StepLR: Step decay")
    print("  - OneCycleLR: One-cycle policy")
    print()

    return scheduler


def main():
    print("=" * 60)
    print("FINE-TUNING PRETRAINED MODELS")
    print("=" * 60)
    print()

    check_environment()

    # Basic fine-tuning
    model_basic = basic_fine_tuning()

    # Selective fine-tuning
    model_selective = selective_fine_tuning()

    # Discriminative learning rates
    model_disc, optimizer_disc = discriminative_learning_rates()

    # Progressive fine-tuning
    model_prog = progressive_fine_tuning()

    # Create data and train
    train_loader = create_synthetic_data()
    trained_model = train_with_fine_tuning(
        model_selective,
        train_loader,
        optim.Adam(filter(lambda p: p.requires_grad, model_selective.parameters()), lr=1e-4),
        epochs=2
    )

    # Compare strategies
    compare_strategies()

    # Learning rate scheduling
    scheduler = learning_rate_scheduling()

    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Fine-tuning adapts pretrained models to new tasks")
    print("2. Use discriminative learning rates (smaller for early layers)")
    print("3. Progressive unfreezing provides stable training")
    print("4. Choose strategy based on dataset size and domain")
    print("5. Learning rate scheduling improves convergence")
    print("=" * 60)


if __name__ == "__main__":
    main()
