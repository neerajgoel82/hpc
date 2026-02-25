"""
Transfer Learning with ResNet - Pretrained Models

This module demonstrates:
- Loading pretrained ResNet models
- Feature extraction vs fine-tuning
- Freezing and unfreezing layers
- Adapting models for custom datasets
- Transfer learning best practices
- Comparing different pretrained architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time


def load_pretrained_resnet():
    """Load and explore pretrained ResNet"""
    print("=" * 60)
    print("Pretrained ResNet Models")
    print("=" * 60)

    # Load pretrained ResNet18
    resnet18 = models.resnet18(pretrained=True)

    print("\nResNet18 Architecture:")
    print(f"  Total layers: {len(list(resnet18.children()))}")

    # Model structure
    print("\n  Main components:")
    for name, module in resnet18.named_children():
        print(f"    {name}: {module.__class__.__name__}")

    # Count parameters
    total_params = sum(p.numel() for p in resnet18.parameters())
    trainable_params = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Final layer
    print(f"\n  Original final layer (ImageNet 1000 classes):")
    print(f"    {resnet18.fc}")

    return resnet18


def feature_extraction_model(num_classes=10):
    """
    Create ResNet model for feature extraction

    Strategy: Freeze all layers except final FC layer
    Use case: Small dataset, similar to ImageNet
    """
    print("\n" + "=" * 60)
    print("Feature Extraction Model")
    print("=" * 60)

    # Load pretrained model
    model = models.resnet18(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    print(f"\nFrozen layers: All convolutional layers")
    print(f"Trainable layer: Final FC layer only")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model


def fine_tuning_model(num_classes=10, freeze_layers=7):
    """
    Create ResNet model for fine-tuning

    Strategy: Freeze early layers, train later layers
    Use case: Medium dataset, somewhat similar to ImageNet
    """
    print("\n" + "=" * 60)
    print("Fine-Tuning Model")
    print("=" * 60)

    model = models.resnet18(pretrained=True)

    # Freeze early layers
    layers = list(model.children())
    for layer in layers[:freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False

    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    print(f"\nFrozen layers: First {freeze_layers} layers")
    print(f"Trainable layers: Last {len(layers) - freeze_layers} layers + new FC")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model


def full_finetuning_model(num_classes=10):
    """
    Create ResNet model for full fine-tuning

    Strategy: Train all layers with small learning rate
    Use case: Large dataset or very different from ImageNet
    """
    print("\n" + "=" * 60)
    print("Full Fine-Tuning Model")
    print("=" * 60)

    model = models.resnet18(pretrained=True)

    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    print(f"\nAll layers trainable")
    print(f"Strategy: Use smaller learning rate for pretrained layers")

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_params:,}")

    return model


def create_discriminative_optimizer(model, lr_fc=0.001, lr_backbone=0.0001):
    """
    Create optimizer with different learning rates for different layers

    Higher LR for new layers, lower LR for pretrained layers
    """
    print("\n" + "=" * 60)
    print("Discriminative Learning Rates")
    print("=" * 60)

    # Separate parameters
    backbone_params = []
    fc_params = []

    for name, param in model.named_parameters():
        if 'fc' in name:
            fc_params.append(param)
        else:
            backbone_params.append(param)

    # Create optimizer with different LRs
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': fc_params, 'lr': lr_fc}
    ])

    print(f"\nBackbone LR: {lr_backbone} (pretrained layers)")
    print(f"FC layer LR: {lr_fc} (new layer)")
    print(f"Ratio: {lr_fc / lr_backbone}x higher for new layers")

    return optimizer


def prepare_data_for_transfer_learning(batch_size=32):
    """Prepare data with ImageNet normalization"""
    print("\n" + "=" * 60)
    print("Data Preparation for Transfer Learning")
    print("=" * 60)

    # ImageNet normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    print("\nImageNet preprocessing:")
    print(f"  Input size: 224x224 (ResNet requirement)")
    print(f"  Normalization: mean={imagenet_mean}, std={imagenet_std}")

    # Use CIFAR-10 as example (convert to 3-channel)
    train_dataset = datasets.CIFAR10(
        './data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        './data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 50 == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')

    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, device, test_loader, criterion):
    """Evaluate model"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy


def compare_transfer_strategies():
    """Compare different transfer learning strategies"""
    print("\n" + "=" * 60)
    print("Transfer Learning Strategy Comparison")
    print("=" * 60)

    strategies = [
        ("Feature Extraction", "Freeze all, train FC only", "Small dataset, similar domain"),
        ("Fine-tuning", "Freeze early layers, train later layers", "Medium dataset"),
        ("Full Fine-tuning", "Train all with small LR", "Large dataset, different domain"),
    ]

    print("\nStrategy Guide:")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Approach':<35} {'Use Case'}")
    print("-" * 60)
    for name, approach, use_case in strategies:
        print(f"{name:<20} {approach:<35} {use_case}")


def demonstrate_progressive_unfreezing():
    """Demonstrate progressive unfreezing technique"""
    print("\n" + "=" * 60)
    print("Progressive Unfreezing")
    print("=" * 60)

    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    print("\nProgressive unfreezing strategy:")
    print("1. Train only final layer (1-2 epochs)")
    print("2. Unfreeze layer4, train (1-2 epochs)")
    print("3. Unfreeze layer3, train (1-2 epochs)")
    print("4. Optionally unfreeze more layers")

    # Example: freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze FC
    for param in model.fc.parameters():
        param.requires_grad = True

    print("\nStep 1: Only FC trainable")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    print("\nStep 2: FC + layer4 trainable")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")


def train_transfer_model():
    """Train a transfer learning model"""
    print("\n" + "=" * 60)
    print("Training Transfer Learning Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Prepare data
    train_loader, test_loader = prepare_data_for_transfer_learning(batch_size=32)

    # Create feature extraction model
    model = feature_extraction_model(num_classes=10).to(device)

    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # Train for a few epochs
    print("\nTraining for 2 epochs...")
    for epoch in range(1, 3):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch}/2 ({epoch_time:.2f}s):")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


def main():
    """Main demonstration function"""
    print("\n" + "=" * 60)
    print("TRANSFER LEARNING WITH RESNET")
    print("=" * 60)

    # Load pretrained model
    resnet = load_pretrained_resnet()

    # Feature extraction
    fe_model = feature_extraction_model(num_classes=10)

    # Fine-tuning
    ft_model = fine_tuning_model(num_classes=10, freeze_layers=6)

    # Full fine-tuning
    full_model = full_finetuning_model(num_classes=10)

    # Discriminative learning rates
    optimizer = create_discriminative_optimizer(full_model)

    # Compare strategies
    compare_transfer_strategies()

    # Progressive unfreezing
    demonstrate_progressive_unfreezing()

    # Train a model
    train_transfer_model()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Transfer learning leverages pretrained knowledge")
    print("2. Feature extraction: freeze all, train FC only")
    print("3. Fine-tuning: selectively train layers")
    print("4. Use lower LR for pretrained layers")
    print("5. Progressive unfreezing for best results")
    print("6. ImageNet normalization required for pretrained models")
    print("=" * 60)


if __name__ == "__main__":
    main()
