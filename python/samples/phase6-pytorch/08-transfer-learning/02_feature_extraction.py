"""
Feature Extraction
Using pretrained models as fixed feature extractors

This module demonstrates:
- Freezing convolutional layers
- Replacing classification head
- Using pretrained features for new tasks
- Creating custom classifiers
- Training only the final layers

Run: python 02_feature_extraction.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torchvision.transforms as transforms


def check_environment():
    """Display PyTorch environment information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()


def create_feature_extractor():
    """Create a feature extractor from pretrained ResNet."""
    print("1. Creating Feature Extractor")
    print("-" * 50)

    # Load pretrained ResNet-18
    resnet = models.resnet18(pretrained=True)

    # Freeze all convolutional layers
    for param in resnet.parameters():
        param.requires_grad = False

    # Get number of input features for final layer
    num_features = resnet.fc.in_features
    print(f"Original FC layer input features: {num_features}")

    # Replace final layer (this layer will be trainable)
    num_classes = 10  # Example: 10 classes for new task
    resnet.fc = nn.Linear(num_features, num_classes)

    # Verify freezing
    total_params = sum(p.numel() for p in resnet.parameters())
    trainable_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    print()

    return resnet


def create_custom_classifier(num_classes=10):
    """Create a custom classifier on top of pretrained features."""
    print("2. Custom Classifier")
    print("-" * 50)

    # Load pretrained model
    resnet = models.resnet18(pretrained=True)

    # Freeze all layers
    for param in resnet.parameters():
        param.requires_grad = False

    # Create custom classifier
    num_features = resnet.fc.in_features

    # Multi-layer classifier
    resnet.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    print("Custom classifier architecture:")
    print(resnet.fc)

    trainable = sum(p.numel() for p in resnet.fc.parameters())
    print(f"\nTrainable parameters in classifier: {trainable:,}")
    print()

    return resnet


def extract_features_only():
    """Use model to extract features without classification."""
    print("3. Extracting Features Only")
    print("-" * 50)

    # Load pretrained ResNet
    resnet = models.resnet18(pretrained=True)

    # Remove final classification layer
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    # Set to eval mode
    feature_extractor.eval()

    # Create sample input
    sample_input = torch.randn(4, 3, 224, 224)

    print(f"Input shape: {sample_input.shape}")

    with torch.no_grad():
        features = feature_extractor(sample_input)

    print(f"Output features shape: {features.shape}")
    print(f"Features shape after squeeze: {features.squeeze().shape}")
    print("\nThese features can be used for:")
    print("  - Training separate classifiers")
    print("  - Similarity search")
    print("  - Clustering")
    print("  - Dimensionality reduction")
    print()

    return feature_extractor


def create_synthetic_data(num_samples=100, num_classes=10):
    """Create synthetic dataset for demonstration."""
    print("4. Creating Synthetic Dataset")
    print("-" * 50)

    # Random images (in practice, use real images)
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))

    print(f"Dataset size: {num_samples} samples")
    print(f"Number of classes: {num_classes}")
    print(f"Image shape: {images[0].shape}")
    print()

    # Create DataLoader
    dataset = TensorDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    return train_loader


def train_feature_extractor(model, train_loader, epochs=3):
    """Train only the final layer of feature extractor."""
    print("5. Training Feature Extractor")
    print("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Only optimize final layer parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Training on device: {device}")
    print(f"Number of batches: {len(train_loader)}")
    print()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx == 0:
                print(f"Epoch {epoch+1}/{epochs} - First batch:")
                print(f"  Loss: {loss.item():.4f}")

        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print()

    return model


def compare_training_speed():
    """Compare training speed: feature extraction vs full training."""
    print("6. Training Speed Comparison")
    print("-" * 50)

    # Feature extraction model (only final layer trainable)
    model_fe = models.resnet18(pretrained=True)
    for param in model_fe.parameters():
        param.requires_grad = False
    model_fe.fc = nn.Linear(model_fe.fc.in_features, 10)

    # Full model (all layers trainable)
    model_full = models.resnet18(pretrained=True)
    model_full.fc = nn.Linear(model_full.fc.in_features, 10)

    # Count parameters
    trainable_fe = sum(p.numel() for p in model_fe.parameters() if p.requires_grad)
    trainable_full = sum(p.numel() for p in model_full.parameters() if p.requires_grad)

    print("Feature Extraction Model:")
    print(f"  Trainable parameters: {trainable_fe:,}")

    print("\nFull Training Model:")
    print(f"  Trainable parameters: {trainable_full:,}")

    print(f"\nSpeedup factor: {trainable_full / trainable_fe:.1f}x fewer parameters")
    print("\nAdvantages of Feature Extraction:")
    print("  + Faster training")
    print("  + Less memory usage")
    print("  + Less prone to overfitting")
    print("  + Good for small datasets")
    print("\nDisadvantages:")
    print("  - May not adapt well to very different domains")
    print("  - Limited flexibility")
    print()


def demonstrate_feature_reuse():
    """Demonstrate reusing features for multiple tasks."""
    print("7. Feature Reuse for Multiple Tasks")
    print("-" * 50)

    # Shared feature extractor
    resnet = models.resnet18(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False

    # Remove final layer
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    num_features = 512  # ResNet-18 feature dimension

    # Multiple task-specific heads
    classifier_task1 = nn.Linear(num_features, 10)  # 10 classes
    classifier_task2 = nn.Linear(num_features, 5)   # 5 classes
    regressor_task3 = nn.Linear(num_features, 1)    # Regression

    print("Shared feature extractor with multiple heads:")
    print(f"  Feature dimension: {num_features}")
    print(f"  Task 1 (classification): 10 classes")
    print(f"  Task 2 (classification): 5 classes")
    print(f"  Task 3 (regression): 1 output")

    # Sample inference
    sample_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        features = feature_extractor(sample_input).squeeze()
        out1 = classifier_task1(features)
        out2 = classifier_task2(features)
        out3 = regressor_task3(features)

    print(f"\nInput shape: {sample_input.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Task 1 output: {out1.shape}")
    print(f"Task 2 output: {out2.shape}")
    print(f"Task 3 output: {out3.shape}")
    print()


def main():
    print("=" * 60)
    print("FEATURE EXTRACTION WITH PRETRAINED MODELS")
    print("=" * 60)
    print()

    check_environment()

    # Create feature extractor
    model = create_feature_extractor()

    # Custom classifier
    custom_model = create_custom_classifier()

    # Extract features only
    feature_extractor = extract_features_only()

    # Create synthetic data
    train_loader = create_synthetic_data()

    # Train the model
    trained_model = train_feature_extractor(model, train_loader, epochs=2)

    # Compare training speeds
    compare_training_speed()

    # Demonstrate feature reuse
    demonstrate_feature_reuse()

    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Freeze pretrained layers with requires_grad=False")
    print("2. Replace final layer for new classification tasks")
    print("3. Feature extraction is faster and uses less memory")
    print("4. Good strategy for small datasets")
    print("5. Features can be reused for multiple tasks")
    print("=" * 60)


if __name__ == "__main__":
    main()
