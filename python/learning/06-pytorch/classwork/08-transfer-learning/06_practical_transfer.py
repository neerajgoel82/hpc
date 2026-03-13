"""
Practical Transfer Learning
Complete end-to-end transfer learning pipeline

This module demonstrates:
- Complete transfer learning workflow
- Data preparation and augmentation
- Model setup and training
- Evaluation and inference
- Saving and loading models
- Real-world best practices

Run: python 06_practical_transfer.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.models as models
import torchvision.transforms as transforms
import time


def check_environment():
    """Display PyTorch environment information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    return device


def create_transforms():
    """Create data augmentation transforms."""
    print("1. Data Preparation")
    print("-" * 50)

    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("Training augmentation:")
    print("  - Resize and random crop")
    print("  - Random horizontal flip")
    print("  - Random rotation (15 degrees)")
    print("  - Color jitter")
    print("  - ImageNet normalization")
    print()

    return train_transform, val_transform


def create_synthetic_dataset(num_samples=500, num_classes=5):
    """Create synthetic dataset for demonstration."""
    print("Creating synthetic dataset...")

    # Generate synthetic data
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))

    # Normalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = (images - mean) / std

    # Create dataset
    dataset = TensorDataset(images, labels)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"  Total samples: {num_samples}")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Number of classes: {num_classes}")
    print()

    return train_dataset, val_dataset, num_classes


def setup_model(num_classes, freeze_features=True):
    """Setup transfer learning model."""
    print("2. Model Setup")
    print("-" * 50)

    # Load pretrained ResNet-50
    model = models.resnet50(pretrained=True)
    print("Loaded pretrained ResNet-50")

    # Freeze feature extraction layers
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
        print("Froze all pretrained layers")

    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    print(f"\nReplaced classifier:")
    print(f"  Input features: {num_features}")
    print(f"  Hidden layer: 512")
    print(f"  Output classes: {num_classes}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter count:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    print()

    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = 100. * correct / total

    return val_loss, val_acc


def train_model(model, train_loader, val_loader, device, epochs=10):
    """Complete training loop."""
    print("3. Training")
    print("-" * 50)

    model = model.to(device)

    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    print(f"Training for {epochs} epochs...")
    print(f"Optimizer: Adam (lr=0.001)")
    print(f"Scheduler: ReduceLROnPlateau")
    print()

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  -> New best model (Val Acc: {val_acc:.2f}%)")

        print()

    print(f"Training complete! Best Val Acc: {best_val_acc:.2f}%")
    print()

    return model, history


def evaluate_model(model, val_loader, device, class_names=None):
    """Detailed model evaluation."""
    print("4. Evaluation")
    print("-" * 50)

    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = probabilities.max(1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())

    # Calculate accuracy
    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    accuracy = 100. * correct / len(all_labels)

    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"Total samples: {len(all_labels)}")
    print(f"Correct predictions: {correct}")

    # Average confidence
    import statistics
    avg_confidence = statistics.mean(all_confidences)
    print(f"Average confidence: {avg_confidence:.4f}")
    print()


def save_and_load_model(model, device):
    """Demonstrate saving and loading models."""
    print("5. Saving and Loading")
    print("-" * 50)

    # Save model
    save_path = 'transfer_learning_model.pth'

    # Save entire model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

    # Load model
    loaded_model = models.resnet50(pretrained=False)
    # Need to match architecture
    loaded_model.fc = model.fc
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    print(f"Model loaded from: {save_path}")
    print("Ready for inference!")
    print()


def inference_example(model, device):
    """Example inference on new data."""
    print("6. Inference Example")
    print("-" * 50)

    model.eval()

    # Create sample input
    sample_image = torch.randn(1, 3, 224, 224).to(device)

    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    sample_image = (sample_image - mean) / std

    print("Running inference on sample image...")

    with torch.no_grad():
        start_time = time.time()
        output = model(sample_image)
        inference_time = (time.time() - start_time) * 1000

        probabilities = torch.softmax(output[0], dim=0)
        confidence, prediction = probabilities.max(0)

    print(f"  Prediction: Class {prediction.item()}")
    print(f"  Confidence: {confidence.item():.4f}")
    print(f"  Inference time: {inference_time:.2f}ms")
    print()


def main():
    print("=" * 60)
    print("PRACTICAL TRANSFER LEARNING PIPELINE")
    print("=" * 60)
    print()

    # Setup
    device = check_environment()

    # Data preparation
    train_transform, val_transform = create_transforms()
    train_dataset, val_dataset, num_classes = create_synthetic_dataset()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Setup model
    model = setup_model(num_classes, freeze_features=True)

    # Train
    trained_model, history = train_model(
        model, train_loader, val_loader, device, epochs=5
    )

    # Evaluate
    evaluate_model(trained_model, val_loader, device)

    # Save and load
    save_and_load_model(trained_model, device)

    # Inference
    inference_example(trained_model, device)

    print("=" * 60)
    print("\nTransfer Learning Workflow Summary:")
    print("1. Prepare data with augmentation")
    print("2. Load pretrained model and modify classifier")
    print("3. Freeze pretrained layers (optional)")
    print("4. Train with appropriate learning rate")
    print("5. Evaluate and fine-tune if needed")
    print("6. Save model for deployment")
    print("=" * 60)


if __name__ == "__main__":
    main()
