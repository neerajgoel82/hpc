"""
Domain Adaptation
Adapting pretrained models to new domains

This module demonstrates:
- Understanding domain shift
- Data augmentation for domain adaptation
- Fine-tuning strategies for different domains
- Medical imaging example
- Satellite imagery example
- When pretrained models need adaptation

Run: python 04_domain_adaptation.py
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


def understand_domain_shift():
    """Explain domain shift and its impact."""
    print("1. Understanding Domain Shift")
    print("-" * 50)

    print("What is Domain Shift?")
    print("  Source domain: Where the model was trained (e.g., ImageNet)")
    print("  Target domain: Where we want to use it (e.g., Medical images)")
    print()

    print("Examples of Domain Shift:")
    print("  ImageNet -> Medical Images:")
    print("    - Different image statistics")
    print("    - Different color distributions")
    print("    - Different textures and patterns")
    print()

    print("  Natural Images -> Satellite Images:")
    print("    - Different viewpoint (top-down)")
    print("    - Different scale")
    print("    - Different color channels (may include IR)")
    print()

    print("  RGB Images -> Grayscale X-rays:")
    print("    - Different number of channels")
    print("    - Different contrast patterns")
    print("    - Different semantic meaning")
    print()

    print("Impact on Model Performance:")
    print("  - Pretrained features may not be optimal")
    print("  - Early layers may need more adaptation")
    print("  - Larger domain gap = more fine-tuning needed")
    print()


def medical_imaging_adaptation():
    """Adapt model for medical imaging domain."""
    print("2. Medical Imaging Domain Adaptation")
    print("-" * 50)

    # Load pretrained model
    model = models.resnet18(pretrained=True)

    print("Challenge: Adapting ImageNet model to X-ray images")
    print()

    # Option 1: Modify first layer for grayscale
    print("Option 1: Grayscale input (1 channel)")
    original_conv = model.conv1
    print(f"  Original conv1: {original_conv.in_channels} input channels")

    # Create new conv layer for single channel
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Initialize with average of RGB weights
    with torch.no_grad():
        model.conv1.weight = nn.Parameter(
            original_conv.weight.mean(dim=1, keepdim=True)
        )

    print(f"  New conv1: {model.conv1.in_channels} input channel")
    print()

    # Option 2: Keep RGB by replicating grayscale
    print("Option 2: Replicate grayscale to 3 channels")
    print("  Convert grayscale X-ray to 3 channels (repeat)")
    print("  Keep original conv1 layer")
    print()

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary: Normal/Abnormal

    print("Adaptation strategy:")
    print("  1. Modify input layer for channel compatibility")
    print("  2. Use aggressive data augmentation")
    print("  3. Fine-tune with small learning rate")
    print("  4. May need to unfreeze more layers due to domain gap")
    print()

    return model


def satellite_imaging_adaptation():
    """Adapt model for satellite imaging."""
    print("3. Satellite Imaging Domain Adaptation")
    print("-" * 50)

    model = models.resnet18(pretrained=True)

    print("Challenge: Top-down satellite view vs ground-level ImageNet")
    print()

    # Satellite images might have 4+ channels (RGB + IR)
    print("Handling multi-spectral data:")
    print("  Option 1: Use only RGB channels")
    print("  Option 2: Expand conv1 for 4+ channels")
    print()

    # Expand first conv layer
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(
        4, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Initialize new channel weights
    with torch.no_grad():
        # Copy RGB weights
        model.conv1.weight[:, :3, :, :] = original_conv.weight
        # Initialize IR channel randomly (or with mean)
        model.conv1.weight[:, 3:, :, :] = original_conv.weight.mean(dim=1, keepdim=True)

    print(f"Modified conv1 to accept 4 channels (RGB + IR)")
    print()

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 land cover classes

    print("Adaptation considerations:")
    print("  - Scale differences (zoomed out view)")
    print("  - Rotation invariance needed")
    print("  - Different texture patterns")
    print()

    return model


def domain_specific_augmentation():
    """Define domain-specific data augmentation."""
    print("4. Domain-Specific Augmentation")
    print("-" * 50)

    print("Medical Imaging Augmentation:")
    medical_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),  # Small rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale
    ])
    print("  - Small rotations (anatomical constraints)")
    print("  - Brightness/contrast adjustment")
    print("  - Horizontal flips (when appropriate)")
    print()

    print("Satellite Imaging Augmentation:")
    satellite_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),  # Full rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
    ])
    print("  - Full rotation (no up direction)")
    print("  - Horizontal and vertical flips")
    print("  - Color jittering")
    print()

    return medical_transform, satellite_transform


def training_strategy_comparison():
    """Compare training strategies for different domain gaps."""
    print("5. Training Strategies by Domain Gap")
    print("-" * 50)

    strategies = {
        "Small Gap (Cats vs Dogs)": {
            "description": "Target similar to ImageNet",
            "strategy": "Feature extraction or minimal fine-tuning",
            "layers_to_train": "Final layer only",
            "learning_rate": "1e-3",
            "epochs": "10-20"
        },
        "Medium Gap (Medical grayscale)": {
            "description": "Different modality, similar structure",
            "strategy": "Selective fine-tuning",
            "layers_to_train": "Last 2-3 blocks + classifier",
            "learning_rate": "1e-4 (base), 1e-3 (classifier)",
            "epochs": "30-50"
        },
        "Large Gap (Satellite)": {
            "description": "Different perspective and scale",
            "strategy": "Full fine-tuning",
            "layers_to_train": "All layers",
            "learning_rate": "1e-5 (base), 1e-4 (classifier)",
            "epochs": "50-100"
        }
    }

    for domain, info in strategies.items():
        print(f"{domain}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()


def adapt_model_architecture(domain_type='medical'):
    """Demonstrate full adaptation pipeline."""
    print("6. Complete Adaptation Pipeline")
    print("-" * 50)

    print(f"Adapting for: {domain_type} domain")
    print()

    # Load pretrained model
    model = models.resnet18(pretrained=True)

    if domain_type == 'medical':
        # Modify for grayscale
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_classes = 2
        print("  Modified input: 1 channel (grayscale)")

    elif domain_type == 'satellite':
        # Modify for multi-spectral
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_classes = 10
        print("  Modified input: 4 channels (RGB + IR)")

    else:
        num_classes = 10
        print("  Using standard RGB input")

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze early layers (domain-dependent)
    if domain_type == 'medical':
        # Freeze less due to larger domain gap
        for name, param in model.named_parameters():
            if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print()

    return model


def evaluate_domain_adaptation():
    """Evaluate effectiveness of domain adaptation."""
    print("7. Evaluating Domain Adaptation")
    print("-" * 50)

    print("Metrics to track:")
    print("  1. Training loss/accuracy")
    print("  2. Validation loss/accuracy")
    print("  3. Domain-specific metrics:")
    print("     - Medical: Sensitivity, Specificity, AUC")
    print("     - Satellite: IoU, F1-score per class")
    print()

    print("Signs of successful adaptation:")
    print("  + Validation accuracy improves")
    print("  + Gap between train/val narrows")
    print("  + Domain-specific metrics improve")
    print()

    print("Signs of poor adaptation:")
    print("  - Large train/val gap (overfitting)")
    print("  - Unstable training")
    print("  - Poor performance on target domain")
    print()


def main():
    print("=" * 60)
    print("DOMAIN ADAPTATION FOR TRANSFER LEARNING")
    print("=" * 60)
    print()

    check_environment()

    # Understand domain shift
    understand_domain_shift()

    # Medical imaging adaptation
    medical_model = medical_imaging_adaptation()

    # Satellite imaging adaptation
    satellite_model = satellite_imaging_adaptation()

    # Domain-specific augmentation
    medical_aug, satellite_aug = domain_specific_augmentation()

    # Training strategies
    training_strategy_comparison()

    # Complete adaptation
    adapted_model = adapt_model_architecture('medical')

    # Evaluation
    evaluate_domain_adaptation()

    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Domain shift affects pretrained model performance")
    print("2. Larger domain gap requires more fine-tuning")
    print("3. Adapt input layers for different modalities")
    print("4. Use domain-specific data augmentation")
    print("5. Monitor both standard and domain-specific metrics")
    print("=" * 60)


if __name__ == "__main__":
    main()
