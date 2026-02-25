"""
Model Zoo Overview
Survey of available pretrained models in torchvision

This module demonstrates:
- ResNet family (ResNet18, 34, 50, 101, 152)
- VGG family (VGG11, 13, 16, 19)
- MobileNet (lightweight models)
- EfficientNet (efficient architectures)
- Vision Transformers (ViT)
- Choosing the right model for your task

Run: python 05_model_zoo.py
"""

import torch
import torch.nn as nn
import torchvision.models as models


def check_environment():
    """Display PyTorch environment information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()


def resnet_family():
    """Explore the ResNet model family."""
    print("1. ResNet Family")
    print("-" * 50)

    resnet_models = {
        'ResNet-18': models.resnet18,
        'ResNet-34': models.resnet34,
        'ResNet-50': models.resnet50,
        'ResNet-101': models.resnet101,
        'ResNet-152': models.resnet152,
    }

    print(f"{'Model':<15} {'Params (M)':<12} {'Depth':<8} {'Top-1 Acc':<12} {'Use Case'}")
    print("-" * 75)

    for name, model_fn in resnet_models.items():
        model = model_fn(pretrained=False)
        params = sum(p.numel() for p in model.parameters()) / 1e6

        # Approximate accuracies and use cases
        info = {
            'ResNet-18': (70.0, 'Fast prototyping'),
            'ResNet-34': (73.3, 'Balanced'),
            'ResNet-50': (76.1, 'Standard choice'),
            'ResNet-101': (77.4, 'High accuracy'),
            'ResNet-152': (78.3, 'Best accuracy'),
        }

        acc, use_case = info[name]
        depth = name.split('-')[1]

        print(f"{name:<15} {params:<12.1f} {depth:<8} {acc:<12.1f}% {use_case}")

    print("\nKey features:")
    print("  - Residual connections (skip connections)")
    print("  - Deep architectures without vanishing gradients")
    print("  - Widely used baseline")
    print()


def vgg_family():
    """Explore the VGG model family."""
    print("2. VGG Family")
    print("-" * 50)

    vgg_models = {
        'VGG-11': models.vgg11,
        'VGG-13': models.vgg13,
        'VGG-16': models.vgg16,
        'VGG-19': models.vgg19,
    }

    print(f"{'Model':<15} {'Params (M)':<12} {'Layers':<8}")
    print("-" * 40)

    for name, model_fn in vgg_models.items():
        model = model_fn(pretrained=False)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        layers = name.split('-')[1]
        print(f"{name:<15} {params:<12.1f} {layers:<8}")

    print("\nKey features:")
    print("  - Simple architecture (conv + pool)")
    print("  - Very deep (up to 19 layers)")
    print("  - Large model size")
    print("  - Good for feature extraction")
    print()


def mobilenet_family():
    """Explore lightweight MobileNet models."""
    print("3. MobileNet Family (Lightweight)")
    print("-" * 50)

    mobile_models = {
        'MobileNet V2': models.mobilenet_v2,
        'MobileNet V3 Large': models.mobilenet_v3_large,
        'MobileNet V3 Small': models.mobilenet_v3_small,
    }

    print(f"{'Model':<20} {'Params (M)':<12} {'Size (MB)':<12}")
    print("-" * 50)

    for name, model_fn in mobile_models.items():
        model = model_fn(pretrained=False)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        size_mb = (params * 4) / 1e6
        print(f"{name:<20} {params:<12.1f} {size_mb:<12.1f}")

    print("\nKey features:")
    print("  - Designed for mobile/embedded devices")
    print("  - Depthwise separable convolutions")
    print("  - Small model size, fast inference")
    print("  - Good accuracy/size trade-off")
    print("\nUse cases:")
    print("  - Mobile apps")
    print("  - Edge devices")
    print("  - Real-time inference")
    print()


def efficientnet_family():
    """Explore EfficientNet models."""
    print("4. EfficientNet Family (Efficient)")
    print("-" * 50)

    # Note: efficientnet requires separate package
    print("EfficientNet models (B0-B7):")
    print("  - Systematically scaled (width, depth, resolution)")
    print("  - State-of-the-art accuracy/efficiency trade-off")
    print("  - EfficientNet-B0: 5.3M params, 77.1% top-1")
    print("  - EfficientNet-B7: 66M params, 84.3% top-1")
    print()

    print("To use EfficientNet:")
    print("  pip install timm")
    print("  import timm")
    print("  model = timm.create_model('efficientnet_b0', pretrained=True)")
    print()


def vision_transformer():
    """Explore Vision Transformer models."""
    print("5. Vision Transformers (ViT)")
    print("-" * 50)

    print("Vision Transformer (ViT):")
    print("  - Transformer architecture for images")
    print("  - Splits image into patches")
    print("  - Self-attention mechanism")
    print("  - Competitive with CNNs on large datasets")
    print()

    # Load a ViT model (if available in torchvision version)
    try:
        vit = models.vit_b_16(pretrained=False)
        params = sum(p.numel() for p in vit.parameters()) / 1e6
        print(f"ViT-B/16:")
        print(f"  Parameters: {params:.1f}M")
        print(f"  Input: 224x224 images")
        print(f"  Patch size: 16x16")
        print()
    except AttributeError:
        print("ViT models available in torchvision >= 0.12")
        print("Install with: pip install torchvision>=0.12")
        print()

    print("Use cases:")
    print("  - Large datasets (ImageNet-21k pretraining)")
    print("  - When you need attention mechanisms")
    print("  - State-of-the-art performance")
    print()


def model_selection_guide():
    """Guide for selecting the right model."""
    print("6. Model Selection Guide")
    print("-" * 50)

    print("Choose based on your constraints:\n")

    print("SPEED (Fast inference):")
    print("  1st: MobileNet V3 Small")
    print("  2nd: MobileNet V2")
    print("  3rd: ResNet-18")
    print()

    print("ACCURACY (Best performance):")
    print("  1st: ResNet-152 or ViT")
    print("  2nd: ResNet-101")
    print("  3rd: ResNet-50")
    print()

    print("MEMORY (Low memory usage):")
    print("  1st: MobileNet V3 Small")
    print("  2nd: MobileNet V2")
    print("  3rd: ResNet-18")
    print()

    print("BALANCED (Good trade-off):")
    print("  1st: ResNet-50")
    print("  2nd: ResNet-34")
    print("  3rd: MobileNet V2")
    print()

    print("FEATURE EXTRACTION:")
    print("  1st: VGG-16 (conv features)")
    print("  2nd: ResNet-50")
    print("  3rd: ResNet-101")
    print()


def compare_models_practical():
    """Practical comparison of popular models."""
    print("7. Practical Model Comparison")
    print("-" * 50)

    models_to_compare = [
        ('ResNet-18', models.resnet18),
        ('ResNet-50', models.resnet50),
        ('MobileNet V2', models.mobilenet_v2),
        ('VGG-16', models.vgg16),
    ]

    print(f"{'Model':<15} {'Params':<12} {'Memory (MB)':<12} {'Recommended For'}")
    print("-" * 70)

    recommendations = {
        'ResNet-18': 'Prototyping, small datasets',
        'ResNet-50': 'General purpose, production',
        'MobileNet V2': 'Mobile, edge devices',
        'VGG-16': 'Feature extraction',
    }

    for name, model_fn in models_to_compare:
        model = model_fn(pretrained=False)
        params = sum(p.numel() for p in model.parameters())
        memory_mb = (params * 4) / (1024 ** 2)

        print(f"{name:<15} {params:>11,} {memory_mb:>11.1f} {recommendations[name]}")

    print()


def load_and_inspect_model():
    """Load and inspect a model in detail."""
    print("8. Detailed Model Inspection")
    print("-" * 50)

    # Load ResNet-50
    model = models.resnet50(pretrained=True)

    print("ResNet-50 Architecture:")
    print(f"  Input: 3x224x224")
    print(f"  Conv1: 7x7, 64 filters")
    print(f"  MaxPool: 3x3")
    print(f"  Layer1: 3 blocks, 256 channels")
    print(f"  Layer2: 4 blocks, 512 channels")
    print(f"  Layer3: 6 blocks, 1024 channels")
    print(f"  Layer4: 3 blocks, 2048 channels")
    print(f"  AvgPool: Global")
    print(f"  FC: 2048 -> 1000 classes")
    print()

    # Count parameters by section
    print("Parameters by section:")
    sections = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
    for section in sections:
        if hasattr(model, section):
            params = sum(p.numel() for p in getattr(model, section).parameters())
            print(f"  {section:<10}: {params:>12,}")

    print()


def custom_model_modifications():
    """Show common model modifications."""
    print("9. Common Model Modifications")
    print("-" * 50)

    model = models.resnet50(pretrained=True)

    print("Modification 1: Change number of classes")
    original_fc = model.fc
    model.fc = nn.Linear(model.fc.in_features, 10)
    print(f"  Original: {original_fc.out_features} classes")
    print(f"  Modified: {model.fc.out_features} classes")
    print()

    print("Modification 2: Add dropout before classifier")
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, 10)
    )
    print("  Added dropout layer")
    print()

    print("Modification 3: Multi-layer classifier")
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 10)
    )
    print("  Added hidden layer with 512 units")
    print()


def main():
    print("=" * 60)
    print("PYTORCH MODEL ZOO OVERVIEW")
    print("=" * 60)
    print()

    check_environment()

    # Explore model families
    resnet_family()
    vgg_family()
    mobilenet_family()
    efficientnet_family()
    vision_transformer()

    # Model selection guide
    model_selection_guide()

    # Practical comparison
    compare_models_practical()

    # Detailed inspection
    load_and_inspect_model()

    # Common modifications
    custom_model_modifications()

    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. ResNet: Best general-purpose choice")
    print("2. MobileNet: For mobile/edge devices")
    print("3. VGG: Good for feature extraction")
    print("4. Choose based on speed/accuracy trade-off")
    print("5. Easy to modify architectures for your task")
    print("=" * 60)


if __name__ == "__main__":
    main()
