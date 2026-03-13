"""
Pretrained Models
Loading and using pretrained models from torchvision.models

This module demonstrates:
- Loading pretrained models from torchvision
- Understanding model architectures
- Making predictions with pretrained models
- Inspecting model structure and layers
- Using models for inference

Run: python 01_pretrained_models.py
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import json


def check_environment():
    """Display PyTorch environment information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()


def load_pretrained_resnet():
    """Load a pretrained ResNet model."""
    print("1. Loading Pretrained ResNet-18")
    print("-" * 50)

    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=True)

    # Set to evaluation mode (important!)
    model.eval()

    print(f"Model type: {type(model).__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()

    return model


def inspect_model_structure(model):
    """Inspect the structure of a pretrained model."""
    print("2. Model Structure")
    print("-" * 50)

    # Print model architecture
    print("Model architecture overview:")
    print(model)
    print()

    # Inspect specific layers
    print("\nFirst layer (conv1):")
    print(model.conv1)

    print("\nLast layer (fc):")
    print(model.fc)
    print(f"Output features: {model.fc.out_features}")
    print()


def explore_different_models():
    """Load and compare different pretrained models."""
    print("3. Different Pretrained Models")
    print("-" * 50)

    models_dict = {
        'ResNet-18': models.resnet18(pretrained=True),
        'ResNet-50': models.resnet50(pretrained=True),
        'VGG-16': models.vgg16(pretrained=True),
        'MobileNet V2': models.mobilenet_v2(pretrained=True),
    }

    print(f"{'Model':<15} {'Parameters':<15} {'Size (MB)':<15}")
    print("-" * 50)

    for name, model in models_dict.items():
        num_params = sum(p.numel() for p in model.parameters())
        # Estimate size in MB (4 bytes per float32 parameter)
        size_mb = (num_params * 4) / (1024 ** 2)
        print(f"{name:<15} {num_params:>13,} {size_mb:>13.2f}")

    print()
    return models_dict['ResNet-18']


def prepare_input_image():
    """Prepare an image for inference."""
    print("4. Preparing Input Image")
    print("-" * 50)

    # Define ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("ImageNet preprocessing pipeline:")
    print("1. Resize to 256x256")
    print("2. Center crop to 224x224")
    print("3. Convert to tensor")
    print("4. Normalize with ImageNet mean/std")
    print()

    # Create a sample image (random for demonstration)
    # In practice, you would load: Image.open('image.jpg')
    sample_image = Image.new('RGB', (400, 400), color='blue')
    input_tensor = preprocess(sample_image)

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor dtype: {input_tensor.dtype}")
    print(f"Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)
    print(f"Batch shape: {input_batch.shape}")
    print()

    return input_batch, preprocess


def make_predictions(model, input_batch):
    """Make predictions with a pretrained model."""
    print("5. Making Predictions")
    print("-" * 50)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_batch = input_batch.to(device)

    print(f"Running inference on: {device}")

    # Inference
    with torch.no_grad():
        output = model(input_batch)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    print(f"\nProbabilities shape: {probabilities.shape}")
    print(f"Sum of probabilities: {probabilities.sum():.6f}")

    # Top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    print("\nTop 5 predictions:")
    for i in range(5):
        print(f"  {i+1}. Class {top5_catid[i].item()}: {top5_prob[i].item():.4f}")

    print()
    return output, probabilities


def freeze_and_inspect_params(model):
    """Demonstrate freezing and inspecting parameters."""
    print("6. Freezing Parameters")
    print("-" * 50)

    # Count parameters before freezing
    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Before freezing:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_before:,}")

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nAfter freezing:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_after:,}")

    # Unfreeze last layer
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable_final = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nAfter unfreezing final layer:")
    print(f"  Trainable parameters: {trainable_final:,}")
    print()


def layer_by_layer_inspection(model):
    """Inspect model layer by layer."""
    print("7. Layer-by-Layer Inspection")
    print("-" * 50)

    print("Model layers:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<10}: {type(module).__name__:<20} ({num_params:>10,} params)")

    print()


def main():
    print("=" * 60)
    print("PRETRAINED MODELS IN PYTORCH")
    print("=" * 60)
    print()

    check_environment()

    # Load pretrained model
    model = load_pretrained_resnet()

    # Inspect structure
    inspect_model_structure(model)

    # Explore different models
    explore_different_models()

    # Prepare input
    input_batch, preprocess = prepare_input_image()

    # Make predictions
    output, probabilities = make_predictions(model, input_batch)

    # Freeze parameters
    freeze_and_inspect_params(model)

    # Layer inspection
    layer_by_layer_inspection(model)

    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Pretrained models are easy to load with torchvision.models")
    print("2. Always set model.eval() for inference")
    print("3. ImageNet models expect 224x224 normalized images")
    print("4. Use softmax to convert logits to probabilities")
    print("5. Parameters can be frozen with requires_grad=False")
    print("=" * 60)


if __name__ == "__main__":
    main()
