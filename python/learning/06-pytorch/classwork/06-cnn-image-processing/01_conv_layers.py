"""
Convolutional Layers - 2D Convolution Operations

This module demonstrates:
- 2D convolution operations
- Kernel/filter concepts
- Padding strategies (valid, same, custom)
- Stride effects on output size
- Multiple input/output channels
- Visualizing convolution results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def create_sample_image():
    """Create a simple test image with patterns"""
    img = torch.zeros(1, 1, 28, 28)

    # Add horizontal lines
    img[0, 0, 10:12, :] = 1.0

    # Add vertical lines
    img[0, 0, :, 15:17] = 1.0

    # Add diagonal pattern
    for i in range(8):
        img[0, 0, 5+i, 5+i] = 1.0

    return img


def demonstrate_basic_conv():
    """Demonstrate basic 2D convolution operation"""
    print("=" * 60)
    print("Basic 2D Convolution")
    print("=" * 60)

    # Create input image: (batch, channels, height, width)
    img = create_sample_image()
    print(f"Input shape: {img.shape}")

    # Define convolution layer
    # in_channels=1, out_channels=1, kernel_size=3
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

    # Set custom kernel (edge detection)
    edge_kernel = torch.tensor([
        [[-1, -1, -1],
         [-1,  8, -1],
         [-1, -1, -1]]
    ], dtype=torch.float32).unsqueeze(0)

    conv.weight.data = edge_kernel

    # Apply convolution
    output = conv(img)
    print(f"Output shape: {output.shape}")
    print(f"Output size reduction: {img.shape[2:]} -> {output.shape[2:]}")

    return img, output


def demonstrate_padding():
    """Demonstrate different padding strategies"""
    print("\n" + "=" * 60)
    print("Padding Strategies")
    print("=" * 60)

    img = create_sample_image()
    kernel_size = 3

    # Valid padding (no padding)
    conv_valid = nn.Conv2d(1, 1, kernel_size, padding=0)
    out_valid = conv_valid(img)
    print(f"\nValid padding (padding=0):")
    print(f"  Input: {img.shape[2:]}, Output: {out_valid.shape[2:]}")

    # Same padding (preserve size)
    # For kernel_size=3, padding=1 gives same size
    conv_same = nn.Conv2d(1, 1, kernel_size, padding=1)
    out_same = conv_same(img)
    print(f"\nSame padding (padding=1):")
    print(f"  Input: {img.shape[2:]}, Output: {out_same.shape[2:]}")

    # Custom padding
    conv_custom = nn.Conv2d(1, 1, kernel_size, padding=2)
    out_custom = conv_custom(img)
    print(f"\nCustom padding (padding=2):")
    print(f"  Input: {img.shape[2:]}, Output: {out_custom.shape[2:]}")

    # Padding modes
    conv_reflect = nn.Conv2d(1, 1, kernel_size, padding=1, padding_mode='reflect')
    out_reflect = conv_reflect(img)
    print(f"\nReflect padding mode:")
    print(f"  Output: {out_reflect.shape[2:]}")

    return out_valid, out_same, out_custom


def demonstrate_stride():
    """Demonstrate stride effects on output size"""
    print("\n" + "=" * 60)
    print("Stride Effects")
    print("=" * 60)

    img = create_sample_image()
    kernel_size = 3
    padding = 1

    for stride in [1, 2, 3]:
        conv = nn.Conv2d(1, 1, kernel_size, stride=stride, padding=padding)
        output = conv(img)

        print(f"\nStride={stride}:")
        print(f"  Input: {img.shape[2:]}")
        print(f"  Output: {output.shape[2:]}")
        print(f"  Downsampling factor: {img.shape[2] / output.shape[2]:.2f}x")


def demonstrate_multiple_channels():
    """Demonstrate multiple input and output channels"""
    print("\n" + "=" * 60)
    print("Multiple Channels")
    print("=" * 60)

    # RGB image simulation (3 channels)
    img_rgb = torch.randn(1, 3, 28, 28)
    print(f"Input RGB image: {img_rgb.shape}")

    # Conv layer with multiple input and output channels
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    output = conv(img_rgb)

    print(f"\nConv2d(3, 16, kernel_size=3):")
    print(f"  Number of kernels: 16 (one per output channel)")
    print(f"  Each kernel shape: (3, 3, 3) - 3 input channels")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {conv.weight.shape} + bias {conv.bias.shape}")
    print(f"  Total params: {conv.weight.numel() + conv.bias.numel()}")


def demonstrate_custom_kernels():
    """Demonstrate various image processing kernels"""
    print("\n" + "=" * 60)
    print("Custom Convolution Kernels")
    print("=" * 60)

    img = create_sample_image()

    # Define custom kernels
    kernels = {
        'Edge Detection': torch.tensor([
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]]
        ], dtype=torch.float32).unsqueeze(0),

        'Horizontal Edges': torch.tensor([
            [[-1, -1, -1],
             [ 0,  0,  0],
             [ 1,  1,  1]]
        ], dtype=torch.float32).unsqueeze(0),

        'Vertical Edges': torch.tensor([
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0),

        'Blur': torch.tensor([
            [[1/9, 1/9, 1/9],
             [1/9, 1/9, 1/9],
             [1/9, 1/9, 1/9]]
        ], dtype=torch.float32).unsqueeze(0),

        'Sharpen': torch.tensor([
            [[ 0, -1,  0],
             [-1,  5, -1],
             [ 0, -1,  0]]
        ], dtype=torch.float32).unsqueeze(0),
    }

    results = {}
    for name, kernel in kernels.items():
        conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        conv.weight.data = kernel
        output = conv(img)
        results[name] = output
        print(f"\n{name}:")
        print(f"  Kernel:\n{kernel[0, 0]}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    return img, results


def calculate_output_size(input_size, kernel_size, stride, padding):
    """Calculate output size of convolution"""
    output_size = ((input_size - kernel_size + 2 * padding) // stride) + 1
    return output_size


def demonstrate_output_size_formula():
    """Demonstrate output size calculation formula"""
    print("\n" + "=" * 60)
    print("Output Size Formula")
    print("=" * 60)

    print("\nFormula: output_size = ((input_size - kernel_size + 2*padding) // stride) + 1")

    test_cases = [
        (28, 3, 1, 0),  # MNIST with valid padding
        (28, 3, 1, 1),  # MNIST with same padding
        (32, 5, 2, 2),  # CIFAR with stride 2
        (224, 7, 2, 3), # ImageNet first layer
    ]

    print("\nExamples:")
    for input_size, kernel_size, stride, padding in test_cases:
        output_size = calculate_output_size(input_size, kernel_size, stride, padding)
        print(f"  Input={input_size}, K={kernel_size}, S={stride}, P={padding} -> Output={output_size}")


def visualize_convolutions(img, results):
    """Visualize original image and convolution results"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Convolution with Different Kernels', fontsize=16)

    # Plot original
    axes[0, 0].imshow(img[0, 0].detach().numpy(), cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Plot results
    items = list(results.items())
    for idx, (name, output) in enumerate(items[:5], start=1):
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(output[0, 0].detach().numpy(), cmap='gray')
        axes[row, col].set_title(name)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('/tmp/convolution_kernels.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to /tmp/convolution_kernels.png")
    plt.close()


def main():
    """Main demonstration function"""
    print("\n" + "=" * 60)
    print("2D CONVOLUTION OPERATIONS IN PYTORCH")
    print("=" * 60)

    # Basic convolution
    img, output = demonstrate_basic_conv()

    # Padding strategies
    demonstrate_padding()

    # Stride effects
    demonstrate_stride()

    # Multiple channels
    demonstrate_multiple_channels()

    # Custom kernels
    img, results = demonstrate_custom_kernels()

    # Output size formula
    demonstrate_output_size_formula()

    # Visualize results
    visualize_convolutions(img, results)

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Convolutions detect local patterns using learned kernels")
    print("2. Padding controls output size (valid/same/custom)")
    print("3. Stride controls downsampling factor")
    print("4. Multiple channels allow detecting many patterns")
    print("5. Different kernels detect different features")
    print("=" * 60)


if __name__ == "__main__":
    main()
