"""
Pooling Layers - MaxPool and AvgPool Operations

This module demonstrates:
- Max pooling for spatial downsampling
- Average pooling for feature aggregation
- Adaptive pooling for fixed output size
- Effects on feature maps
- Pooling in CNN architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def create_feature_map():
    """Create a sample feature map for demonstration"""
    # 8x8 feature map with distinct regions
    feature = torch.zeros(1, 1, 8, 8)

    # Top-left: high values
    feature[0, 0, 0:4, 0:4] = torch.randn(4, 4) + 2.0

    # Top-right: medium values
    feature[0, 0, 0:4, 4:8] = torch.randn(4, 4) + 1.0

    # Bottom-left: low values
    feature[0, 0, 4:8, 0:4] = torch.randn(4, 4) - 1.0

    # Bottom-right: zero values
    feature[0, 0, 4:8, 4:8] = torch.randn(4, 4)

    return feature


def demonstrate_maxpool():
    """Demonstrate max pooling operations"""
    print("=" * 60)
    print("Max Pooling")
    print("=" * 60)

    feature = create_feature_map()
    print(f"Input feature map shape: {feature.shape}")
    print(f"Input values:\n{feature[0, 0]}")

    # Max pooling with kernel_size=2, stride=2
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    output = maxpool(feature)

    print(f"\nMaxPool2d(kernel_size=2, stride=2):")
    print(f"Output shape: {output.shape}")
    print(f"Output values:\n{output[0, 0]}")
    print(f"Size reduction: {feature.shape[2:]} -> {output.shape[2:]}")

    # Max pooling with different kernel sizes
    print("\n" + "-" * 60)
    print("Different Kernel Sizes:")
    for k in [2, 3, 4]:
        pool = nn.MaxPool2d(kernel_size=k, stride=k)
        out = pool(feature)
        print(f"  kernel_size={k}: {feature.shape[2:]} -> {out.shape[2:]}")

    return feature, output


def demonstrate_avgpool():
    """Demonstrate average pooling operations"""
    print("\n" + "=" * 60)
    print("Average Pooling")
    print("=" * 60)

    feature = create_feature_map()
    print(f"Input feature map shape: {feature.shape}")

    # Average pooling with kernel_size=2, stride=2
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    output = avgpool(feature)

    print(f"\nAvgPool2d(kernel_size=2, stride=2):")
    print(f"Output shape: {output.shape}")
    print(f"Output values:\n{output[0, 0]}")

    # Compare max vs average pooling
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    max_out = maxpool(feature)

    print(f"\n" + "-" * 60)
    print("Max vs Average Pooling Comparison:")
    print(f"Max pooling output range: [{max_out.min():.3f}, {max_out.max():.3f}]")
    print(f"Avg pooling output range: [{output.min():.3f}, {output.max():.3f}]")

    return output


def demonstrate_adaptive_pooling():
    """Demonstrate adaptive pooling for fixed output size"""
    print("\n" + "=" * 60)
    print("Adaptive Pooling")
    print("=" * 60)

    # Different input sizes
    input_sizes = [(1, 1, 32, 32), (1, 1, 64, 64), (1, 1, 128, 128)]
    target_size = (7, 7)

    adaptive_maxpool = nn.AdaptiveMaxPool2d(target_size)
    adaptive_avgpool = nn.AdaptiveAvgPool2d(target_size)

    print(f"Target output size: {target_size}")
    print("\n" + "-" * 60)

    for input_shape in input_sizes:
        x = torch.randn(input_shape)
        max_out = adaptive_maxpool(x)
        avg_out = adaptive_avgpool(x)

        print(f"Input: {input_shape[2:]} -> Output: {max_out.shape[2:]}")

    print("\n" + "-" * 60)
    print("Use case: Flexible input sizes for classification")
    print("Example: ResNet uses AdaptiveAvgPool2d((1, 1)) before FC layer")


def demonstrate_pooling_properties():
    """Demonstrate key properties of pooling layers"""
    print("\n" + "=" * 60)
    print("Pooling Layer Properties")
    print("=" * 60)

    feature = torch.randn(1, 1, 8, 8)

    # Property 1: No learnable parameters
    maxpool = nn.MaxPool2d(2)
    print(f"\n1. No Learnable Parameters:")
    print(f"   MaxPool2d parameters: {sum(p.numel() for p in maxpool.parameters())}")

    # Property 2: Translation invariance
    print(f"\n2. Translation Invariance:")
    shifted = torch.roll(feature, shifts=(1, 1), dims=(2, 3))
    out1 = maxpool(feature)
    out2 = maxpool(shifted)
    print(f"   Original and shifted inputs produce similar pooled features")

    # Property 3: Spatial downsampling
    print(f"\n3. Spatial Downsampling:")
    print(f"   Input: {feature.shape[2:]}, Output: {out1.shape[2:]}")
    print(f"   Reduces spatial dimensions by factor of {feature.shape[2] // out1.shape[2]}")

    # Property 4: Channel-wise operation
    multi_channel = torch.randn(1, 64, 8, 8)
    out_multi = maxpool(multi_channel)
    print(f"\n4. Channel-wise Operation:")
    print(f"   Input channels: {multi_channel.shape[1]}")
    print(f"   Output channels: {out_multi.shape[1]}")
    print(f"   Each channel pooled independently")


def demonstrate_stride_vs_kernel():
    """Demonstrate stride vs kernel_size effects"""
    print("\n" + "=" * 60)
    print("Stride vs Kernel Size")
    print("=" * 60)

    feature = create_feature_map()

    # Same kernel, different stride
    print("\nSame kernel_size=2, different stride:")
    for stride in [1, 2]:
        pool = nn.MaxPool2d(kernel_size=2, stride=stride)
        output = pool(feature)
        print(f"  stride={stride}: {feature.shape[2:]} -> {output.shape[2:]}")

    # Same stride, different kernel
    print("\nSame stride=2, different kernel_size:")
    for kernel in [2, 3]:
        pool = nn.MaxPool2d(kernel_size=kernel, stride=2)
        output = pool(feature)
        print(f"  kernel_size={kernel}: {feature.shape[2:]} -> {output.shape[2:]}")

    # Overlapping pools (kernel > stride)
    print("\nOverlapping pools (kernel_size > stride):")
    pool = nn.MaxPool2d(kernel_size=3, stride=1)
    output = pool(feature)
    print(f"  kernel_size=3, stride=1: {feature.shape[2:]} -> {output.shape[2:]}")
    print(f"  Creates overlapping receptive fields")


def demonstrate_pooling_in_cnn():
    """Demonstrate pooling usage in CNN architecture"""
    print("\n" + "=" * 60)
    print("Pooling in CNN Architecture")
    print("=" * 60)

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            print(f"  Input: {x.shape}")
            x = F.relu(self.conv1(x))
            print(f"  After Conv1: {x.shape}")
            x = self.pool1(x)
            print(f"  After Pool1: {x.shape}")
            x = F.relu(self.conv2(x))
            print(f"  After Conv2: {x.shape}")
            x = self.pool2(x)
            print(f"  After Pool2: {x.shape}")
            x = F.relu(self.conv3(x))
            print(f"  After Conv3: {x.shape}")
            x = self.adaptive_pool(x)
            print(f"  After AdaptivePool: {x.shape}")
            return x

    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)

    print("\nForward pass through CNN with pooling:")
    output = model(x)


def visualize_pooling_effects():
    """Visualize pooling effects on feature maps"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Pooling Operations Comparison', fontsize=16)

    # Create feature map
    feature = create_feature_map()

    # Different pooling operations
    maxpool2 = nn.MaxPool2d(2, 2)
    avgpool2 = nn.AvgPool2d(2, 2)
    maxpool3 = nn.MaxPool2d(3, 2)
    avgpool3 = nn.AvgPool2d(3, 2)
    adaptive = nn.AdaptiveMaxPool2d((4, 4))

    operations = [
        ('Original', feature),
        ('MaxPool 2x2', maxpool2(feature)),
        ('AvgPool 2x2', avgpool2(feature)),
        ('MaxPool 3x3', maxpool3(feature)),
        ('AvgPool 3x3', avgpool3(feature)),
        ('Adaptive 4x4', adaptive(feature)),
    ]

    for idx, (name, img) in enumerate(operations):
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(img[0, 0].detach().numpy(), cmap='viridis')
        axes[row, col].set_title(f'{name}\n{img.shape[2:]}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('/tmp/pooling_operations.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to /tmp/pooling_operations.png")
    plt.close()


def main():
    """Main demonstration function"""
    print("\n" + "=" * 60)
    print("POOLING LAYERS IN PYTORCH")
    print("=" * 60)

    # Max pooling
    feature, output = demonstrate_maxpool()

    # Average pooling
    demonstrate_avgpool()

    # Adaptive pooling
    demonstrate_adaptive_pooling()

    # Pooling properties
    demonstrate_pooling_properties()

    # Stride vs kernel size
    demonstrate_stride_vs_kernel()

    # Pooling in CNN
    demonstrate_pooling_in_cnn()

    # Visualize effects
    visualize_pooling_effects()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Pooling reduces spatial dimensions (downsampling)")
    print("2. MaxPool captures strongest activations")
    print("3. AvgPool smooths features")
    print("4. Adaptive pooling handles variable input sizes")
    print("5. No learnable parameters in pooling layers")
    print("6. Provides translation invariance")
    print("=" * 60)


if __name__ == "__main__":
    main()
