"""
Custom Architectures
Design custom neural network architectures

This module demonstrates:
- Building custom network architectures
- Residual connections (ResNet-style)
- Multi-branch architectures
- Skip connections and concatenation
- Flexible module design patterns

Run: python 01_custom_architectures.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity  # Skip connection
        out = F.relu(out)
        return out


class MultiBranchBlock(nn.Module):
    """Multi-branch block (Inception-style)."""

    def __init__(self, in_channels):
        super().__init__()
        # Branch 1: 1x1 conv
        self.branch1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        # Branch 2: 1x1 -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        # Branch 3: 1x1 -> 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
        )

        # Branch 4: MaxPool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Concatenate along channel dimension
        return torch.cat([b1, b2, b3, b4], dim=1)


class CustomResNet(nn.Module):
    """Custom ResNet-style architecture."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class UNetBlock(nn.Module):
    """U-Net style encoder-decoder block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def demonstrate_custom_architectures():
    """Demonstrate custom architecture building."""
    print("1. Custom Architectures")
    print("-" * 60)

    # ResidualBlock
    print("\nResidual Block:")
    res_block = ResidualBlock(64, 128, stride=2)
    x = torch.randn(2, 64, 32, 32)
    out = res_block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in res_block.parameters()):,}")

    # MultiBranchBlock
    print("\nMulti-Branch Block (Inception-style):")
    mb_block = MultiBranchBlock(128)
    x = torch.randn(2, 128, 16, 16)
    out = mb_block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Custom ResNet
    print("\nCustom ResNet:")
    model = CustomResNet(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def demonstrate_flexible_architecture():
    """Demonstrate flexible architecture patterns."""
    print("2. Flexible Architecture Patterns")
    print("-" * 60)

    class FlexibleNet(nn.Module):
        """Network with configurable depth and width."""

        def __init__(self, layers=[64, 128, 256], num_classes=10):
            super().__init__()
            self.encoder = nn.ModuleList()

            in_ch = 3
            for out_ch in layers:
                self.encoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    )
                )
                in_ch = out_ch

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(layers[-1], num_classes)

        def forward(self, x):
            for layer in self.encoder:
                x = layer(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    # Different configurations
    model_small = FlexibleNet([32, 64], num_classes=10)
    model_large = FlexibleNet([64, 128, 256, 512], num_classes=10)

    print(f"Small model parameters: {sum(p.numel() for p in model_small.parameters()):,}")
    print(f"Large model parameters: {sum(p.numel() for p in model_large.parameters()):,}")
    print()


def main():
    print("=" * 70)
    print("CUSTOM NEURAL NETWORK ARCHITECTURES")
    print("=" * 70)
    print()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    demonstrate_custom_architectures()
    demonstrate_flexible_architecture()

    print("=" * 70)


if __name__ == "__main__":
    main()
