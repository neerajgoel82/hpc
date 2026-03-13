"""
Linear Layers
Fully connected layers and weight initialization

This module demonstrates:
- Linear/Dense layers
- Weight initialization strategies
- Xavier/Glorot initialization
- He initialization
- Custom initialization
- Bias handling

Run: python 02_linear_layers.py
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math


class LinearLayerDemo(nn.Module):
    """Basic linear layer demonstration"""

    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MultiLayerPerceptron(nn.Module):
    """MLP with multiple linear layers"""

    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CustomInitModel(nn.Module):
    """Model with custom weight initialization"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def initialize_weights(self, method: str = 'xavier'):
        """Initialize weights with specified method"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == 'xavier':
                    init.xavier_uniform_(module.weight)
                elif method == 'xavier_normal':
                    init.xavier_normal_(module.weight)
                elif method == 'he':
                    init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif method == 'he_normal':
                    init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif method == 'zeros':
                    init.zeros_(module.weight)
                elif method == 'ones':
                    init.ones_(module.weight)
                elif method == 'uniform':
                    init.uniform_(module.weight, -0.1, 0.1)
                elif method == 'normal':
                    init.normal_(module.weight, mean=0.0, std=0.01)

                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


def demonstrate_basic_linear():
    """Basic linear layer operations"""
    print("\n1. Basic Linear Layer")
    print("-" * 60)

    # Create linear layer
    linear = nn.Linear(in_features=10, out_features=5)

    print(f"Input features: {linear.in_features}")
    print(f"Output features: {linear.out_features}")
    print(f"Weight shape: {linear.weight.shape}")
    print(f"Bias shape: {linear.bias.shape}")

    # Forward pass
    x = torch.randn(3, 10)  # batch_size=3, features=10
    output = linear(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Mathematical operation: y = xW^T + b
    manual_output = torch.matmul(x, linear.weight.t()) + linear.bias
    print(f"\nOutputs match: {torch.allclose(output, manual_output)}")


def demonstrate_no_bias():
    """Linear layer without bias"""
    print("\n2. Linear Layer Without Bias")
    print("-" * 60)

    linear_with_bias = nn.Linear(5, 3, bias=True)
    linear_no_bias = nn.Linear(5, 3, bias=False)

    print(f"With bias: {linear_with_bias.bias is not None}")
    print(f"Without bias: {linear_no_bias.bias is not None}")

    x = torch.randn(2, 5)
    out1 = linear_with_bias(x)
    out2 = linear_no_bias(x)

    print(f"\nOutput with bias: {out1[0]}")
    print(f"Output without bias: {out2[0]}")


def demonstrate_weight_inspection():
    """Inspect weight values"""
    print("\n3. Weight Inspection")
    print("-" * 60)

    linear = nn.Linear(4, 3)

    print("Default initialization:")
    print(f"Weight:\n{linear.weight.data}")
    print(f"Bias:\n{linear.bias.data}")

    print(f"\nWeight statistics:")
    print(f"  Mean: {linear.weight.data.mean().item():.6f}")
    print(f"  Std:  {linear.weight.data.std().item():.6f}")
    print(f"  Min:  {linear.weight.data.min().item():.6f}")
    print(f"  Max:  {linear.weight.data.max().item():.6f}")


def demonstrate_xavier_initialization():
    """Xavier/Glorot initialization"""
    print("\n4. Xavier Initialization")
    print("-" * 60)

    # Xavier is good for sigmoid/tanh activations
    linear = nn.Linear(100, 50)

    # Xavier uniform
    init.xavier_uniform_(linear.weight)
    print("Xavier Uniform:")
    print(f"  Mean: {linear.weight.data.mean().item():.6f}")
    print(f"  Std:  {linear.weight.data.std().item():.6f}")
    expected_std = math.sqrt(2.0 / (100 + 50))
    print(f"  Expected std: {expected_std:.6f}")

    # Xavier normal
    init.xavier_normal_(linear.weight)
    print("\nXavier Normal:")
    print(f"  Mean: {linear.weight.data.mean().item():.6f}")
    print(f"  Std:  {linear.weight.data.std().item():.6f}")


def demonstrate_he_initialization():
    """He/Kaiming initialization"""
    print("\n5. He Initialization")
    print("-" * 60)

    # He is good for ReLU activations
    linear = nn.Linear(100, 50)

    # He uniform
    init.kaiming_uniform_(linear.weight, nonlinearity='relu')
    print("He Uniform:")
    print(f"  Mean: {linear.weight.data.mean().item():.6f}")
    print(f"  Std:  {linear.weight.data.std().item():.6f}")

    # He normal
    init.kaiming_normal_(linear.weight, nonlinearity='relu')
    print("\nHe Normal:")
    print(f"  Mean: {linear.weight.data.mean().item():.6f}")
    print(f"  Std:  {linear.weight.data.std().item():.6f}")
    expected_std = math.sqrt(2.0 / 100)
    print(f"  Expected std: {expected_std:.6f}")


def demonstrate_custom_initialization():
    """Custom initialization methods"""
    print("\n6. Custom Initialization")
    print("-" * 60)

    model = CustomInitModel(10, 20, 5)

    methods = ['xavier', 'he', 'uniform', 'normal']
    for method in methods:
        model.initialize_weights(method)
        weight_mean = model.layer1.weight.data.mean().item()
        weight_std = model.layer1.weight.data.std().item()
        print(f"{method:15} - Mean: {weight_mean:8.6f}, Std: {weight_std:.6f}")


def demonstrate_sequential_layers():
    """Multiple linear layers in sequence"""
    print("\n7. Sequential Linear Layers")
    print("-" * 60)

    # Create MLP
    model = MultiLayerPerceptron([10, 20, 20, 5])

    print("Network structure:")
    print(model)

    x = torch.randn(3, 10)
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")


def demonstrate_gradient_flow():
    """Visualize gradient flow through linear layers"""
    print("\n8. Gradient Flow")
    print("-" * 60)

    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 3)
    )

    x = torch.randn(2, 5, requires_grad=True)
    target = torch.randn(2, 3)

    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    print("Gradients exist:")
    for name, param in model.named_parameters():
        has_grad = param.grad is not None
        if has_grad:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: {has_grad}, norm={grad_norm:.6f}")
        else:
            print(f"  {name}: {has_grad}")


def main():
    print("=" * 60)
    print("Linear Layers and Weight Initialization")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    demonstrate_basic_linear()
    demonstrate_no_bias()
    demonstrate_weight_inspection()
    demonstrate_xavier_initialization()
    demonstrate_he_initialization()
    demonstrate_custom_initialization()
    demonstrate_sequential_layers()
    demonstrate_gradient_flow()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Key concepts:")
    print("- nn.Linear implements y = xW^T + b")
    print("- Xavier init: good for sigmoid/tanh (std = sqrt(2/(in+out)))")
    print("- He init: good for ReLU (std = sqrt(2/in))")
    print("- Default PyTorch init is uniform based on fan_in")
    print("- Proper initialization helps gradient flow")
    print("=" * 60)


if __name__ == "__main__":
    main()
