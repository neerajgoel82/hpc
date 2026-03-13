"""
Activation Functions
ReLU, Sigmoid, Tanh, LeakyReLU, and more

This module demonstrates:
- Common activation functions
- Activation function properties
- Gradient behavior
- Choosing the right activation
- Custom activation functions

Run: python 03_activations.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelWithActivations(nn.Module):
    """Model demonstrating different activation functions"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 activation: str = 'relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class CustomActivation(nn.Module):
    """Custom activation function: Swish"""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


def demonstrate_relu():
    """ReLU activation function"""
    print("\n1. ReLU (Rectified Linear Unit)")
    print("-" * 60)

    relu = nn.ReLU()
    x = torch.linspace(-5, 5, 11)
    y = relu(x)

    print(f"Input:  {x}")
    print(f"Output: {y}")
    print("\nProperties:")
    print("- Range: [0, inf)")
    print("- Non-linear but piecewise linear")
    print("- Solves vanishing gradient problem")
    print("- Can cause 'dying ReLU' problem")


def demonstrate_sigmoid():
    """Sigmoid activation function"""
    print("\n2. Sigmoid")
    print("-" * 60)

    sigmoid = nn.Sigmoid()
    x = torch.linspace(-5, 5, 11)
    y = sigmoid(x)

    print(f"Input:  {x}")
    print(f"Output: {y}")
    print("\nProperties:")
    print("- Range: (0, 1)")
    print("- Good for binary classification")
    print("- Suffers from vanishing gradients")
    print("- Output not zero-centered")


def demonstrate_tanh():
    """Tanh activation function"""
    print("\n3. Tanh (Hyperbolic Tangent)")
    print("-" * 60)

    tanh = nn.Tanh()
    x = torch.linspace(-5, 5, 11)
    y = tanh(x)

    print(f"Input:  {x}")
    print(f"Output: {y}")
    print("\nProperties:")
    print("- Range: (-1, 1)")
    print("- Zero-centered (better than sigmoid)")
    print("- Still suffers from vanishing gradients")


def demonstrate_leaky_relu():
    """LeakyReLU activation function"""
    print("\n4. LeakyReLU")
    print("-" * 60)

    leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    x = torch.linspace(-5, 5, 11)
    y = leaky_relu(x)

    print(f"Input:  {x}")
    print(f"Output: {y}")
    print("\nProperties:")
    print("- Range: (-inf, inf)")
    print("- Fixes dying ReLU problem")
    print("- Allows small gradient when x < 0")


def demonstrate_elu():
    """ELU (Exponential Linear Unit)"""
    print("\n5. ELU (Exponential Linear Unit)")
    print("-" * 60)

    elu = nn.ELU(alpha=1.0)
    x = torch.linspace(-5, 5, 11)
    y = elu(x)

    print(f"Input:  {x}")
    print(f"Output: {y}")
    print("\nProperties:")
    print("- Range: (-alpha, inf)")
    print("- Smooth curve for negative values")
    print("- Mean activation closer to zero")


def demonstrate_gelu():
    """GELU (Gaussian Error Linear Unit)"""
    print("\n6. GELU (Gaussian Error Linear Unit)")
    print("-" * 60)

    gelu = nn.GELU()
    x = torch.linspace(-5, 5, 11)
    y = gelu(x)

    print(f"Input:  {x}")
    print(f"Output: {y}")
    print("\nProperties:")
    print("- Used in BERT and GPT models")
    print("- Smooth, non-monotonic")
    print("- Weights inputs by their magnitude")


def demonstrate_silu_swish():
    """SiLU/Swish activation"""
    print("\n7. SiLU/Swish")
    print("-" * 60)

    silu = nn.SiLU()  # Also known as Swish
    x = torch.linspace(-5, 5, 11)
    y = silu(x)

    print(f"Input:  {x}")
    print(f"Output: {y}")
    print("\nProperties:")
    print("- Range: (-inf, inf)")
    print("- Self-gated: x * sigmoid(x)")
    print("- Used in EfficientNet")


def demonstrate_gradient_behavior():
    """Compare gradient behavior of activations"""
    print("\n8. Gradient Behavior")
    print("-" * 60)

    x = torch.linspace(-3, 3, 7, requires_grad=True)

    activations = {
        'ReLU': nn.ReLU(),
        'Sigmoid': nn.Sigmoid(),
        'Tanh': nn.Tanh(),
        'LeakyReLU': nn.LeakyReLU(),
    }

    for name, activation in activations.items():
        x.grad = None
        y = activation(x)
        loss = y.sum()
        loss.backward()

        print(f"{name}:")
        print(f"  Gradients: {x.grad}")


def demonstrate_functional_vs_module():
    """Functional vs Module API"""
    print("\n9. Functional vs Module API")
    print("-" * 60)

    x = torch.randn(2, 5)

    # Module API (stateful, can be part of nn.Sequential)
    relu_module = nn.ReLU()
    out1 = relu_module(x)

    # Functional API (stateless, direct computation)
    out2 = F.relu(x)

    print(f"Module output: {out1[0]}")
    print(f"Functional output: {out2[0]}")
    print(f"Outputs match: {torch.allclose(out1, out2)}")
    print("\nUse Module API in model definitions")
    print("Use Functional API for one-off operations")


def demonstrate_activation_in_network():
    """Activations in neural networks"""
    print("\n10. Activations in Networks")
    print("-" * 60)

    models = {
        'ReLU': ModelWithActivations(10, 20, 5, 'relu'),
        'Tanh': ModelWithActivations(10, 20, 5, 'tanh'),
        'GELU': ModelWithActivations(10, 20, 5, 'gelu'),
    }

    x = torch.randn(3, 10)

    for name, model in models.items():
        output = model(x)
        print(f"{name} network output shape: {output.shape}")


def demonstrate_custom_activation():
    """Custom activation function"""
    print("\n11. Custom Activation")
    print("-" * 60)

    custom_swish = CustomActivation(beta=1.0)
    builtin_silu = nn.SiLU()

    x = torch.linspace(-3, 3, 7)
    y1 = custom_swish(x)
    y2 = builtin_silu(x)

    print(f"Input: {x}")
    print(f"Custom Swish: {y1}")
    print(f"Built-in SiLU: {y2}")
    print(f"Close enough: {torch.allclose(y1, y2, atol=1e-6)}")


def main():
    print("=" * 60)
    print("Activation Functions")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    demonstrate_relu()
    demonstrate_sigmoid()
    demonstrate_tanh()
    demonstrate_leaky_relu()
    demonstrate_elu()
    demonstrate_gelu()
    demonstrate_silu_swish()
    demonstrate_gradient_behavior()
    demonstrate_functional_vs_module()
    demonstrate_activation_in_network()
    demonstrate_custom_activation()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Choosing activations:")
    print("- Hidden layers: ReLU, LeakyReLU, GELU (default: ReLU)")
    print("- Binary output: Sigmoid")
    print("- Multi-class output: Softmax")
    print("- Regression output: None (linear)")
    print("- Modern architectures: GELU, SiLU/Swish")
    print("=" * 60)


if __name__ == "__main__":
    main()
