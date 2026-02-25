"""
nn.Module Basics
Creating neural network modules with PyTorch

This module demonstrates:
- Creating custom nn.Module subclasses
- Understanding forward pass
- Module parameters and state
- Nested modules
- Module inspection

Run: python 01_nn_module.py
"""

import torch
import torch.nn as nn


class SimpleLinearModel(nn.Module):
    """Basic linear model: y = Wx + b"""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TwoLayerNet(nn.Module):
    """Two-layer neural network"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class CustomModule(nn.Module):
    """Custom module with manual parameter registration"""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        # Register parameters manually
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t()) + self.bias


class ModuleWithBuffer(nn.Module):
    """Module demonstrating buffers (non-trainable state)"""

    def __init__(self, size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size))
        # Register buffer (not trained, but part of state_dict)
        self.register_buffer('running_mean', torch.zeros(size))
        self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update running mean
        if self.training:
            batch_mean = x.mean(0)
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * batch_mean
        return x * self.weight


def demonstrate_basic_module():
    """Basic module creation and usage"""
    print("\n1. Basic Linear Module")
    print("-" * 60)

    model = SimpleLinearModel(input_size=10, output_size=5)
    print(f"Model: {model}")

    # Create input
    x = torch.randn(3, 10)  # batch_size=3, input_size=10
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")


def demonstrate_parameters():
    """Inspect module parameters"""
    print("\n2. Module Parameters")
    print("-" * 60)

    model = TwoLayerNet(input_size=4, hidden_size=8, output_size=2)

    print("Named parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


def demonstrate_nested_modules():
    """Working with nested modules"""
    print("\n3. Nested Modules")
    print("-" * 60)

    model = TwoLayerNet(4, 8, 2)

    print("Named modules:")
    for name, module in model.named_modules():
        print(f"  {name}: {module.__class__.__name__}")

    print("\nDirect children:")
    for name, module in model.named_children():
        print(f"  {name}: {module.__class__.__name__}")


def demonstrate_custom_parameters():
    """Custom parameter registration"""
    print("\n4. Custom Parameters")
    print("-" * 60)

    model = CustomModule(input_size=5, output_size=3)

    x = torch.randn(2, 5)
    output = model(x)

    print(f"Weight shape: {model.weight.shape}")
    print(f"Bias shape: {model.bias.shape}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")


def demonstrate_buffers():
    """Buffers vs parameters"""
    print("\n5. Buffers (Non-trainable State)")
    print("-" * 60)

    model = ModuleWithBuffer(size=4)

    print("Parameters (trainable):")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")

    print("\nBuffers (non-trainable):")
    for name, buffer in model.named_buffers():
        print(f"  {name}: {buffer.shape}")

    # Forward pass updates running mean
    model.train()
    x = torch.randn(10, 4)
    _ = model(x)
    print(f"\nRunning mean after forward: {model.running_mean}")


def demonstrate_training_mode():
    """Training vs evaluation mode"""
    print("\n6. Training vs Evaluation Mode")
    print("-" * 60)

    model = ModuleWithBuffer(size=3)

    print(f"Initial training mode: {model.training}")

    # Training mode
    model.train()
    print(f"After .train(): {model.training}")

    # Evaluation mode
    model.eval()
    print(f"After .eval(): {model.training}")

    # Context manager (recommended for inference)
    with torch.no_grad():
        x = torch.randn(5, 3)
        output = model(x)
        print(f"\nInference output shape: {output.shape}")


def demonstrate_state_dict():
    """Saving and loading model state"""
    print("\n7. State Dictionary")
    print("-" * 60)

    model = TwoLayerNet(4, 8, 2)

    # Get state dict
    state = model.state_dict()
    print("State dict keys:")
    for key in state.keys():
        print(f"  {key}: {state[key].shape}")

    # Save to file (example)
    print("\nSaving model state...")
    torch.save(state, '/tmp/model_state.pth')

    # Load to new model
    new_model = TwoLayerNet(4, 8, 2)
    new_model.load_state_dict(torch.load('/tmp/model_state.pth'))
    print("State loaded successfully!")


def main():
    print("=" * 60)
    print("nn.Module Basics")
    print("=" * 60)

    # Check PyTorch availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    demonstrate_basic_module()
    demonstrate_parameters()
    demonstrate_nested_modules()
    demonstrate_custom_parameters()
    demonstrate_buffers()
    demonstrate_training_mode()
    demonstrate_state_dict()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Key concepts:")
    print("- nn.Module is the base class for all neural network modules")
    print("- Override __init__ and forward methods")
    print("- Parameters are automatically tracked when assigned as attributes")
    print("- Buffers store non-trainable state (e.g., running statistics)")
    print("- Use .train() and .eval() to switch modes")
    print("- state_dict() contains all parameters and buffers")
    print("=" * 60)


if __name__ == "__main__":
    main()
