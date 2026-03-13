"""
PyTorch Introduction
====================
First PyTorch program, basic workflow, and simple tensors.

Topics:
- PyTorch ecosystem overview
- Basic workflow
- Creating simple tensors
- Simple operations
- PyTorch vs traditional programming

Run: python 02_pytorch_intro.py
"""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pytorch_ecosystem():
    """Overview of PyTorch ecosystem components."""
    print("\n" + "=" * 60)
    print("PyTorch Ecosystem")
    print("=" * 60)

    print("\nCore Components:")
    print("- torch: Core library for tensors and operations")
    print("- torch.nn: Neural network layers and modules")
    print("- torch.optim: Optimization algorithms (SGD, Adam, etc.)")
    print("- torch.autograd: Automatic differentiation")
    print("- torch.utils.data: Data loading and preprocessing")

    print("\nExtensions:")
    print("- torchvision: Computer vision (datasets, models, transforms)")
    print("- torchaudio: Audio processing")
    print("- torchtext: Natural language processing")
    print("- TensorBoard: Visualization and monitoring")
    print("- PyTorch Lightning: High-level training framework")


def hello_pytorch():
    """First PyTorch program - the hello world equivalent."""
    print("\n" + "=" * 60)
    print("Hello PyTorch!")
    print("=" * 60)

    # Create a simple tensor
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"\nCreated tensor: {x}")
    print(f"Tensor type: {x.dtype}")
    print(f"Tensor shape: {x.shape}")
    print(f"Tensor device: {x.device}")

    # Simple operation
    y = x * 2
    print(f"\nMultiply by 2: {y}")

    # Sum operation
    total = torch.sum(x)
    print(f"Sum of elements: {total}")
    print(f"Sum type: {type(total)}")
    print(f"Sum as Python number: {total.item()}")


def basic_workflow():
    """Demonstrate basic PyTorch workflow."""
    print("\n" + "=" * 60)
    print("Basic PyTorch Workflow")
    print("=" * 60)

    print("\nStep 1: Create input data")
    # Simulate some input features (e.g., 5 samples, 3 features each)
    X = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ])
    print(f"Input X shape: {X.shape}")  # (5, 3)
    print(f"Input X:\n{X}")

    print("\nStep 2: Define parameters (weights)")
    # Simple linear transformation: y = X @ W + b
    W = torch.tensor([
        [0.5],
        [0.3],
        [0.2]
    ])
    b = torch.tensor([1.0])
    print(f"Weight W shape: {W.shape}")  # (3, 1)
    print(f"Bias b shape: {b.shape}")     # (1,)

    print("\nStep 3: Forward pass (compute predictions)")
    y_pred = torch.matmul(X, W) + b
    print(f"Predictions shape: {y_pred.shape}")  # (5, 1)
    print(f"Predictions:\n{y_pred}")

    print("\nStep 4: Compare with targets")
    y_true = torch.tensor([[5.0], [10.0], [15.0], [20.0], [25.0]])
    print(f"True values:\n{y_true}")

    print("\nStep 5: Compute loss (Mean Squared Error)")
    loss = torch.mean((y_pred - y_true) ** 2)
    print(f"Loss: {loss.item():.4f}")


def tensor_creation_basics():
    """Basic tensor creation methods."""
    print("\n" + "=" * 60)
    print("Creating Tensors - Basic Methods")
    print("=" * 60)

    # From Python list
    print("\n1. From Python list:")
    a = torch.tensor([1, 2, 3, 4, 5])
    print(f"   {a}")

    # From nested list (2D)
    print("\n2. From nested list (2D):")
    b = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"   Shape: {b.shape}")
    print(f"   {b}")

    # Zeros
    print("\n3. Zeros tensor:")
    c = torch.zeros(3, 4)
    print(f"   Shape: {c.shape}")
    print(f"   {c}")

    # Ones
    print("\n4. Ones tensor:")
    d = torch.ones(2, 3)
    print(f"   {d}")

    # Random
    print("\n5. Random tensor (uniform [0, 1)):")
    e = torch.rand(2, 2)
    print(f"   {e}")

    # Random normal
    print("\n6. Random normal (mean=0, std=1):")
    f = torch.randn(2, 3)
    print(f"   {f}")

    # Range
    print("\n7. Range tensor:")
    g = torch.arange(0, 10, 2)
    print(f"   {g}")

    # Linspace
    print("\n8. Linspace (10 points from 0 to 1):")
    h = torch.linspace(0, 1, 10)
    print(f"   {h}")


def simple_operations():
    """Demonstrate simple tensor operations."""
    print("\n" + "=" * 60)
    print("Simple Tensor Operations")
    print("=" * 60)

    # Element-wise operations
    print("\nElement-wise operations:")
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([5.0, 6.0, 7.0, 8.0])

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")  # Element-wise multiplication
    print(f"a / b = {a / b}")
    print(f"a ** 2 = {a ** 2}")

    # Reduction operations
    print("\nReduction operations:")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"x = {x}")
    print(f"sum(x) = {torch.sum(x).item()}")
    print(f"mean(x) = {torch.mean(x).item()}")
    print(f"max(x) = {torch.max(x).item()}")
    print(f"min(x) = {torch.min(x).item()}")
    print(f"std(x) = {torch.std(x).item():.4f}")

    # Matrix operations
    print("\nMatrix operations:")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"A @ B (matrix multiplication) = \n{A @ B}")
    print(f"A.T (transpose) = \n{A.T}")


def why_pytorch():
    """Explain why PyTorch is useful for deep learning."""
    print("\n" + "=" * 60)
    print("Why PyTorch?")
    print("=" * 60)

    print("\n1. GPU Acceleration:")
    print("   - Automatically use GPU for faster computation")
    x_cpu = torch.randn(1000, 1000)
    if torch.cuda.is_available():
        x_gpu = x_cpu.to('cuda')
        print(f"   CPU tensor device: {x_cpu.device}")
        print(f"   GPU tensor device: {x_gpu.device}")
    else:
        print("   (GPU not available, but code would work the same)")

    print("\n2. Automatic Differentiation:")
    print("   - Automatically compute gradients for backpropagation")
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2
    y.backward()
    print(f"   x = {x.item()}, y = x^2 = {y.item()}")
    print(f"   dy/dx = {x.grad.item()} (computed automatically!)")

    print("\n3. Dynamic Computation Graphs:")
    print("   - Build graphs on-the-fly (Pythonic and flexible)")
    print("   - Easy debugging with standard Python tools")

    print("\n4. Rich Ecosystem:")
    print("   - Pre-trained models, datasets, utilities")
    print("   - Active community and industry adoption")

    print("\n5. Production Ready:")
    print("   - TorchScript for deployment")
    print("   - ONNX export for interoperability")
    print("   - Mobile and edge device support")


def main():
    """Main function to run all demonstrations."""
    print("=" * 60)
    print("PyTorch Introduction")
    print("=" * 60)
    print(f"\nRunning on device: {device}")

    # Run all demonstrations
    pytorch_ecosystem()
    hello_pytorch()
    basic_workflow()
    tensor_creation_basics()
    simple_operations()
    why_pytorch()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. PyTorch is a deep learning framework built around tensors")
    print("2. Tensors are like NumPy arrays but can run on GPU")
    print("3. Basic workflow: data -> forward pass -> loss -> optimization")
    print("4. Automatic differentiation enables easy gradient computation")
    print("5. Rich ecosystem makes building models easier")
    print("\nNext: 03_tensor_basics.py for deeper dive into tensors")
    print("=" * 60)


if __name__ == "__main__":
    main()
