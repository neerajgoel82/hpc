"""
Autograd Basics - Understanding Automatic Differentiation in PyTorch

This module demonstrates the fundamental concepts of automatic differentiation
including requires_grad, backward(), and gradient computation.

Key Concepts:
- requires_grad flag for tracking operations
- backward() method for computing gradients
- grad attribute for accessing computed gradients
- Gradient tracking on/off
"""

import torch
import math


def basic_requires_grad():
    """Demonstrate the requires_grad flag and its purpose."""
    print("=" * 70)
    print("Basic requires_grad Example")
    print("=" * 70)

    # Tensor without gradient tracking
    x = torch.tensor([2.0, 3.0, 4.0])
    print(f"x = {x}")
    print(f"x.requires_grad = {x.requires_grad}")

    # Tensor with gradient tracking
    y = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    print(f"\ny = {y}")
    print(f"y.requires_grad = {y.requires_grad}")

    # Enable gradient tracking on existing tensor
    z = torch.tensor([2.0, 3.0, 4.0])
    z.requires_grad_(True)  # In-place operation
    print(f"\nz = {z}")
    print(f"z.requires_grad = {z.requires_grad}")


def simple_backward():
    """Demonstrate basic backward pass with f(x) = x^2."""
    print("\n" + "=" * 70)
    print("Simple Backward Pass: f(x) = x^2")
    print("=" * 70)

    # Create a scalar tensor
    x = torch.tensor(3.0, requires_grad=True)
    print(f"x = {x}")

    # Compute y = x^2
    y = x ** 2
    print(f"y = x^2 = {y}")

    # Compute gradient dy/dx = 2x
    y.backward()
    print(f"dy/dx (computed) = {x.grad}")
    print(f"dy/dx (expected) = {2 * x.item()}")


def vector_backward():
    """Demonstrate backward pass with vector inputs."""
    print("\n" + "=" * 70)
    print("Vector Backward Pass: f(x) = x^2")
    print("=" * 70)

    # Create a vector tensor
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    print(f"x = {x}")

    # Compute y = x^2 (element-wise)
    y = x ** 2
    print(f"y = x^2 = {y}")

    # For non-scalar output, we need to specify gradient
    # This is equivalent to computing sum(y) then backward
    y.backward(gradient=torch.ones_like(y))
    print(f"dy/dx = {x.grad}")
    print(f"Expected (2*x) = {2 * x.detach()}")


def sum_then_backward():
    """Demonstrate backward pass with sum reduction."""
    print("\n" + "=" * 70)
    print("Sum Then Backward: f(x) = sum(x^2)")
    print("=" * 70)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    print(f"x = {x}")

    # Compute y = sum(x^2)
    y = (x ** 2).sum()
    print(f"y = sum(x^2) = {y}")

    # Compute gradient
    y.backward()
    print(f"dy/dx = {x.grad}")
    print(f"Expected (2*x) = {2 * x.detach()}")


def chain_rule_example():
    """Demonstrate chain rule: f(g(x)) where g(x) = x^2 and f(u) = u^3."""
    print("\n" + "=" * 70)
    print("Chain Rule: f(g(x)) where g(x) = x^2, f(u) = u^3")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)
    print(f"x = {x}")

    # Compute g(x) = x^2
    u = x ** 2
    print(f"u = x^2 = {u}")

    # Compute f(u) = u^3
    y = u ** 3
    print(f"y = u^3 = (x^2)^3 = x^6 = {y}")

    # Compute gradient
    y.backward()
    print(f"\ndy/dx (computed) = {x.grad}")

    # Manual calculation: dy/dx = dy/du * du/dx = 3u^2 * 2x = 3(x^2)^2 * 2x = 6x^5
    manual_grad = 6 * (x.item() ** 5)
    print(f"dy/dx (expected) = 6x^5 = {manual_grad}")


def multiple_operations():
    """Demonstrate gradient computation with multiple operations."""
    print("\n" + "=" * 70)
    print("Multiple Operations: f(x, y) = 2x^2 + 3y^3 + 4xy")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    print(f"x = {x}, y = {y}")

    # Compute f(x, y) = 2x^2 + 3y^3 + 4xy
    z = 2 * x**2 + 3 * y**3 + 4 * x * y
    print(f"z = 2x^2 + 3y^3 + 4xy = {z}")

    # Compute gradients
    z.backward()
    print(f"\ndz/dx (computed) = {x.grad}")
    print(f"dz/dx (expected) = 4x + 4y = {4 * x.item() + 4 * y.item()}")
    print(f"\ndz/dy (computed) = {y.grad}")
    print(f"dz/dy (expected) = 9y^2 + 4x = {9 * y.item()**2 + 4 * x.item()}")


def gradient_accumulation():
    """Demonstrate how gradients accumulate by default."""
    print("\n" + "=" * 70)
    print("Gradient Accumulation")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)

    # First backward pass
    y1 = x ** 2
    y1.backward()
    print(f"After first backward (y1 = x^2):")
    print(f"x.grad = {x.grad}")

    # Second backward pass - gradients accumulate!
    y2 = x ** 3
    y2.backward()
    print(f"\nAfter second backward (y2 = x^3):")
    print(f"x.grad = {x.grad} (accumulated)")

    # Clear gradients
    x.grad.zero_()
    print(f"\nAfter x.grad.zero_():")
    print(f"x.grad = {x.grad}")

    # Third backward pass
    y3 = x ** 4
    y3.backward()
    print(f"\nAfter third backward (y3 = x^4):")
    print(f"x.grad = {x.grad} (fresh gradient)")


def no_grad_context():
    """Demonstrate torch.no_grad() context manager."""
    print("\n" + "=" * 70)
    print("No Gradient Context")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)

    # Normal operation - gradient tracked
    y1 = x ** 2
    print(f"y1 = x^2, requires_grad = {y1.requires_grad}")

    # Inside no_grad context - no gradient tracking
    with torch.no_grad():
        y2 = x ** 2
        print(f"y2 = x^2 (no_grad), requires_grad = {y2.requires_grad}")

    # Can also use decorator
    @torch.no_grad()
    def compute_without_grad(x):
        return x ** 2

    y3 = compute_without_grad(x)
    print(f"y3 = x^2 (decorator), requires_grad = {y3.requires_grad}")


def detach_example():
    """Demonstrate tensor.detach() method."""
    print("\n" + "=" * 70)
    print("Detach Example")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2

    # Detach creates a new tensor without gradient tracking
    y_detached = y.detach()
    print(f"y.requires_grad = {y.requires_grad}")
    print(f"y_detached.requires_grad = {y_detached.requires_grad}")

    # Operations on detached tensor don't track gradients
    z = y_detached * 2
    print(f"z = y_detached * 2, requires_grad = {z.requires_grad}")


def main():
    """Run all autograd basics examples."""
    print("\n" + "=" * 70)
    print("PYTORCH AUTOGRAD BASICS")
    print("=" * 70)

    basic_requires_grad()
    simple_backward()
    vector_backward()
    sum_then_backward()
    chain_rule_example()
    multiple_operations()
    gradient_accumulation()
    no_grad_context()
    detach_example()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
