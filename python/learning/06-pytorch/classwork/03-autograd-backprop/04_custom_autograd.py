"""
Custom Autograd Functions - Implementing Custom Backward Functions

This module demonstrates how to create custom autograd functions using
torch.autograd.Function for operations that need custom gradient computation.

Key Concepts:
- torch.autograd.Function
- Custom forward and backward methods
- Saving tensors for backward
- Custom gradients for non-differentiable operations
"""

import torch
import torch.nn as nn
import math


class Square(torch.autograd.Function):
    """Custom autograd function for squaring: f(x) = x^2."""

    @staticmethod
    def forward(ctx, input):
        """
        Forward pass: compute output.
        ctx: context object to save information for backward pass
        """
        # Save input for backward pass
        ctx.save_for_backward(input)
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradient.
        grad_output: gradient of loss w.r.t. output
        Returns: gradient of loss w.r.t. input
        """
        # Retrieve saved input
        input, = ctx.saved_tensors
        # Gradient: d(x^2)/dx = 2x
        grad_input = 2 * input * grad_output
        return grad_input


def test_square_function():
    """Test custom Square function."""
    print("=" * 70)
    print("Custom Square Function: f(x) = x^2")
    print("=" * 70)

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Use custom function
    y = Square.apply(x)

    print(f"x = {x}")
    print(f"y = x^2 = {y}")

    # Backward pass
    loss = y.sum()
    loss.backward()

    print(f"∂loss/∂x = {x.grad}")
    print(f"Expected: 2x = {2 * x.detach()}")


class Exp(torch.autograd.Function):
    """Custom autograd function for exponential: f(x) = e^x."""

    @staticmethod
    def forward(ctx, input):
        """Forward pass: compute e^x."""
        result = input.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: gradient of e^x is e^x."""
        result, = ctx.saved_tensors
        return grad_output * result


def test_exp_function():
    """Test custom Exp function."""
    print("\n" + "=" * 70)
    print("Custom Exp Function: f(x) = e^x")
    print("=" * 70)

    x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)

    y = Exp.apply(x)

    print(f"x = {x}")
    print(f"y = e^x = {y}")

    loss = y.sum()
    loss.backward()

    print(f"∂loss/∂x = {x.grad}")
    print(f"Expected: e^x = {torch.exp(x.detach())}")


class LinearFunction(torch.autograd.Function):
    """Custom autograd function for linear layer: y = xW + b."""

    @staticmethod
    def forward(ctx, input, weight, bias):
        """Forward pass: compute xW + b."""
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: compute gradients w.r.t. input, weight, and bias."""
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        # Gradient w.r.t. input
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        # Gradient w.r.t. weight
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        # Gradient w.r.t. bias
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


def test_linear_function():
    """Test custom Linear function."""
    print("\n" + "=" * 70)
    print("Custom Linear Function: y = xW + b")
    print("=" * 70)

    # Input: (batch_size=2, in_features=3)
    x = torch.randn(2, 3, requires_grad=True)
    # Weight: (out_features=2, in_features=3)
    w = torch.randn(2, 3, requires_grad=True)
    # Bias: (out_features=2)
    b = torch.randn(2, requires_grad=True)

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {w.shape}")
    print(f"Bias shape: {b.shape}")

    # Forward pass
    y = LinearFunction.apply(x, w, b)
    print(f"Output shape: {y.shape}")

    # Backward pass
    loss = y.sum()
    loss.backward()

    print(f"\nGradient shapes:")
    print(f"∂loss/∂x: {x.grad.shape}")
    print(f"∂loss/∂w: {w.grad.shape}")
    print(f"∂loss/∂b: {b.grad.shape}")


class ReLU(torch.autograd.Function):
    """Custom autograd function for ReLU activation."""

    @staticmethod
    def forward(ctx, input):
        """Forward pass: compute max(0, x)."""
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: gradient is 1 if x > 0, else 0."""
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


def test_relu_function():
    """Test custom ReLU function."""
    print("\n" + "=" * 70)
    print("Custom ReLU Function")
    print("=" * 70)

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

    y = ReLU.apply(x)

    print(f"x = {x}")
    print(f"ReLU(x) = {y}")

    loss = y.sum()
    loss.backward()

    print(f"∂loss/∂x = {x.grad}")
    print("(gradient is 1 where x > 0, else 0)")


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-through estimator for binary quantization.
    Forward: quantize to 0 or 1
    Backward: pass gradient through unchanged
    """

    @staticmethod
    def forward(ctx, input):
        """Forward pass: binarize input."""
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: straight-through (identity)."""
        return grad_output


def test_straight_through():
    """Test straight-through estimator."""
    print("\n" + "=" * 70)
    print("Straight-Through Estimator")
    print("=" * 70)

    x = torch.tensor([0.2, 0.4, 0.6, 0.8], requires_grad=True)

    # Forward pass quantizes
    y = StraightThroughEstimator.apply(x)

    print(f"x (continuous) = {x}")
    print(f"y (binary) = {y}")

    # Backward pass treats as identity
    loss = y.sum()
    loss.backward()

    print(f"∂loss/∂x = {x.grad}")
    print("(gradient passes through unchanged)")


class Sigmoid(torch.autograd.Function):
    """Custom autograd function for sigmoid activation."""

    @staticmethod
    def forward(ctx, input):
        """Forward pass: compute sigmoid(x) = 1 / (1 + e^(-x))."""
        output = 1 / (1 + torch.exp(-input))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))."""
        output, = ctx.saved_tensors
        grad_input = grad_output * output * (1 - output)
        return grad_input


def test_sigmoid_function():
    """Test custom Sigmoid function."""
    print("\n" + "=" * 70)
    print("Custom Sigmoid Function")
    print("=" * 70)

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

    y = Sigmoid.apply(x)

    print(f"x = {x}")
    print(f"sigmoid(x) = {y}")

    loss = y.sum()
    loss.backward()

    print(f"∂loss/∂x = {x.grad}")

    # Compare with PyTorch's sigmoid
    x2 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y2 = torch.sigmoid(x2)
    y2.sum().backward()
    print(f"PyTorch gradient = {x2.grad}")


class MultiplyAdd(torch.autograd.Function):
    """Custom function with multiple inputs and outputs."""

    @staticmethod
    def forward(ctx, x, y, z):
        """Forward pass: compute (x * y, x + z)."""
        ctx.save_for_backward(x, y, z)
        return x * y, x + z

    @staticmethod
    def backward(ctx, grad_out1, grad_out2):
        """Backward pass for multiple outputs."""
        x, y, z = ctx.saved_tensors

        # Gradient w.r.t. x from both outputs
        grad_x = grad_out1 * y + grad_out2
        # Gradient w.r.t. y from first output
        grad_y = grad_out1 * x
        # Gradient w.r.t. z from second output
        grad_z = grad_out2

        return grad_x, grad_y, grad_z


def test_multiply_add():
    """Test custom function with multiple inputs/outputs."""
    print("\n" + "=" * 70)
    print("Custom Function with Multiple Inputs/Outputs")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = torch.tensor(4.0, requires_grad=True)

    out1, out2 = MultiplyAdd.apply(x, y, z)

    print(f"x = {x.item()}, y = {y.item()}, z = {z.item()}")
    print(f"out1 = x * y = {out1.item()}")
    print(f"out2 = x + z = {out2.item()}")

    # Backward from both outputs
    loss = out1 + out2
    loss.backward()

    print(f"\nGradients:")
    print(f"∂loss/∂x = {x.grad.item()}")
    print(f"∂loss/∂y = {y.grad.item()}")
    print(f"∂loss/∂z = {z.grad.item()}")


class ClipGradient(torch.autograd.Function):
    """Custom function that clips gradients during backward pass."""

    @staticmethod
    def forward(ctx, input, min_val, max_val):
        """Forward pass: identity function."""
        ctx.min_val = min_val
        ctx.max_val = max_val
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: clip gradients."""
        grad_input = grad_output.clone()
        grad_input = torch.clamp(grad_input, ctx.min_val, ctx.max_val)
        return grad_input, None, None


def test_gradient_clipping():
    """Test custom gradient clipping."""
    print("\n" + "=" * 70)
    print("Custom Gradient Clipping")
    print("=" * 70)

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Apply gradient clipping
    y = ClipGradient.apply(x, -1.0, 1.0)

    # Large gradients
    loss = (y ** 3).sum()
    loss.backward()

    print(f"x = {x.detach()}")
    print(f"Gradient (clipped to [-1, 1]): {x.grad}")
    print(f"Without clipping would be: {3 * x.detach() ** 2}")


def main():
    """Run all custom autograd examples."""
    print("\n" + "=" * 70)
    print("CUSTOM AUTOGRAD FUNCTIONS")
    print("=" * 70)

    test_square_function()
    test_exp_function()
    test_linear_function()
    test_relu_function()
    test_straight_through()
    test_sigmoid_function()
    test_multiply_add()
    test_gradient_clipping()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
