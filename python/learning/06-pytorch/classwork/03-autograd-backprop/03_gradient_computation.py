"""
Gradient Computation - Computing and Accessing Gradients in PyTorch

This module demonstrates various aspects of gradient computation including
partial derivatives, Jacobians, gradients of vector-valued functions,
and higher-order derivatives.

Key Concepts:
- Partial derivatives
- Jacobian matrices
- Gradient vectors
- Higher-order derivatives
- torch.autograd.grad function
"""

import torch


def partial_derivatives():
    """Demonstrate computing partial derivatives."""
    print("=" * 70)
    print("Partial Derivatives: f(x, y) = x^2 + 2xy + y^2")
    print("=" * 70)

    x = torch.tensor(3.0, requires_grad=True)
    y = torch.tensor(4.0, requires_grad=True)

    # Function f(x, y) = x^2 + 2xy + y^2
    f = x**2 + 2*x*y + y**2

    print(f"f(x, y) = x^2 + 2xy + y^2")
    print(f"x = {x.item()}, y = {y.item()}")
    print(f"f = {f.item()}")

    # Compute gradients
    f.backward()

    print(f"\nPartial derivatives:")
    print(f"∂f/∂x = {x.grad.item()}")
    print(f"Expected: 2x + 2y = {2*x.item() + 2*y.item()}")
    print(f"\n∂f/∂y = {y.grad.item()}")
    print(f"Expected: 2x + 2y = {2*x.item() + 2*y.item()}")


def gradient_of_scalar_function():
    """Compute gradient of a scalar function with multiple inputs."""
    print("\n" + "=" * 70)
    print("Gradient Vector of Scalar Function")
    print("=" * 70)

    # f(x1, x2, x3) = x1^2 + x2^2 + x3^2
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    f = (x ** 2).sum()

    print(f"f(x) = sum(x^2)")
    print(f"x = {x}")
    print(f"f = {f.item()}")

    f.backward()

    print(f"\nGradient vector ∇f:")
    print(f"∇f = {x.grad}")
    print(f"Expected: 2x = {2 * x.detach()}")


def jacobian_vector_product():
    """Demonstrate Jacobian-vector product for vector-valued functions."""
    print("\n" + "=" * 70)
    print("Jacobian-Vector Product")
    print("=" * 70)

    x = torch.tensor([1.0, 2.0], requires_grad=True)

    # Vector-valued function: f(x) = [x1^2, x2^3, x1*x2]
    y = torch.stack([x[0]**2, x[1]**3, x[0]*x[1]])

    print(f"f(x) = [x1^2, x2^3, x1*x2]")
    print(f"x = {x}")
    print(f"f(x) = {y}")

    # Compute Jacobian-vector product with v = [1, 1, 1]
    v = torch.ones_like(y)
    y.backward(gradient=v)

    print(f"\nJacobian-vector product with v = {v}:")
    print(f"J^T * v = {x.grad}")

    # Manual calculation:
    # J = [[2*x1,    0  ],
    #      [  0,   3*x2^2],
    #      [ x2,    x1   ]]
    # J^T * v = [2*x1 + x2, 3*x2^2 + x1]
    manual = torch.tensor([2*x[0].item() + x[1].item(),
                          3*x[1].item()**2 + x[0].item()])
    print(f"Expected: {manual}")


def torch_autograd_grad():
    """Demonstrate torch.autograd.grad function."""
    print("\n" + "=" * 70)
    print("Using torch.autograd.grad")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    # f(x, y) = x^3 + y^2
    z = x**3 + y**2

    print(f"f(x, y) = x^3 + y^2")
    print(f"x = {x.item()}, y = {y.item()}")
    print(f"f = {z.item()}")

    # Compute gradients using torch.autograd.grad
    grads = torch.autograd.grad(z, [x, y])

    print(f"\nGradients (using torch.autograd.grad):")
    print(f"∂f/∂x = {grads[0].item()}")
    print(f"Expected: 3x^2 = {3 * x.item()**2}")
    print(f"∂f/∂y = {grads[1].item()}")
    print(f"Expected: 2y = {2 * y.item()}")


def selective_gradient_computation():
    """Demonstrate computing gradients for specific tensors only."""
    print("\n" + "=" * 70)
    print("Selective Gradient Computation")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = torch.tensor(4.0, requires_grad=True)

    # f(x, y, z) = x^2 + y^2 + z^2
    f = x**2 + y**2 + z**2

    print(f"f(x, y, z) = x^2 + y^2 + z^2")

    # Compute gradient only with respect to x
    grad_x = torch.autograd.grad(f, x, retain_graph=True)
    print(f"\n∂f/∂x = {grad_x[0].item()} (only x)")

    # Compute gradient only with respect to y
    grad_y = torch.autograd.grad(f, y, retain_graph=True)
    print(f"∂f/∂y = {grad_y[0].item()} (only y)")

    # Compute all gradients
    grad_all = torch.autograd.grad(f, [x, y, z])
    print(f"\nAll gradients: {[g.item() for g in grad_all]}")


def higher_order_derivatives():
    """Demonstrate computing second-order derivatives."""
    print("\n" + "=" * 70)
    print("Higher-Order Derivatives")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)

    # f(x) = x^4
    y = x ** 4

    print(f"f(x) = x^4")
    print(f"x = {x.item()}")
    print(f"f = {y.item()}")

    # First derivative: f'(x) = 4x^3
    grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"\nFirst derivative f'(x) = {grad1.item()}")
    print(f"Expected: 4x^3 = {4 * x.item()**3}")

    # Second derivative: f''(x) = 12x^2
    grad2 = torch.autograd.grad(grad1, x)[0]
    print(f"\nSecond derivative f''(x) = {grad2.item()}")
    print(f"Expected: 12x^2 = {12 * x.item()**2}")


def hessian_diagonal():
    """Compute diagonal elements of Hessian matrix."""
    print("\n" + "=" * 70)
    print("Hessian Diagonal: f(x, y) = x^2*y + y^3")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    # f(x, y) = x^2*y + y^3
    z = x**2 * y + y**3

    print(f"f(x, y) = x^2*y + y^3")
    print(f"x = {x.item()}, y = {y.item()}")

    # First derivatives
    grad_x = torch.autograd.grad(z, x, create_graph=True)[0]
    grad_y = torch.autograd.grad(z, y, create_graph=True)[0]

    print(f"\nFirst derivatives:")
    print(f"∂f/∂x = {grad_x.item()}")
    print(f"∂f/∂y = {grad_y.item()}")

    # Second derivatives (Hessian diagonal)
    hess_xx = torch.autograd.grad(grad_x, x, retain_graph=True)[0]
    hess_yy = torch.autograd.grad(grad_y, y)[0]

    print(f"\nHessian diagonal:")
    print(f"∂²f/∂x² = {hess_xx.item()}")
    print(f"∂²f/∂y² = {hess_yy.item()}")


def gradient_of_gradient():
    """Demonstrate gradient of gradient (meta-gradients)."""
    print("\n" + "=" * 70)
    print("Gradient of Gradient")
    print("=" * 70)

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    w = torch.tensor([0.5, 0.3, 0.2], requires_grad=True)

    # Loss function
    y = (w * x**2).sum()

    print(f"f(x, w) = sum(w * x^2)")
    print(f"x = {x}")
    print(f"w = {w}")

    # First gradient w.r.t. x
    grad_x = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"\n∂f/∂x = {grad_x}")

    # Gradient of gradient w.r.t. w
    # This shows how changing w affects the gradient
    grad_grad = torch.autograd.grad(grad_x.sum(), w)[0]
    print(f"\n∂(∂f/∂x)/∂w = {grad_grad}")


def batch_gradients():
    """Compute gradients for batched inputs."""
    print("\n" + "=" * 70)
    print("Batch Gradient Computation")
    print("=" * 70)

    # Batch of inputs
    x = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], requires_grad=True)

    # Compute f(x) = sum(x^2) for entire batch
    y = (x ** 2).sum()

    print(f"Batch input shape: {x.shape}")
    print(f"x =\n{x}")
    print(f"f = sum(x^2) = {y}")

    y.backward()

    print(f"\nGradient (same shape as input):")
    print(f"∂f/∂x =\n{x.grad}")
    print(f"Expected: 2x =\n{2 * x.detach()}")


def grad_with_create_graph():
    """Demonstrate create_graph parameter."""
    print("\n" + "=" * 70)
    print("Using create_graph for Higher-Order Gradients")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3

    # Without create_graph - can't compute higher-order gradients
    grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"f(x) = x^3, x = {x.item()}")
    print(f"f'(x) = {grad1.item()}")

    # Now we can compute gradient of gradient
    grad2 = torch.autograd.grad(grad1, x)[0]
    print(f"f''(x) = {grad2.item()}")


def main():
    """Run all gradient computation examples."""
    print("\n" + "=" * 70)
    print("PYTORCH GRADIENT COMPUTATION")
    print("=" * 70)

    partial_derivatives()
    gradient_of_scalar_function()
    jacobian_vector_product()
    torch_autograd_grad()
    selective_gradient_computation()
    higher_order_derivatives()
    hessian_diagonal()
    gradient_of_gradient()
    batch_gradients()
    grad_with_create_graph()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
