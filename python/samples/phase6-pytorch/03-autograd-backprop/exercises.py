"""
Autograd and Backpropagation Exercises

Practice problems for understanding automatic differentiation, computational
graphs, and gradient computation in PyTorch.
"""

import torch
import torch.nn as nn
import math


def exercise1_manual_gradients():
    """
    Exercise 1: Manual Gradient Computation

    Given f(x) = 3x^3 + 2x^2 - 5x + 1:
    a) Compute the gradient at x = 2 using PyTorch autograd
    b) Verify by computing the analytical gradient manually
    c) Compare the results
    """
    print("=" * 70)
    print("Exercise 1: Manual Gradient Computation")
    print("=" * 70)
    print("Given: f(x) = 3x^3 + 2x^2 - 5x + 1")
    print("Find: f'(x) at x = 2")

    # TODO: Implement this exercise
    # Hints:
    # - Create x = torch.tensor(2.0, requires_grad=True)
    # - Compute y = 3*x**3 + 2*x**2 - 5*x + 1
    # - Call backward()
    # - Analytical derivative: f'(x) = 9x^2 + 4x - 5

    # YOUR CODE HERE
    x = torch.tensor(2.0, requires_grad=True)
    y = 3*x**3 + 2*x**2 - 5*x + 1
    y.backward()

    computed_grad = x.grad.item()
    analytical_grad = 9*x.item()**2 + 4*x.item() - 5

    print(f"\nSolution:")
    print(f"PyTorch gradient: {computed_grad}")
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Difference: {abs(computed_grad - analytical_grad):.10f}")


def exercise2_chain_rule():
    """
    Exercise 2: Chain Rule

    Given h(x) = sin(x^2), compute dh/dx at x = π/4

    Use chain rule: dh/dx = cos(x^2) * 2x
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Chain Rule")
    print("=" * 70)
    print("Given: h(x) = sin(x^2)")
    print("Find: h'(x) at x = π/4")

    # TODO: Implement this exercise
    # Hints:
    # - Use torch.sin() for sine
    # - Analytical derivative: dh/dx = cos(x^2) * 2x

    # YOUR CODE HERE
    x = torch.tensor(math.pi / 4, requires_grad=True)
    h = torch.sin(x ** 2)
    h.backward()

    computed_grad = x.grad.item()
    analytical_grad = math.cos(x.item()**2) * 2 * x.item()

    print(f"\nSolution:")
    print(f"PyTorch gradient: {computed_grad:.6f}")
    print(f"Analytical gradient: {analytical_grad:.6f}")
    print(f"Difference: {abs(computed_grad - analytical_grad):.10f}")


def exercise3_partial_derivatives():
    """
    Exercise 3: Partial Derivatives

    Given f(x, y, z) = x^2*y + y^2*z + z^2*x
    Compute ∂f/∂x, ∂f/∂y, ∂f/∂z at (x=1, y=2, z=3)
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Partial Derivatives")
    print("=" * 70)
    print("Given: f(x, y, z) = x^2*y + y^2*z + z^2*x")
    print("Find: ∂f/∂x, ∂f/∂y, ∂f/∂z at (1, 2, 3)")

    # TODO: Implement this exercise
    # Hints:
    # - Create three tensors with requires_grad=True
    # - Analytical: ∂f/∂x = 2xy + z^2
    #              ∂f/∂y = x^2 + 2yz
    #              ∂f/∂z = y^2 + 2zx

    # YOUR CODE HERE
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = torch.tensor(3.0, requires_grad=True)

    f = x**2 * y + y**2 * z + z**2 * x
    f.backward()

    print(f"\nSolution:")
    print(f"∂f/∂x = {x.grad.item():.6f} (analytical: {2*x.item()*y.item() + z.item()**2})")
    print(f"∂f/∂y = {y.grad.item():.6f} (analytical: {x.item()**2 + 2*y.item()*z.item()})")
    print(f"∂f/∂z = {z.grad.item():.6f} (analytical: {y.item()**2 + 2*z.item()*x.item()})")


def exercise4_custom_autograd_function():
    """
    Exercise 4: Custom Autograd Function

    Implement a custom autograd function for f(x) = sqrt(x)
    The gradient is df/dx = 1 / (2*sqrt(x))
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Custom Autograd Function")
    print("=" * 70)
    print("Implement: f(x) = sqrt(x) with custom backward")

    # TODO: Complete this custom function
    class Sqrt(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # YOUR CODE HERE
            result = torch.sqrt(input)
            ctx.save_for_backward(result)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            # YOUR CODE HERE
            # Gradient: df/dx = 1 / (2*sqrt(x))
            result, = ctx.saved_tensors
            grad_input = grad_output / (2 * result)
            return grad_input

    # Test the custom function
    x = torch.tensor([1.0, 4.0, 9.0, 16.0], requires_grad=True)
    y = Sqrt.apply(x)
    loss = y.sum()
    loss.backward()

    print(f"\nSolution:")
    print(f"x = {x.detach()}")
    print(f"sqrt(x) = {y.detach()}")
    print(f"Gradient = {x.grad}")
    print(f"Expected: {1 / (2 * torch.sqrt(x.detach()))}")


def exercise5_gradient_descent_optimization():
    """
    Exercise 5: Gradient Descent Optimization

    Minimize f(x, y) = (x - 3)^2 + (y + 2)^2 using gradient descent.
    The minimum is at (3, -2).
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Gradient Descent Optimization")
    print("=" * 70)
    print("Minimize: f(x, y) = (x - 3)^2 + (y + 2)^2")
    print("Starting point: (0, 0)")

    # TODO: Implement gradient descent
    # Hints:
    # - Initialize x and y far from the optimum
    # - Use learning rate around 0.1
    # - Run for 50-100 steps

    # YOUR CODE HERE
    x = torch.tensor(0.0, requires_grad=True)
    y = torch.tensor(0.0, requires_grad=True)
    learning_rate = 0.1

    print(f"\nOptimization steps:")
    for step in range(50):
        # Compute function value
        f = (x - 3)**2 + (y + 2)**2

        # Backward pass
        if x.grad is not None:
            x.grad.zero_()
            y.grad.zero_()
        f.backward()

        # Update parameters
        with torch.no_grad():
            x -= learning_rate * x.grad
            y -= learning_rate * y.grad

        if step % 10 == 0:
            print(f"Step {step:2d}: x={x.item():7.4f}, y={y.item():7.4f}, f={f.item():8.5f}")

    print(f"\nFinal: x={x.item():.6f}, y={y.item():.6f}")
    print(f"Optimal: x=3.0, y=-2.0")


def exercise6_higher_order_derivatives():
    """
    Exercise 6: Higher-Order Derivatives

    Given f(x) = e^(x^2), compute:
    a) First derivative at x = 1
    b) Second derivative at x = 1
    """
    print("\n" + "=" * 70)
    print("Exercise 6: Higher-Order Derivatives")
    print("=" * 70)
    print("Given: f(x) = e^(x^2)")
    print("Find: f'(1) and f''(1)")

    # TODO: Implement this exercise
    # Hints:
    # - Use create_graph=True for first derivative
    # - Analytical: f'(x) = 2x * e^(x^2)
    #              f''(x) = 2 * e^(x^2) + 4x^2 * e^(x^2)

    # YOUR CODE HERE
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.exp(x ** 2)

    # First derivative
    grad1 = torch.autograd.grad(y, x, create_graph=True)[0]

    # Second derivative
    grad2 = torch.autograd.grad(grad1, x)[0]

    # Analytical values
    exp_x2 = math.exp(x.item() ** 2)
    analytical_grad1 = 2 * x.item() * exp_x2
    analytical_grad2 = 2 * exp_x2 + 4 * x.item()**2 * exp_x2

    print(f"\nSolution:")
    print(f"f'(1) = {grad1.item():.6f} (analytical: {analytical_grad1:.6f})")
    print(f"f''(1) = {grad2.item():.6f} (analytical: {analytical_grad2:.6f})")


def exercise7_gradient_accumulation():
    """
    Exercise 7: Gradient Accumulation

    Implement a training loop that:
    - Uses mini-batches of size 4
    - Accumulates gradients over 8 mini-batches
    - Effective batch size of 32
    """
    print("\n" + "=" * 70)
    print("Exercise 7: Gradient Accumulation")
    print("=" * 70)
    print("Train a linear model with gradient accumulation")

    # TODO: Implement gradient accumulation training
    # Hints:
    # - Create synthetic data: y = 2x + 1 + noise
    # - Use nn.Linear(1, 1)
    # - Accumulate over multiple batches before updating

    # YOUR CODE HERE
    torch.manual_seed(42)

    # Generate data: y = 2x + 1 + noise
    X = torch.randn(128, 1)
    y_true = 2 * X + 1 + 0.1 * torch.randn(128, 1)

    # Model
    model = nn.Linear(1, 1)
    learning_rate = 0.01
    batch_size = 4
    accumulation_steps = 8

    print(f"Batch size: {batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")
    print(f"\nTrue parameters: w=2.0, b=1.0")
    print(f"Initial: w={model.weight.item():.4f}, b={model.bias.item():.4f}")

    # Training
    model.zero_grad()
    for i in range(0, accumulation_steps * batch_size, batch_size):
        # Mini-batch
        x_batch = X[i:i+batch_size]
        y_batch = y_true[i:i+batch_size]

        # Forward
        pred = model(x_batch)
        loss = ((pred - y_batch) ** 2).mean()

        # Backward (accumulate)
        (loss / accumulation_steps).backward()

    # Update
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

    print(f"\nAfter one update:")
    print(f"w={model.weight.item():.4f}, b={model.bias.item():.4f}")
    print(f"Gradient accumulated over {accumulation_steps} batches")


def exercise8_computational_graph():
    """
    Exercise 8: Understanding Computational Graphs

    Given: z = (x^2 + y^2) * (x - y)
    Compute gradients and explain the graph structure.
    """
    print("\n" + "=" * 70)
    print("Exercise 8: Computational Graph Analysis")
    print("=" * 70)
    print("Given: z = (x^2 + y^2) * (x - y)")
    print("Analyze the computational graph")

    # TODO: Implement and analyze
    # Hints:
    # - Break down the computation into steps
    # - Use retain_grad() to see intermediate gradients

    # YOUR CODE HERE
    x = torch.tensor(3.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)

    # Build graph step by step
    a = x ** 2
    b = y ** 2
    c = a + b
    d = x - y
    z = c * d

    # Retain intermediate gradients
    a.retain_grad()
    b.retain_grad()
    c.retain_grad()
    d.retain_grad()

    print(f"\nForward pass:")
    print(f"a = x^2 = {a.item()}")
    print(f"b = y^2 = {b.item()}")
    print(f"c = a + b = {c.item()}")
    print(f"d = x - y = {d.item()}")
    print(f"z = c * d = {z.item()}")

    # Backward pass
    z.backward()

    print(f"\nBackward pass (gradients):")
    print(f"dz/dx = {x.grad.item():.4f}")
    print(f"dz/dy = {y.grad.item():.4f}")
    print(f"dz/da = {a.grad.item():.4f}")
    print(f"dz/db = {b.grad.item():.4f}")
    print(f"dz/dc = {c.grad.item():.4f}")
    print(f"dz/dd = {d.grad.item():.4f}")

    print(f"\nGraph structure:")
    print(f"  x, y (inputs)")
    print(f"  |  |")
    print(f"  v  v")
    print(f"  a  b    d")
    print(f"  |  |    |")
    print(f"  v  v    |")
    print(f"    c     |")
    print(f"    |     |")
    print(f"    v     v")
    print(f"       z")


def bonus_exercise_jacobian():
    """
    Bonus Exercise: Jacobian Matrix

    Given vector function f: R^2 -> R^3:
    f(x) = [x1^2, x1*x2, x2^2]

    Compute the Jacobian matrix at (2, 3).
    """
    print("\n" + "=" * 70)
    print("Bonus Exercise: Jacobian Matrix")
    print("=" * 70)
    print("Given: f(x1, x2) = [x1^2, x1*x2, x2^2]")
    print("Compute Jacobian at (2, 3)")

    # TODO: Compute Jacobian matrix
    # Jacobian = [[∂f1/∂x1, ∂f1/∂x2],
    #             [∂f2/∂x1, ∂f2/∂x2],
    #             [∂f3/∂x1, ∂f3/∂x2]]

    # YOUR CODE HERE
    x = torch.tensor([2.0, 3.0], requires_grad=True)

    # Compute function
    f = torch.stack([x[0]**2, x[0]*x[1], x[1]**2])

    print(f"\nf(2, 3) = {f.detach()}")

    # Compute Jacobian rows
    jacobian = []
    for i in range(3):
        x.grad = None
        f[i].backward(retain_graph=True)
        jacobian.append(x.grad.clone())

    jacobian = torch.stack(jacobian)

    print(f"\nJacobian matrix:")
    print(jacobian)

    print(f"\nAnalytical Jacobian:")
    print(f"[[2*x1,    0  ],  = [[{2*x[0].item():4.1f}, {0:4.1f}],")
    print(f" [ x2,    x1  ],     [{x[1].item():4.1f}, {x[0].item():4.1f}],")
    print(f" [  0,   2*x2]]      [{0:4.1f}, {2*x[1].item():4.1f}]]")


def main():
    """Run all exercises."""
    print("\n" + "=" * 70)
    print("AUTOGRAD AND BACKPROPAGATION EXERCISES")
    print("=" * 70)

    exercise1_manual_gradients()
    exercise2_chain_rule()
    exercise3_partial_derivatives()
    exercise4_custom_autograd_function()
    exercise5_gradient_descent_optimization()
    exercise6_higher_order_derivatives()
    exercise7_gradient_accumulation()
    exercise8_computational_graph()
    bonus_exercise_jacobian()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. PyTorch autograd automatically computes gradients")
    print("2. Computational graphs are built dynamically")
    print("3. Custom autograd functions enable custom gradients")
    print("4. Gradient accumulation enables large effective batches")
    print("5. Understanding backprop is crucial for deep learning")
    print("=" * 70)


if __name__ == "__main__":
    main()
