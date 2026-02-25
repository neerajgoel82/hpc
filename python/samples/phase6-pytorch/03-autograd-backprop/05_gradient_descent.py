"""
Manual Gradient Descent - Understanding Optimization Through Manual Updates

This module demonstrates manual implementation of gradient descent and its
variants without using torch.optim, showing how optimization works under the hood.

Key Concepts:
- Basic gradient descent
- Learning rate effects
- Gradient descent variants (SGD, momentum, RMSprop, Adam)
- Manual parameter updates
- Optimization of simple functions
"""

import torch
import math


def basic_gradient_descent():
    """Demonstrate basic gradient descent on f(x) = x^2."""
    print("=" * 70)
    print("Basic Gradient Descent: f(x) = x^2")
    print("=" * 70)

    # Initialize parameter
    x = torch.tensor(10.0, requires_grad=True)
    learning_rate = 0.1

    print(f"Initial x = {x.item()}")
    print(f"Learning rate = {learning_rate}")
    print("\nOptimization steps:")

    # Run gradient descent
    for step in range(20):
        # Forward pass
        y = x ** 2

        # Compute gradient
        if x.grad is not None:
            x.grad.zero_()
        y.backward()

        # Manual update: x = x - lr * gradient
        with torch.no_grad():
            x -= learning_rate * x.grad

        if step % 5 == 0:
            print(f"Step {step:2d}: x = {x.item():8.5f}, f(x) = {y.item():8.5f}")

    print(f"\nFinal x = {x.item():.6f} (optimal: 0)")


def learning_rate_comparison():
    """Compare different learning rates."""
    print("\n" + "=" * 70)
    print("Learning Rate Comparison")
    print("=" * 70)

    learning_rates = [0.01, 0.1, 0.5, 0.9]

    for lr in learning_rates:
        x = torch.tensor(10.0, requires_grad=True)

        print(f"\nLearning rate = {lr}")
        for step in range(10):
            y = x ** 2
            if x.grad is not None:
                x.grad.zero_()
            y.backward()

            with torch.no_grad():
                x -= lr * x.grad

        print(f"Final x = {x.item():8.5f}")


def multivariable_optimization():
    """Optimize a multivariable function: f(x, y) = x^2 + y^2."""
    print("\n" + "=" * 70)
    print("Multivariable Optimization: f(x, y) = x^2 + y^2")
    print("=" * 70)

    # Initialize parameters
    x = torch.tensor(5.0, requires_grad=True)
    y = torch.tensor(-3.0, requires_grad=True)
    learning_rate = 0.1

    print(f"Initial: x = {x.item()}, y = {y.item()}")
    print("\nOptimization steps:")

    for step in range(20):
        # Forward pass
        z = x**2 + y**2

        # Backward pass
        if x.grad is not None:
            x.grad.zero_()
            y.grad.zero_()
        z.backward()

        # Manual update
        with torch.no_grad():
            x -= learning_rate * x.grad
            y -= learning_rate * y.grad

        if step % 5 == 0:
            print(f"Step {step:2d}: x = {x.item():7.4f}, y = {y.item():7.4f}, "
                  f"f = {z.item():7.4f}")

    print(f"\nFinal: x = {x.item():.6f}, y = {y.item():.6f} (optimal: 0, 0)")


def linear_regression_manual():
    """Implement linear regression with manual gradient descent."""
    print("\n" + "=" * 70)
    print("Linear Regression with Manual GD")
    print("=" * 70)

    # Generate synthetic data: y = 3x + 2 + noise
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    y_true = 3 * X + 2 + 0.1 * torch.randn(100, 1)

    # Initialize parameters
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    learning_rate = 0.01
    n_epochs = 100

    print("True parameters: w = 3, b = 2")
    print(f"Initial: w = {w.item():.4f}, b = {b.item():.4f}")

    for epoch in range(n_epochs):
        # Forward pass
        y_pred = X @ w + b
        loss = ((y_pred - y_true) ** 2).mean()

        # Backward pass
        if w.grad is not None:
            w.grad.zero_()
            b.grad.zero_()
        loss.backward()

        # Update parameters
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: loss = {loss.item():.6f}, "
                  f"w = {w.item():.4f}, b = {b.item():.4f}")

    print(f"\nFinal: w = {w.item():.4f}, b = {b.item():.4f}")


def momentum_gradient_descent():
    """Implement gradient descent with momentum."""
    print("\n" + "=" * 70)
    print("Gradient Descent with Momentum")
    print("=" * 70)

    # Optimize f(x) = x^2 with momentum
    x = torch.tensor(10.0, requires_grad=True)
    learning_rate = 0.01
    momentum = 0.9
    velocity = 0.0

    print(f"Initial x = {x.item()}")
    print(f"Learning rate = {learning_rate}, momentum = {momentum}")
    print("\nOptimization steps:")

    for step in range(30):
        y = x ** 2

        if x.grad is not None:
            x.grad.zero_()
        y.backward()

        # Update velocity and parameter
        with torch.no_grad():
            velocity = momentum * velocity + learning_rate * x.grad
            x -= velocity

        if step % 5 == 0:
            print(f"Step {step:2d}: x = {x.item():8.5f}, velocity = {velocity:8.5f}")

    print(f"\nFinal x = {x.item():.6f}")


def rmsprop_optimization():
    """Implement RMSprop optimizer manually."""
    print("\n" + "=" * 70)
    print("RMSprop Optimization")
    print("=" * 70)

    # Optimize f(x, y) = x^2 + 10*y^2 (different scales)
    x = torch.tensor(5.0, requires_grad=True)
    y = torch.tensor(5.0, requires_grad=True)

    learning_rate = 0.1
    decay_rate = 0.9
    epsilon = 1e-8

    # RMSprop state
    cache_x = 0.0
    cache_y = 0.0

    print("Function: f(x, y) = x^2 + 10*y^2")
    print(f"Initial: x = {x.item()}, y = {y.item()}")

    for step in range(30):
        z = x**2 + 10 * y**2

        if x.grad is not None:
            x.grad.zero_()
            y.grad.zero_()
        z.backward()

        with torch.no_grad():
            # Update cache (moving average of squared gradients)
            cache_x = decay_rate * cache_x + (1 - decay_rate) * x.grad**2
            cache_y = decay_rate * cache_y + (1 - decay_rate) * y.grad**2

            # Adaptive learning rate
            x -= learning_rate * x.grad / (math.sqrt(cache_x) + epsilon)
            y -= learning_rate * y.grad / (math.sqrt(cache_y) + epsilon)

        if step % 5 == 0:
            print(f"Step {step:2d}: x = {x.item():7.4f}, y = {y.item():7.4f}, "
                  f"f = {z.item():8.4f}")

    print(f"\nFinal: x = {x.item():.6f}, y = {y.item():.6f}")


def adam_optimization():
    """Implement Adam optimizer manually."""
    print("\n" + "=" * 70)
    print("Adam Optimization")
    print("=" * 70)

    # Optimize f(x) = x^2
    x = torch.tensor(10.0, requires_grad=True)

    learning_rate = 0.1
    beta1 = 0.9  # Momentum
    beta2 = 0.999  # RMSprop
    epsilon = 1e-8

    # Adam state
    m = 0.0  # First moment
    v = 0.0  # Second moment

    print(f"Initial x = {x.item()}")
    print(f"Hyperparameters: lr = {learning_rate}, beta1 = {beta1}, beta2 = {beta2}")

    for step in range(1, 21):
        y = x ** 2

        if x.grad is not None:
            x.grad.zero_()
        y.backward()

        with torch.no_grad():
            # Update biased first moment
            m = beta1 * m + (1 - beta1) * x.grad
            # Update biased second moment
            v = beta2 * v + (1 - beta2) * x.grad**2

            # Bias correction
            m_hat = m / (1 - beta1**step)
            v_hat = v / (1 - beta2**step)

            # Update parameter
            x -= learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)

        if step % 5 == 0:
            print(f"Step {step:2d}: x = {x.item():8.5f}, f(x) = {y.item():8.5f}")

    print(f"\nFinal x = {x.item():.6f}")


def rosenbrock_optimization():
    """Optimize the Rosenbrock function (non-convex)."""
    print("\n" + "=" * 70)
    print("Rosenbrock Function: f(x, y) = (1-x)^2 + 100(y-x^2)^2")
    print("=" * 70)

    # Initialize far from optimum
    x = torch.tensor(-1.0, requires_grad=True)
    y = torch.tensor(-1.0, requires_grad=True)

    learning_rate = 0.001

    print(f"Initial: x = {x.item()}, y = {y.item()}")
    print("Optimal: x = 1, y = 1")

    for step in range(500):
        # Rosenbrock function
        z = (1 - x)**2 + 100 * (y - x**2)**2

        if x.grad is not None:
            x.grad.zero_()
            y.grad.zero_()
        z.backward()

        with torch.no_grad():
            x -= learning_rate * x.grad
            y -= learning_rate * y.grad

        if step % 100 == 0:
            print(f"Step {step:3d}: x = {x.item():7.4f}, y = {y.item():7.4f}, "
                  f"f = {z.item():10.4f}")

    print(f"\nFinal: x = {x.item():.4f}, y = {y.item():.4f}")


def gradient_descent_convergence():
    """Analyze convergence of gradient descent."""
    print("\n" + "=" * 70)
    print("Gradient Descent Convergence Analysis")
    print("=" * 70)

    x = torch.tensor(10.0, requires_grad=True)
    learning_rate = 0.1

    losses = []

    for step in range(50):
        y = x ** 2

        if x.grad is not None:
            x.grad.zero_()
        y.backward()

        losses.append(y.item())

        with torch.no_grad():
            x -= learning_rate * x.grad

    print(f"Loss reduction:")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"After 10 steps: {losses[10]:.6f}")
    print(f"After 20 steps: {losses[20]:.6f}")
    print(f"After 30 steps: {losses[30]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")

    # Convergence rate
    print(f"\nConvergence rate (exponential decay):")
    for i in [10, 20, 30, 40]:
        if losses[i-10] > 0:
            rate = losses[i] / losses[i-10]
            print(f"Steps {i-10}-{i}: reduction factor = {rate:.4f}")


def main():
    """Run all manual gradient descent examples."""
    print("\n" + "=" * 70)
    print("MANUAL GRADIENT DESCENT")
    print("=" * 70)

    basic_gradient_descent()
    learning_rate_comparison()
    multivariable_optimization()
    linear_regression_manual()
    momentum_gradient_descent()
    rmsprop_optimization()
    adam_optimization()
    rosenbrock_optimization()
    gradient_descent_convergence()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
