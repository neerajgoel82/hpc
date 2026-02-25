"""
Gradient Accumulation - Accumulating Gradients Over Multiple Batches

This module demonstrates gradient accumulation techniques, which are useful
for training with large effective batch sizes when memory is limited.

Key Concepts:
- Gradient accumulation over batches
- Simulating large batch sizes
- When and why to use gradient accumulation
- Proper gradient scaling
- Zero_grad timing
"""

import torch
import torch.nn as nn


def basic_gradient_accumulation():
    """Demonstrate basic gradient accumulation."""
    print("=" * 70)
    print("Basic Gradient Accumulation")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)

    print("Accumulating gradients from multiple forward passes:")

    # First forward-backward
    y1 = x ** 2
    y1.backward()
    print(f"After first backward (y1 = x^2): x.grad = {x.grad}")

    # Second forward-backward (accumulates)
    y2 = x ** 3
    y2.backward()
    print(f"After second backward (y2 = x^3): x.grad = {x.grad}")

    # Third forward-backward (accumulates)
    y3 = x ** 4
    y3.backward()
    print(f"After third backward (y3 = x^4): x.grad = {x.grad}")

    print("\nGradients accumulate by default!")
    print("Total gradient = 2x + 3x^2 + 4x^3")
    expected = 2*x.item() + 3*x.item()**2 + 4*x.item()**3
    print(f"Expected: {expected}")


def gradient_accumulation_for_batches():
    """Simulate large batch training with gradient accumulation."""
    print("\n" + "=" * 70)
    print("Gradient Accumulation for Large Batch Simulation")
    print("=" * 70)

    # Create a simple model
    model = nn.Linear(10, 1)

    # Synthetic data
    batch_size = 8
    accumulation_steps = 4
    effective_batch_size = batch_size * accumulation_steps

    print(f"Small batch size: {batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")

    # Training with accumulation
    print("\n--- Training with Gradient Accumulation ---")
    model.zero_grad()

    total_loss = 0
    for step in range(accumulation_steps):
        # Mini-batch
        x = torch.randn(batch_size, 10)
        y = torch.randn(batch_size, 1)

        # Forward pass
        pred = model(x)
        loss = ((pred - y) ** 2).mean()

        # Backward pass (gradients accumulate)
        loss.backward()

        total_loss += loss.item()
        print(f"Step {step + 1}: loss = {loss.item():.6f}")

    print(f"\nAverage loss: {total_loss / accumulation_steps:.6f}")
    print(f"Gradient norm: {model.weight.grad.norm().item():.6f}")

    # Compare with full batch
    print("\n--- Training with Full Batch (comparison) ---")
    model2 = nn.Linear(10, 1)
    model2.weight.data = model.weight.data.clone()
    model2.bias.data = model.bias.data.clone()
    model2.zero_grad()

    # Full batch
    x_full = torch.randn(effective_batch_size, 10)
    y_full = torch.randn(effective_batch_size, 1)

    pred = model2(x_full)
    loss = ((pred - y_full) ** 2).mean()
    loss.backward()

    print(f"Full batch loss: {loss.item():.6f}")
    print(f"Gradient norm: {model2.weight.grad.norm().item():.6f}")


def proper_gradient_scaling():
    """Demonstrate proper gradient scaling during accumulation."""
    print("\n" + "=" * 70)
    print("Proper Gradient Scaling")
    print("=" * 70)

    model = nn.Linear(5, 1)
    accumulation_steps = 4

    print("Method 1: Scale loss before backward")
    print("-" * 40)

    model.zero_grad()
    for step in range(accumulation_steps):
        x = torch.randn(4, 5)
        y = torch.randn(4, 1)

        pred = model(x)
        loss = ((pred - y) ** 2).mean()

        # Scale loss before backward
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        if step == 0:
            print(f"Step 1 - gradient norm: {model.weight.grad.norm().item():.6f}")

    print(f"Final accumulated gradient norm: {model.weight.grad.norm().item():.6f}")

    print("\nMethod 2: Scale gradients after accumulation")
    print("-" * 40)

    model.zero_grad()
    for step in range(accumulation_steps):
        x = torch.randn(4, 5)
        y = torch.randn(4, 1)

        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()

    # Scale gradients after accumulation
    with torch.no_grad():
        for param in model.parameters():
            param.grad /= accumulation_steps

    print(f"Final scaled gradient norm: {model.weight.grad.norm().item():.6f}")


def gradient_accumulation_training_loop():
    """Complete training loop with gradient accumulation."""
    print("\n" + "=" * 70)
    print("Complete Training Loop with Gradient Accumulation")
    print("=" * 70)

    # Simple dataset
    torch.manual_seed(42)
    X_data = torch.randn(128, 10)
    y_data = torch.randn(128, 1)

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    learning_rate = 0.01
    batch_size = 8
    accumulation_steps = 4
    n_epochs = 5

    print(f"Dataset size: {len(X_data)}")
    print(f"Batch size: {batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")

    for epoch in range(n_epochs):
        model.zero_grad()
        epoch_loss = 0

        for i in range(0, len(X_data), batch_size):
            # Get batch
            x_batch = X_data[i:i + batch_size]
            y_batch = y_data[i:i + batch_size]

            # Forward pass
            pred = model(x_batch)
            loss = ((pred - y_batch) ** 2).mean()

            # Scale loss
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            epoch_loss += loss.item()

            # Update weights every accumulation_steps
            if (i // batch_size + 1) % accumulation_steps == 0:
                # Gradient update
                with torch.no_grad():
                    for param in model.parameters():
                        param -= learning_rate * param.grad

                # Zero gradients
                model.zero_grad()

        avg_loss = epoch_loss / (len(X_data) / batch_size)
        print(f"Epoch {epoch + 1}: avg loss = {avg_loss:.6f}")


def when_to_accumulate():
    """Demonstrate scenarios where gradient accumulation is useful."""
    print("\n" + "=" * 70)
    print("When to Use Gradient Accumulation")
    print("=" * 70)

    print("Scenario 1: Memory-constrained training")
    print("-" * 40)
    print("- Large models that don't fit with large batches")
    print("- Can use smaller batches but maintain effective batch size")
    print("- Example: batch_size=4, accumulation=8 -> effective_batch=32")

    print("\nScenario 2: Stable gradients for small batches")
    print("-" * 40)
    print("- Small datasets where batch size is limited")
    print("- Accumulation provides more stable gradient estimates")

    print("\nScenario 3: Gradient checkpointing")
    print("-" * 40)
    print("- Trade computation for memory")
    print("- Accumulate over multiple forward passes")


def gradient_accumulation_with_different_losses():
    """Accumulate gradients from different loss terms."""
    print("\n" + "=" * 70)
    print("Gradient Accumulation with Multiple Loss Terms")
    print("=" * 70)

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    print("Accumulating gradients from different loss functions:")

    # First loss: L1 = sum(x^2)
    loss1 = (x ** 2).sum()
    loss1.backward()
    print(f"After L1 = sum(x^2): x.grad = {x.grad}")

    # Second loss: L2 = sum(x^3) (accumulates)
    loss2 = (x ** 3).sum()
    loss2.backward()
    print(f"After L2 = sum(x^3): x.grad = {x.grad}")

    # This is equivalent to optimizing L1 + L2
    x2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    combined_loss = (x2 ** 2).sum() + (x2 ** 3).sum()
    combined_loss.backward()
    print(f"Combined L1 + L2: x.grad = {x2.grad}")


def accumulation_with_varying_batch_sizes():
    """Handle gradient accumulation with varying batch sizes."""
    print("\n" + "=" * 70)
    print("Gradient Accumulation with Varying Batch Sizes")
    print("=" * 70)

    model = nn.Linear(5, 1)

    batch_sizes = [4, 6, 5, 3]  # Varying sizes
    total_samples = sum(batch_sizes)

    print(f"Batch sizes: {batch_sizes}")
    print(f"Total samples: {total_samples}")

    model.zero_grad()

    for i, bs in enumerate(batch_sizes):
        x = torch.randn(bs, 5)
        y = torch.randn(bs, 1)

        pred = model(x)
        loss = ((pred - y) ** 2).mean()

        # Scale by batch size to average correctly
        weighted_loss = loss * bs / total_samples
        weighted_loss.backward()

        print(f"Batch {i + 1} (size={bs}): loss={loss.item():.6f}, "
              f"weight={bs/total_samples:.3f}")

    print(f"\nFinal gradient norm: {model.weight.grad.norm().item():.6f}")


def gradient_accumulation_memory_efficiency():
    """Compare memory usage with and without accumulation."""
    print("\n" + "=" * 70)
    print("Memory Efficiency of Gradient Accumulation")
    print("=" * 70)

    feature_size = 1000

    print(f"Feature size: {feature_size}")
    print("\nScenario 1: Large batch without accumulation")
    print(f"Batch size: 128")
    print(f"Memory for activations: ~{128 * feature_size * 4 / 1024:.2f} KB")

    print("\nScenario 2: Small batches with accumulation")
    print(f"Batch size: 16 x 8 accumulation steps")
    print(f"Memory for activations: ~{16 * feature_size * 4 / 1024:.2f} KB")
    print(f"Memory savings: ~{((128-16)/128*100):.1f}%")

    print("\nNote: Gradient accumulation allows training with")
    print("larger effective batch sizes in memory-constrained environments.")


def main():
    """Run all gradient accumulation examples."""
    print("\n" + "=" * 70)
    print("GRADIENT ACCUMULATION")
    print("=" * 70)

    basic_gradient_accumulation()
    gradient_accumulation_for_batches()
    proper_gradient_scaling()
    gradient_accumulation_training_loop()
    when_to_accumulate()
    gradient_accumulation_with_different_losses()
    accumulation_with_varying_batch_sizes()
    gradient_accumulation_memory_efficiency()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
