"""
Computational Graphs - Understanding How PyTorch Builds and Uses Graphs

This module demonstrates how PyTorch builds computational graphs during the
forward pass and uses them during the backward pass for automatic differentiation.

Key Concepts:
- Dynamic computational graphs
- Graph nodes and edges
- retain_graph parameter
- Graph visualization concepts
- Leaf vs non-leaf tensors
"""

import torch


def basic_graph_structure():
    """Demonstrate basic computational graph structure."""
    print("=" * 70)
    print("Basic Computational Graph Structure")
    print("=" * 70)

    # Create leaf tensors (inputs)
    x = torch.tensor(2.0, requires_grad=True)
    w = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)

    print(f"Leaf tensors (inputs):")
    print(f"x = {x}, is_leaf = {x.is_leaf}")
    print(f"w = {w}, is_leaf = {w.is_leaf}")
    print(f"b = {b}, is_leaf = {b.is_leaf}")

    # Build computational graph: y = w * x + b
    y = w * x + b

    print(f"\nIntermediate/output tensor:")
    print(f"y = w * x + b = {y}, is_leaf = {y.is_leaf}")

    # Examine gradient function
    print(f"\nGradient function:")
    print(f"y.grad_fn = {y.grad_fn}")


def leaf_vs_nonleaf():
    """Demonstrate the difference between leaf and non-leaf tensors."""
    print("\n" + "=" * 70)
    print("Leaf vs Non-Leaf Tensors")
    print("=" * 70)

    # Leaf tensor - created by user
    x = torch.tensor(2.0, requires_grad=True)
    print(f"x is_leaf = {x.is_leaf}")

    # Non-leaf tensor - result of operation
    y = x * 2
    print(f"y = x * 2, is_leaf = {y.is_leaf}")

    # Another non-leaf
    z = y + 3
    print(f"z = y + 3, is_leaf = {z.is_leaf}")

    # Compute gradients
    z.backward()

    # Only leaf tensors retain gradients by default
    print(f"\nAfter backward:")
    print(f"x.grad = {x.grad}")
    print(f"y.grad = {y.grad}")  # None for non-leaf tensors
    print(f"z.grad = {z.grad}")  # None for output tensor


def retain_intermediate_gradients():
    """Demonstrate how to retain gradients for non-leaf tensors."""
    print("\n" + "=" * 70)
    print("Retaining Intermediate Gradients")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)
    y = x * 2

    # Register a hook to retain gradient
    y.retain_grad()

    z = y + 3
    z.backward()

    print(f"x.grad = {x.grad}")
    print(f"y.grad = {y.grad}")  # Now available!


def graph_lifecycle():
    """Demonstrate the lifecycle of computational graphs."""
    print("\n" + "=" * 70)
    print("Computational Graph Lifecycle")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)

    # Forward pass - graph is built
    y = x ** 2
    z = y * 3

    print(f"After forward pass:")
    print(f"z.grad_fn = {z.grad_fn}")

    # First backward - graph is freed by default
    z.backward()
    print(f"\nAfter first backward:")
    print(f"x.grad = {x.grad}")

    # Try to call backward again - will fail!
    try:
        z.backward()
    except RuntimeError as e:
        print(f"\nError on second backward: {str(e)[:60]}...")


def retain_graph_example():
    """Demonstrate retain_graph parameter."""
    print("\n" + "=" * 70)
    print("Retaining Graph for Multiple Backwards")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2
    z = y * 3

    # First backward with retain_graph=True
    z.backward(retain_graph=True)
    print(f"After first backward:")
    print(f"x.grad = {x.grad}")

    # Second backward - graph still exists
    # But gradients accumulate!
    z.backward()
    print(f"\nAfter second backward:")
    print(f"x.grad = {x.grad} (accumulated)")


def complex_graph():
    """Demonstrate a more complex computational graph."""
    print("\n" + "=" * 70)
    print("Complex Computational Graph")
    print("=" * 70)

    # Create inputs
    x1 = torch.tensor(2.0, requires_grad=True)
    x2 = torch.tensor(3.0, requires_grad=True)

    print("Graph structure:")
    print("  x1, x2 (inputs)")
    print("  |    |")
    print("  v    v")
    print("  a = x1^2")
    print("  b = x2^3")
    print("  |    |")
    print("  v    v")
    print("  c = a + b")
    print("  |")
    print("  v")
    print("  d = c * 2")

    # Build graph
    a = x1 ** 2
    b = x2 ** 3
    c = a + b
    d = c * 2

    # Retain gradients for visualization
    a.retain_grad()
    b.retain_grad()
    c.retain_grad()

    # Backward pass
    d.backward()

    print(f"\nGradients:")
    print(f"dd/dx1 = {x1.grad}")
    print(f"dd/dx2 = {x2.grad}")
    print(f"dd/da = {a.grad}")
    print(f"dd/db = {b.grad}")
    print(f"dd/dc = {c.grad}")


def branching_graph():
    """Demonstrate a graph with branching paths."""
    print("\n" + "=" * 70)
    print("Branching Computational Graph")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)

    print("Graph structure:")
    print("      x")
    print("    /   \\")
    print("   v     v")
    print("  a=x^2 b=x^3")
    print("   |     |")
    print("    \\   /")
    print("     v v")
    print("   c=a+b")

    # Build branching graph
    a = x ** 2
    b = x ** 3
    c = a + b

    # Compute gradient
    c.backward()

    print(f"\nc = x^2 + x^3 = {c}")
    print(f"dc/dx = {x.grad}")
    print(f"Expected: 2x + 3x^2 = {2*x.item() + 3*x.item()**2}")


def multiple_outputs():
    """Demonstrate graph with multiple outputs."""
    print("\n" + "=" * 70)
    print("Multiple Output Graph")
    print("=" * 70)

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Create multiple outputs
    y1 = x.sum()
    y2 = (x ** 2).sum()

    print(f"x = {x}")
    print(f"y1 = sum(x) = {y1}")
    print(f"y2 = sum(x^2) = {y2}")

    # Can only backward from one output at a time
    # First output
    y1.backward(retain_graph=True)
    print(f"\nAfter y1.backward():")
    print(f"x.grad = {x.grad}")

    # Clear gradients
    x.grad.zero_()

    # Second output
    y2.backward()
    print(f"\nAfter y2.backward():")
    print(f"x.grad = {x.grad}")


def inplace_operations():
    """Demonstrate issues with in-place operations."""
    print("\n" + "=" * 70)
    print("In-Place Operations and Graphs")
    print("=" * 70)

    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2

    # This is fine - not in-place
    z = y * 2
    z.backward()
    print(f"Regular operation - x.grad = {x.grad}")

    # Reset
    x.grad.zero_()
    y = x ** 2

    # In-place operation can cause issues
    print("\nIn-place operations can break gradient computation")
    print("because they modify tensors that might be needed")
    print("for backward pass.")


def gradient_flow_example():
    """Demonstrate gradient flow through the graph."""
    print("\n" + "=" * 70)
    print("Gradient Flow Through Graph")
    print("=" * 70)

    # Simple chain: x -> y -> z
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2  # y = 4
    z = y * 3   # z = 12

    print(f"Forward pass:")
    print(f"x = {x}")
    print(f"y = x^2 = {y}")
    print(f"z = y * 3 = {z}")

    print(f"\nBackward pass (gradient flow):")
    print(f"dz/dz = 1 (initial)")
    print(f"dz/dy = dz/dz * 3 = 3")
    print(f"dz/dx = dz/dy * dy/dx = 3 * 2x = 3 * 2 * {x.item()} = {3 * 2 * x.item()}")

    z.backward()
    print(f"\nComputed gradient:")
    print(f"dz/dx = {x.grad}")


def main():
    """Run all computational graph examples."""
    print("\n" + "=" * 70)
    print("PYTORCH COMPUTATIONAL GRAPHS")
    print("=" * 70)

    basic_graph_structure()
    leaf_vs_nonleaf()
    retain_intermediate_gradients()
    graph_lifecycle()
    retain_graph_example()
    complex_graph()
    branching_graph()
    multiple_outputs()
    inplace_operations()
    gradient_flow_example()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
