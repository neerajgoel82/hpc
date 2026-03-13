"""
Optimizers
SGD, Adam, AdamW comparison

This module demonstrates:
- Stochastic Gradient Descent (SGD)
- SGD with momentum
- Adam optimizer
- AdamW optimizer
- Learning rate and hyperparameters
- Optimizer comparison

Run: python 05_optimizers.py
"""

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNet(nn.Module):
    """Simple network for optimizer demonstrations"""

    def __init__(self, input_size: int = 10, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_steps(model, optimizer, data, target, steps: int = 3):
    """Run a few training steps"""
    losses = []
    criterion = nn.MSELoss()

    for step in range(steps):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def demonstrate_sgd():
    """Basic SGD optimizer"""
    print("\n1. SGD (Stochastic Gradient Descent)")
    print("-" * 60)

    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Parameters groups: {len(optimizer.param_groups)}")

    # Show optimizer state
    print("\nOptimizer defaults:")
    print(f"  {optimizer.defaults}")


def demonstrate_sgd_momentum():
    """SGD with momentum"""
    print("\n2. SGD with Momentum")
    print("-" * 60)

    model = SimpleNet()

    # Without momentum
    sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    print(f"SGD without momentum: {sgd.defaults}")

    # With momentum
    sgd_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print(f"SGD with momentum: {sgd_momentum.defaults}")

    print("\nMomentum helps accelerate convergence")
    print("Typical values: 0.9 or 0.99")


def demonstrate_adam():
    """Adam optimizer"""
    print("\n3. Adam Optimizer")
    print("-" * 60)

    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Betas (momentum): {optimizer.param_groups[0]['betas']}")
    print(f"Epsilon: {optimizer.param_groups[0]['eps']}")

    print("\nAdam adapts learning rate per parameter")
    print("Good default choice for many problems")


def demonstrate_adamw():
    """AdamW optimizer (Adam with weight decay fix)"""
    print("\n4. AdamW Optimizer")
    print("-" * 60)

    model = SimpleNet()

    # Adam with weight decay (incorrect)
    adam = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    print(f"Adam with weight_decay: {adam.defaults['weight_decay']}")

    # AdamW (correct decoupled weight decay)
    adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    print(f"AdamW with weight_decay: {adamw.defaults['weight_decay']}")

    print("\nAdamW fixes weight decay implementation")
    print("Preferred for transformer models")


def demonstrate_learning_rates():
    """Different learning rates"""
    print("\n5. Learning Rate Comparison")
    print("-" * 60)

    data = torch.randn(32, 10)
    target = torch.randn(32, 1)

    learning_rates = [0.001, 0.01, 0.1]

    for lr in learning_rates:
        model = SimpleNet()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        losses = train_steps(model, optimizer, data, target, steps=5)

        print(f"\nLR={lr}:")
        print(f"  Losses: {[f'{l:.4f}' for l in losses]}")


def demonstrate_optimizer_comparison():
    """Compare different optimizers"""
    print("\n6. Optimizer Comparison")
    print("-" * 60)

    data = torch.randn(32, 10)
    target = torch.randn(32, 1)

    optimizers_config = [
        ('SGD', lambda p: optim.SGD(p, lr=0.01)),
        ('SGD+Momentum', lambda p: optim.SGD(p, lr=0.01, momentum=0.9)),
        ('Adam', lambda p: optim.Adam(p, lr=0.001)),
        ('AdamW', lambda p: optim.AdamW(p, lr=0.001, weight_decay=0.01)),
    ]

    for name, opt_fn in optimizers_config:
        model = SimpleNet()
        optimizer = opt_fn(model.parameters())
        losses = train_steps(model, optimizer, data, target, steps=10)

        print(f"\n{name}:")
        print(f"  Initial: {losses[0]:.4f}")
        print(f"  Final:   {losses[-1]:.4f}")
        print(f"  Improvement: {(losses[0] - losses[-1]):.4f}")


def demonstrate_parameter_groups():
    """Different learning rates for different layers"""
    print("\n7. Parameter Groups")
    print("-" * 60)

    model = SimpleNet()

    # Different learning rates for different layers
    optimizer = optim.SGD([
        {'params': model.fc1.parameters(), 'lr': 0.01},
        {'params': model.fc2.parameters(), 'lr': 0.001}
    ])

    print("Parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  Group {i}: lr={group['lr']}, "
              f"params={sum(p.numel() for p in group['params'])}")


def demonstrate_optimizer_state():
    """Optimizer state and checkpointing"""
    print("\n8. Optimizer State")
    print("-" * 60)

    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initial state
    print(f"Initial state_dict keys: {list(optimizer.state_dict().keys())}")

    # Run one step to populate state
    data = torch.randn(32, 10)
    target = torch.randn(32, 1)
    output = model(data)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()

    # Check state after step
    state = optimizer.state_dict()
    print(f"\nAfter one step:")
    print(f"  State keys: {list(state.keys())}")
    print(f"  Has state for {len(state['state'])} parameters")

    # Save and load optimizer state
    torch.save(state, '/tmp/optimizer_state.pth')
    print("\nOptimizer state can be saved/loaded for checkpointing")


def demonstrate_zero_grad():
    """Importance of zero_grad()"""
    print("\n9. Zero Grad")
    print("-" * 60)

    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    data = torch.randn(4, 10)
    target = torch.randn(4, 1)
    criterion = nn.MSELoss()

    # First forward-backward
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # Check gradients
    grad_before = model.fc1.weight.grad.clone()
    print(f"Gradient norm after 1st backward: {grad_before.norm().item():.4f}")

    # Second backward WITHOUT zero_grad (wrong!)
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    grad_accumulated = model.fc1.weight.grad
    print(f"Gradient norm without zero_grad: {grad_accumulated.norm().item():.4f}")

    # Correct way: zero_grad before backward
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    grad_correct = model.fc1.weight.grad
    print(f"Gradient norm with zero_grad: {grad_correct.norm().item():.4f}")

    print("\nAlways call optimizer.zero_grad() before backward!")


def demonstrate_gradient_clipping():
    """Gradient clipping"""
    print("\n10. Gradient Clipping")
    print("-" * 60)

    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    data = torch.randn(4, 10)
    target = torch.randn(4, 1)

    # Forward and backward
    output = model(data)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    # Check gradient norm before clipping
    total_norm_before = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_before += p.grad.data.norm(2).item() ** 2
    total_norm_before = total_norm_before ** 0.5
    print(f"Gradient norm before clipping: {total_norm_before:.4f}")

    # Clip gradients
    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    total_norm_after = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_after += p.grad.data.norm(2).item() ** 2
    total_norm_after = total_norm_after ** 0.5
    print(f"Gradient norm after clipping:  {total_norm_after:.4f}")
    print(f"Max norm: {max_norm}")

    print("\nGradient clipping prevents exploding gradients")


def main():
    print("=" * 60)
    print("Optimizers")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    demonstrate_sgd()
    demonstrate_sgd_momentum()
    demonstrate_adam()
    demonstrate_adamw()
    demonstrate_learning_rates()
    demonstrate_optimizer_comparison()
    demonstrate_parameter_groups()
    demonstrate_optimizer_state()
    demonstrate_zero_grad()
    demonstrate_gradient_clipping()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Choosing optimizers:")
    print("- Default choice: Adam or AdamW (lr=0.001)")
    print("- Computer vision: SGD with momentum (lr=0.01-0.1)")
    print("- Transformers: AdamW with scheduler")
    print("- Fast prototyping: Adam")
    print("- Best final performance: Often SGD with tuned LR")
    print("=" * 60)


if __name__ == "__main__":
    main()
