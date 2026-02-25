"""
Mixed Precision Training
FP16 training with torch.cuda.amp for speed and memory efficiency

This module demonstrates:
- Automatic Mixed Precision (AMP) with torch.cuda.amp
- GradScaler for handling gradient scaling
- Performance comparison FP32 vs FP16
- Memory usage optimization
- Best practices for mixed precision training

Run: python 03_mixed_precision.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
import time


class ConvNet(nn.Module):
    """Convolutional network for demonstration."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_dummy_data(num_samples=1000, img_size=32):
    """Create dummy image data."""
    X = torch.randn(num_samples, 3, img_size, img_size)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def train_fp32(model, train_loader, device, epochs=3):
    """Train with FP32 (standard precision)."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

    elapsed = time.time() - start_time
    return elapsed, avg_loss


def train_mixed_precision(model, train_loader, device, epochs=3):
    """Train with mixed precision (FP16 + FP32)."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()  # Gradient scaler for mixed precision

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass with autocast
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

    elapsed = time.time() - start_time
    return elapsed, avg_loss


def demonstrate_basic_amp():
    """Demonstrate basic automatic mixed precision usage."""
    print("1. Basic Automatic Mixed Precision (AMP)")
    print("-" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not torch.cuda.is_available():
        print("CUDA not available. AMP requires CUDA.")
        print("Running on CPU (no speedup expected)\n")
        return

    # Create model and data
    model = ConvNet()
    train_loader = create_dummy_data(num_samples=1000)

    # Compare FP32 vs mixed precision
    print("\nTraining with FP32...")
    model_fp32 = ConvNet()
    time_fp32, loss_fp32 = train_fp32(model_fp32, train_loader, device, epochs=3)
    print(f"FP32 - Time: {time_fp32:.2f}s, Loss: {loss_fp32:.4f}")

    print("\nTraining with Mixed Precision...")
    model_amp = ConvNet()
    time_amp, loss_amp = train_mixed_precision(model_amp, train_loader, device, epochs=3)
    print(f"AMP  - Time: {time_amp:.2f}s, Loss: {loss_amp:.4f}")

    speedup = time_fp32 / time_amp if time_amp > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")
    print()


def demonstrate_gradscaler():
    """Demonstrate gradient scaler functionality."""
    print("2. Gradient Scaler")
    print("-" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GradScaler demo.\n")
        return

    model = nn.Linear(100, 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scaler = GradScaler()

    print("GradScaler properties:")
    print(f"Initial scale: {scaler.get_scale()}")
    print(f"Growth factor: {scaler._growth_factor}")
    print(f"Backoff factor: {scaler._backoff_factor}")
    print(f"Growth interval: {scaler._growth_interval}")

    # Simulate training steps
    for step in range(5):
        optimizer.zero_grad()

        with autocast():
            x = torch.randn(32, 100, device=device)
            output = model(x)
            loss = output.sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 2 == 0:
            print(f"Step {step}: Scale = {scaler.get_scale()}")

    print()


def demonstrate_memory_usage():
    """Demonstrate memory efficiency of mixed precision."""
    print("3. Memory Efficiency")
    print("-" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("CUDA not available. Cannot measure GPU memory.\n")
        return

    # Create larger model
    model = nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1000),
    )

    # Measure FP32 memory
    torch.cuda.reset_peak_memory_stats()
    model_fp32 = model.to(device)
    x = torch.randn(128, 1000, device=device)
    _ = model_fp32(x)
    mem_fp32 = torch.cuda.max_memory_allocated() / 1024**2

    # Measure mixed precision memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_amp = model.to(device)
    with autocast():
        _ = model_amp(x)
    mem_amp = torch.cuda.max_memory_allocated() / 1024**2

    print(f"FP32 memory: {mem_fp32:.2f} MB")
    print(f"AMP memory:  {mem_amp:.2f} MB")
    print(f"Reduction:   {(1 - mem_amp/mem_fp32)*100:.1f}%")
    print()


def demonstrate_best_practices():
    """Demonstrate best practices for mixed precision training."""
    print("4. Best Practices")
    print("-" * 60)

    print("Best Practices for Mixed Precision Training:")
    print()
    print("1. Use GradScaler for gradient scaling")
    print("   - Prevents underflow in FP16 gradients")
    print("   - Automatically adjusts scale factor")
    print()
    print("2. Wrap forward pass in autocast() context")
    print("   - Model forward pass")
    print("   - Loss computation")
    print("   - Don't wrap backward pass")
    print()
    print("3. Operations automatically use FP16 when safe:")
    print("   - Matrix multiplications")
    print("   - Convolutions")
    print("   - Linear layers")
    print()
    print("4. Operations stay in FP32 when needed:")
    print("   - Batch normalization")
    print("   - Loss functions")
    print("   - Reductions (sum, mean)")
    print()
    print("5. For maximum performance:")
    print("   - Use Tensor Cores (GPU compute capability >= 7.0)")
    print("   - Use larger batch sizes")
    print("   - Enable cuDNN autotuner: torch.backends.cudnn.benchmark=True")
    print()


def demonstrate_manual_precision():
    """Demonstrate manual precision control."""
    print("5. Manual Precision Control")
    print("-" * 60)

    # Automatic vs manual casting
    x_fp32 = torch.randn(3, 3)
    print(f"Original dtype: {x_fp32.dtype}")

    # Manual casting
    x_fp16 = x_fp32.half()
    print(f"Manual cast to FP16: {x_fp16.dtype}")

    # Back to FP32
    x_back = x_fp16.float()
    print(f"Cast back to FP32: {x_back.dtype}")

    # Autocast (automatic)
    with autocast():
        y = torch.matmul(x_fp32, x_fp32)
        print(f"\nInside autocast context:")
        print(f"  Input dtype: {x_fp32.dtype}")
        print(f"  Output dtype: {y.dtype}")

    print()


def main():
    print("=" * 70)
    print("MIXED PRECISION TRAINING")
    print("=" * 70)
    print()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    demonstrate_basic_amp()
    demonstrate_gradscaler()
    demonstrate_memory_usage()
    demonstrate_best_practices()
    demonstrate_manual_precision()

    print("=" * 70)


if __name__ == "__main__":
    main()
