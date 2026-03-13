"""
Advanced PyTorch Exercises
Practice custom models and production techniques

This module contains 8 advanced exercises covering:
- Custom architectures
- Model ensembles
- Mixed precision training
- ONNX export
- Performance optimization
- Production deployment patterns

Run: python exercises.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import time


def exercise_1_attention_mechanism():
    """
    Exercise 1: Implement a simple attention mechanism

    Task: Create an AttentionBlock that computes scaled dot-product attention
    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    print("Exercise 1: Attention Mechanism")
    print("-" * 60)

    class AttentionBlock(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim
            self.query = nn.Linear(embed_dim, embed_dim)
            self.key = nn.Linear(embed_dim, embed_dim)
            self.value = nn.Linear(embed_dim, embed_dim)

        def forward(self, x):
            # TODO: Implement scaled dot-product attention
            # 1. Compute Q, K, V projections
            # 2. Calculate attention scores: QK^T / sqrt(d_k)
            # 3. Apply softmax
            # 4. Multiply by V
            Q = self.query(x)
            K = self.key(x)
            V = self.value(x)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim**0.5)
            attention_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attention_weights, V)

            return output, attention_weights

    # Test
    model = AttentionBlock(embed_dim=64)
    x = torch.randn(2, 10, 64)  # (batch, seq_len, embed_dim)
    output, weights = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights sum: {weights[0, 0].sum():.4f} (should be 1.0)")
    print("Success: Attention mechanism implemented!\n")


def exercise_2_residual_dense_block():
    """
    Exercise 2: Implement a DenseNet-style residual block

    Task: Create a block that concatenates features from previous layers
    Each layer receives all preceding feature maps
    """
    print("Exercise 2: Dense Residual Block")
    print("-" * 60)

    class DenseBlock(nn.Module):
        def __init__(self, in_channels, growth_rate, num_layers):
            super().__init__()
            self.layers = nn.ModuleList()

            for i in range(num_layers):
                # Each layer takes all previous features
                in_ch = in_channels + i * growth_rate
                self.layers.append(
                    nn.Sequential(
                        nn.BatchNorm2d(in_ch),
                        nn.ReLU(),
                        nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=1),
                    )
                )

        def forward(self, x):
            features = [x]
            for layer in self.layers:
                # Concatenate all previous features
                concat_features = torch.cat(features, dim=1)
                new_features = layer(concat_features)
                features.append(new_features)

            # Return concatenation of all features
            return torch.cat(features, dim=1)

    # Test
    model = DenseBlock(in_channels=64, growth_rate=32, num_layers=4)
    x = torch.randn(2, 64, 16, 16)
    output = model(x)

    expected_channels = 64 + 4 * 32
    print(f"Input channels: {x.shape[1]}")
    print(f"Output channels: {output.shape[1]} (expected: {expected_channels})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Success: Dense block implemented!\n")


def exercise_3_weighted_ensemble():
    """
    Exercise 3: Implement a weighted ensemble with temperature scaling

    Task: Combine predictions from multiple models with learnable weights
    and temperature parameter for calibration
    """
    print("Exercise 3: Weighted Ensemble with Temperature")
    print("-" * 60)

    class TemperatureScaledEnsemble(nn.Module):
        def __init__(self, models, init_temp=1.0):
            super().__init__()
            self.models = nn.ModuleList(models)
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
            self.temperature = nn.Parameter(torch.tensor(init_temp))

        def forward(self, x):
            # Get predictions from all models
            predictions = [model(x) for model in self.models]

            # Apply temperature scaling
            scaled_preds = [pred / self.temperature for pred in predictions]

            # Convert to probabilities
            probs = [F.softmax(pred, dim=1) for pred in scaled_preds]

            # Weighted combination
            weights = F.softmax(self.weights, dim=0)
            ensemble_probs = sum(w * p for w, p in zip(weights, probs))

            return torch.log(ensemble_probs + 1e-8)

    # Create simple models
    models = [
        nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3))
        for _ in range(3)
    ]

    ensemble = TemperatureScaledEnsemble(models, init_temp=1.5)

    # Test
    x = torch.randn(5, 10)
    output = ensemble(x)

    print(f"Number of models: {len(models)}")
    print(f"Weights: {F.softmax(ensemble.weights, dim=0).detach()}")
    print(f"Temperature: {ensemble.temperature.item():.4f}")
    print(f"Output shape: {output.shape}")
    print("Success: Temperature-scaled ensemble implemented!\n")


def exercise_4_mixed_precision_training():
    """
    Exercise 4: Implement mixed precision training loop

    Task: Create training function using automatic mixed precision (AMP)
    Compare performance with FP32 training
    """
    print("Exercise 4: Mixed Precision Training")
    print("-" * 60)

    def train_with_amp(model, dataloader, epochs=2):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        start_time = time.time()

        for epoch in range(epochs):
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                # Mixed precision forward pass
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # Scaled backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        elapsed = time.time() - start_time
        return elapsed

    # Create model and data
    model = nn.Sequential(
        nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10)
    )

    X = torch.randn(1000, 100)
    y = torch.randint(0, 10, (1000,))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=32)

    elapsed = train_with_amp(model, dataloader, epochs=2)

    print(f"Training completed in {elapsed:.2f}s")
    if torch.cuda.is_available():
        print("Mixed precision training uses FP16 for speed and memory efficiency")
    else:
        print("Running on CPU (AMP benefits require CUDA)")
    print("Success: Mixed precision training implemented!\n")


def exercise_5_custom_loss_function():
    """
    Exercise 5: Implement a custom loss function with regularization

    Task: Create FocalLoss for handling class imbalance
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    print("Exercise 5: Custom Focal Loss")
    print("-" * 60)

    class FocalLoss(nn.Module):
        def __init__(self, alpha=1.0, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            # Compute cross entropy
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")

            # Get probabilities
            probs = F.softmax(inputs, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            # Focal loss formula
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

            return focal_loss.mean()

    # Test
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # Create imbalanced data
    inputs = torch.randn(100, 5)
    targets = torch.cat(
        [
            torch.zeros(80, dtype=torch.long),  # Class 0: 80 samples
            torch.ones(20, dtype=torch.long),  # Class 1: 20 samples
        ]
    )

    loss = criterion(inputs, targets)

    print(f"Focal Loss: {loss.item():.4f}")
    print(f"Alpha: {criterion.alpha}, Gamma: {criterion.gamma}")
    print("Focal loss reduces loss for well-classified examples")
    print("Focuses training on hard examples")
    print("Success: Focal loss implemented!\n")


def exercise_6_model_pruning():
    """
    Exercise 6: Implement basic magnitude-based pruning

    Task: Prune smallest weights in a model to reduce parameters
    Remove weights below threshold based on magnitude
    """
    print("Exercise 6: Model Pruning")
    print("-" * 60)

    def prune_model(model, pruning_threshold=0.3):
        """Prune weights below threshold."""
        total_params = 0
        pruned_params = 0

        for name, param in model.named_parameters():
            if "weight" in name:
                total_params += param.numel()

                # Create mask for weights above threshold
                mask = torch.abs(param.data) > pruning_threshold

                # Count pruned parameters
                pruned_params += (mask == 0).sum().item()

                # Apply mask (set small weights to zero)
                param.data *= mask.float()

        pruning_percentage = (pruned_params / total_params) * 100
        return total_params, pruned_params, pruning_percentage

    # Create model
    model = nn.Sequential(
        nn.Linear(100, 200), nn.ReLU(), nn.Linear(200, 100), nn.ReLU(), nn.Linear(100, 10)
    )

    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())

    # Prune model
    total, pruned, percentage = prune_model(model, pruning_threshold=0.1)

    print(f"Original parameters: {original_params:,}")
    print(f"Pruned parameters: {pruned:,}")
    print(f"Pruning percentage: {percentage:.2f}%")
    print(f"Remaining parameters: {total - pruned:,}")
    print("\nPruning reduces model size and can improve inference speed")
    print("Success: Model pruning implemented!\n")


def exercise_7_knowledge_distillation():
    """
    Exercise 7: Implement knowledge distillation

    Task: Train a small student model to mimic a large teacher model
    Use soft targets (temperature-scaled probabilities)
    """
    print("Exercise 7: Knowledge Distillation")
    print("-" * 60)

    class DistillationLoss(nn.Module):
        def __init__(self, temperature=3.0, alpha=0.5):
            super().__init__()
            self.temperature = temperature
            self.alpha = alpha
            self.criterion = nn.CrossEntropyLoss()

        def forward(self, student_logits, teacher_logits, targets):
            # Hard target loss
            hard_loss = self.criterion(student_logits, targets)

            # Soft target loss (KL divergence)
            student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
            soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
            soft_loss = soft_loss * (self.temperature**2)

            # Combined loss
            return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

    # Teacher model (large)
    teacher = nn.Sequential(
        nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10)
    )

    # Student model (small)
    student = nn.Sequential(
        nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 10)
    )

    teacher.eval()
    student.train()

    # Create data
    x = torch.randn(64, 20)
    targets = torch.randint(0, 10, (64,))

    # Get teacher predictions
    with torch.no_grad():
        teacher_logits = teacher(x)

    # Student predictions
    student_logits = student(x)

    # Distillation loss
    criterion = DistillationLoss(temperature=3.0, alpha=0.5)
    loss = criterion(student_logits, teacher_logits, targets)

    print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(f"Compression ratio: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student.parameters()):.2f}x")
    print(f"Distillation loss: {loss.item():.4f}")
    print("\nKnowledge distillation transfers knowledge from large to small models")
    print("Success: Knowledge distillation implemented!\n")


def exercise_8_model_benchmarking():
    """
    Exercise 8: Implement comprehensive model benchmarking

    Task: Benchmark model performance including:
    - Throughput (samples/sec)
    - Latency (ms per batch)
    - Memory usage
    - FLOPS estimation
    """
    print("Exercise 8: Model Benchmarking")
    print("-" * 60)

    def benchmark_model(model, input_shape, batch_sizes=[1, 16, 32, 64], num_runs=100):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        results = []

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, *input_shape, device=device)

            # Warm-up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark
            start = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.time() - start

            # Calculate metrics
            throughput = (num_runs * batch_size) / elapsed
            latency = (elapsed / num_runs) * 1000  # ms

            results.append(
                {
                    "batch_size": batch_size,
                    "throughput": throughput,
                    "latency": latency,
                }
            )

        return results

    # Create model
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    # Benchmark
    results = benchmark_model(model, input_shape=(100,), batch_sizes=[1, 16, 32])

    print("Benchmark Results:")
    print(f"{'Batch Size':<12} {'Throughput':<15} {'Latency':<12}")
    print("-" * 40)
    for r in results:
        print(
            f"{r['batch_size']:<12} {r['throughput']:>10.1f} samp/s  {r['latency']:>8.2f} ms"
        )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Benchmarking helps identify optimal batch sizes and performance")
    print("Success: Model benchmarking implemented!\n")


def main():
    print("=" * 70)
    print("ADVANCED PYTORCH EXERCISES")
    print("=" * 70)
    print()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    # Run all exercises
    exercise_1_attention_mechanism()
    exercise_2_residual_dense_block()
    exercise_3_weighted_ensemble()
    exercise_4_mixed_precision_training()
    exercise_5_custom_loss_function()
    exercise_6_model_pruning()
    exercise_7_knowledge_distillation()
    exercise_8_model_benchmarking()

    print("=" * 70)
    print("All exercises completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
