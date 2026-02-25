"""
Distributed Training Basics
Multi-GPU training with DistributedDataParallel

This module demonstrates:
- DataParallel vs DistributedDataParallel
- Basic distributed training setup
- Multi-GPU training patterns
- Process groups and initialization
- Gradient synchronization

Run: python 04_distributed_basics.py
Multi-GPU run: torchrun --nproc_per_node=2 04_distributed_basics.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os


class SimpleNet(nn.Module):
    """Simple network for distributed training demonstration."""

    def __init__(self, input_dim=100, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def create_dataset(num_samples=1000):
    """Create synthetic dataset."""
    X = torch.randn(num_samples, 100)
    y = torch.randint(0, 10, (num_samples,))
    return TensorDataset(X, y)


def demonstrate_dataparallel():
    """Demonstrate DataParallel (simple but less efficient)."""
    print("1. DataParallel (Simple Multi-GPU)")
    print("-" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping DataParallel demo.\n")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    if num_gpus < 2:
        print("DataParallel works best with 2+ GPUs")
        print("Running on single GPU for demonstration")

    # Create model
    model = SimpleNet()
    print(f"Original model device: cpu")

    # Wrap with DataParallel
    if num_gpus > 0:
        model = model.cuda()
        if num_gpus > 1:
            model = DataParallel(model)
            print(f"Model wrapped with DataParallel")
            print(f"Using GPUs: {list(range(num_gpus))}")

    # Create data
    dataset = create_dataset(1000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Simple training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for batch_idx, (inputs, targets) in enumerate(loader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            print(f"Batch shape: {inputs.shape}")
            print(f"Output shape: {outputs.shape}")
            break

    print("DataParallel advantages:")
    print("  + Very easy to use (single line change)")
    print("  + No code restructuring needed")
    print("DataParallel disadvantages:")
    print("  - Less efficient than DDP")
    print("  - GIL contention in multi-threaded mode")
    print("  - Uneven GPU utilization (GPU 0 bottleneck)")
    print()


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        print("Not running in distributed mode")
        return None, None, None

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_distributed(rank, world_size, local_rank):
    """Training function for distributed setup."""
    print(f"[Rank {rank}] Starting training on GPU {local_rank}")

    # Create model and move to GPU
    model = SimpleNet().cuda()
    model = DDP(model, device_ids=[local_rank])

    # Create dataset with DistributedSampler
    dataset = create_dataset(1000)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(2):
        sampler.set_epoch(epoch)  # Important for proper shuffling

        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"[Rank {rank}] Epoch {epoch+1} complete")

    print(f"[Rank {rank}] Training complete")


def demonstrate_ddp():
    """Demonstrate DistributedDataParallel."""
    print("2. DistributedDataParallel (Efficient Multi-GPU)")
    print("-" * 60)

    rank, world_size, local_rank = setup_distributed()

    if rank is None:
        print("DDP requires torchrun or torch.distributed.launch")
        print("\nTo run DDP, use:")
        print("  torchrun --nproc_per_node=2 04_distributed_basics.py")
        print("\nOr with specific GPUs:")
        print("  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 04_distributed_basics.py")
        print("\nDDP advantages:")
        print("  + Much more efficient than DataParallel")
        print("  + Better GPU utilization")
        print("  + Scales to multiple nodes")
        print("  + Each process has its own Python interpreter (no GIL)")
        print("\nDDP requirements:")
        print("  - More complex setup")
        print("  - Need to use DistributedSampler")
        print("  - Must initialize process group")
        print()
        return

    train_distributed(rank, world_size, local_rank)
    cleanup_distributed()
    print()


def demonstrate_gradient_sync():
    """Demonstrate gradient synchronization in DDP."""
    print("3. Gradient Synchronization")
    print("-" * 60)

    print("In DistributedDataParallel:")
    print()
    print("1. Forward pass: Each GPU processes different batch")
    print("2. Backward pass: Gradients computed locally")
    print("3. Gradient sync: Gradients averaged across all GPUs")
    print("4. Optimizer step: Each GPU updates with averaged gradients")
    print()
    print("Communication pattern:")
    print("  - Uses NCCL backend for GPU communication")
    print("  - All-reduce operation for gradient averaging")
    print("  - Overlaps computation and communication")
    print()
    print("Tips for efficient DDP:")
    print("  - Use gradient accumulation for large models")
    print("  - Set find_unused_parameters=False if possible")
    print("  - Use torch.cuda.amp for mixed precision")
    print("  - Profile with torch.profiler")
    print()


def demonstrate_process_group():
    """Demonstrate process group concepts."""
    print("4. Process Groups and Communication")
    print("-" * 60)

    if not dist.is_initialized():
        print("Process group concepts:")
        print()
        print("Rank: Unique identifier for each process (0 to world_size-1)")
        print("World Size: Total number of processes")
        print("Local Rank: GPU index on current node")
        print()
        print("Backends:")
        print("  - NCCL: Best for GPU training (NVIDIA)")
        print("  - Gloo: CPU training or mixed CPU/GPU")
        print("  - MPI: Research clusters")
        print()
        print("Environment variables:")
        print("  - RANK: Global rank")
        print("  - WORLD_SIZE: Total processes")
        print("  - LOCAL_RANK: GPU on current node")
        print("  - MASTER_ADDR: Master node address")
        print("  - MASTER_PORT: Communication port")
        print()
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Process rank: {rank}")
        print(f"World size: {world_size}")
        print(f"Backend: {dist.get_backend()}")
        print()


def demonstrate_best_practices():
    """Demonstrate best practices for distributed training."""
    print("5. Best Practices")
    print("-" * 60)

    print("1. Data Loading:")
    print("   - Use DistributedSampler")
    print("   - Set num_workers > 0 for DataLoader")
    print("   - Call sampler.set_epoch() each epoch")
    print()
    print("2. Model Setup:")
    print("   - Move model to GPU before wrapping with DDP")
    print("   - Use sync_batch_norm for BatchNorm layers")
    print("   - Save/load only from rank 0")
    print()
    print("3. Training:")
    print("   - Use mixed precision (torch.cuda.amp)")
    print("   - Gradient accumulation for large models")
    print("   - Monitor all ranks for debugging")
    print()
    print("4. Common Pitfalls:")
    print("   - Forgetting to set sampler epoch")
    print("   - Not handling randomness properly")
    print("   - Saving from all ranks (causes conflicts)")
    print("   - Using different random seeds across ranks")
    print()


def main():
    print("=" * 70)
    print("DISTRIBUTED TRAINING BASICS")
    print("=" * 70)
    print()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    print()

    demonstrate_dataparallel()
    demonstrate_ddp()
    demonstrate_gradient_sync()
    demonstrate_process_group()
    demonstrate_best_practices()

    print("=" * 70)


if __name__ == "__main__":
    main()
