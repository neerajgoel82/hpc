"""
PyTorch Lightning
Simplified training with the Lightning framework

This module demonstrates:
- LightningModule basics
- Trainer configuration
- Automatic logging and checkpointing
- Multi-GPU training with Lightning
- Callbacks and hooks

Note: Install with: pip install pytorch-lightning
Run: python 05_pytorch_lightning.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("PyTorch Lightning not installed. Install with: pip install pytorch-lightning")


# LightningModule implementation
if LIGHTNING_AVAILABLE:

    class LitClassifier(pl.LightningModule):
        """PyTorch Lightning model for classification."""

        def __init__(self, input_dim=100, hidden_dim=256, num_classes=10, lr=0.001):
            super().__init__()
            self.save_hyperparameters()

            # Model architecture
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes),
            )

            self.criterion = nn.CrossEntropyLoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            """Training step (called automatically)."""
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)

            # Log metrics
            self.log("train_loss", loss, prog_bar=True)

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            self.log("train_acc", acc, prog_bar=True)

            return loss

        def validation_step(self, batch, batch_idx):
            """Validation step (called automatically)."""
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)

            self.log("val_loss", loss, prog_bar=True)

            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            self.log("val_acc", acc, prog_bar=True)

            return loss

        def test_step(self, batch, batch_idx):
            """Test step (called automatically)."""
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)

            self.log("test_loss", loss)

            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            self.log("test_acc", acc)

            return loss

        def configure_optimizers(self):
            """Configure optimizer and scheduler."""
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

            # Optional: add learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }

    class LitDataModule(pl.LightningDataModule):
        """PyTorch Lightning DataModule."""

        def __init__(self, batch_size=32, num_samples=10000):
            super().__init__()
            self.batch_size = batch_size
            self.num_samples = num_samples

        def setup(self, stage=None):
            """Create datasets."""
            # Generate synthetic data
            X = torch.randn(self.num_samples, 100)
            y = torch.randint(0, 10, (self.num_samples,))
            dataset = TensorDataset(X, y)

            # Split into train, val, test
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_size = len(dataset) - train_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )

        def train_dataloader(self):
            return DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )

        def val_dataloader(self):
            return DataLoader(self.val_dataset, batch_size=self.batch_size)

        def test_dataloader(self):
            return DataLoader(self.test_dataset, batch_size=self.batch_size)


def demonstrate_basic_lightning():
    """Demonstrate basic PyTorch Lightning usage."""
    print("1. Basic PyTorch Lightning")
    print("-" * 60)

    if not LIGHTNING_AVAILABLE:
        print("PyTorch Lightning not available.\n")
        return

    # Create model and data
    model = LitClassifier(input_dim=100, hidden_dim=256, num_classes=10, lr=0.001)
    datamodule = LitDataModule(batch_size=32, num_samples=1000)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",  # Automatically uses GPU if available
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print("Training with PyTorch Lightning...")
    trainer.fit(model, datamodule)

    print("\nLightning automatically handles:")
    print("  - Training/validation loops")
    print("  - GPU/CPU transfer")
    print("  - Gradient accumulation")
    print("  - Logging")
    print("  - Checkpointing")
    print()


def demonstrate_callbacks():
    """Demonstrate Lightning callbacks."""
    print("2. Callbacks and Checkpointing")
    print("-" * 60)

    if not LIGHTNING_AVAILABLE:
        print("PyTorch Lightning not available.\n")
        return

    model = LitClassifier()
    datamodule = LitDataModule(batch_size=32, num_samples=1000)

    # Define callbacks
    early_stop = EarlyStopping(
        monitor="val_loss", patience=3, mode="min", verbose=True
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    # Trainer with callbacks
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        callbacks=[early_stop, checkpoint],
        enable_progress_bar=False,
    )

    print("Training with callbacks...")
    trainer.fit(model, datamodule)

    print(f"\nBest model saved at: {checkpoint.best_model_path}")
    print(f"Best val_loss: {checkpoint.best_model_score:.4f}")
    print()


def demonstrate_multi_gpu():
    """Demonstrate multi-GPU training with Lightning."""
    print("3. Multi-GPU Training")
    print("-" * 60)

    if not LIGHTNING_AVAILABLE:
        print("PyTorch Lightning not available.\n")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    if num_gpus < 2:
        print("Multi-GPU requires 2+ GPUs")
        print("Lightning automatically uses DDP for multi-GPU training")
    else:
        model = LitClassifier()
        datamodule = LitDataModule(batch_size=32, num_samples=1000)

        # Multi-GPU trainer
        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="gpu",
            devices=num_gpus,
            strategy="ddp",  # DistributedDataParallel
            enable_progress_bar=False,
        )

        print(f"Training on {num_gpus} GPUs with DDP...")
        trainer.fit(model, datamodule)

    print("\nLightning multi-GPU features:")
    print("  - Automatic DDP setup")
    print("  - No manual process initialization")
    print("  - Works with multiple strategies (ddp, ddp_spawn, etc.)")
    print()


def demonstrate_lightning_benefits():
    """Demonstrate benefits of PyTorch Lightning."""
    print("4. Benefits of PyTorch Lightning")
    print("-" * 60)

    print("PyTorch Lightning Benefits:")
    print()
    print("1. Less Boilerplate:")
    print("   - No manual training loops")
    print("   - Automatic GPU handling")
    print("   - Built-in logging")
    print()
    print("2. Better Organization:")
    print("   - Clear separation of research code")
    print("   - Reproducible experiments")
    print("   - Standardized structure")
    print()
    print("3. Advanced Features:")
    print("   - Easy multi-GPU/TPU training")
    print("   - Mixed precision training")
    print("   - Gradient accumulation")
    print("   - Early stopping")
    print()
    print("4. Production Ready:")
    print("   - Checkpointing")
    print("   - Logging (TensorBoard, Wandb, etc.)")
    print("   - Profiling tools")
    print("   - ONNX export")
    print()


def demonstrate_comparison():
    """Compare PyTorch vs Lightning code."""
    print("5. PyTorch vs Lightning Comparison")
    print("-" * 60)

    print("Pure PyTorch (manual):")
    print("  - Define model")
    print("  - Write training loop")
    print("  - Handle GPU transfer")
    print("  - Implement validation")
    print("  - Add logging manually")
    print("  - Handle checkpointing")
    print("  - ~100+ lines of code")
    print()
    print("PyTorch Lightning (automatic):")
    print("  - Define LightningModule")
    print("  - Implement training_step()")
    print("  - Implement validation_step()")
    print("  - Trainer handles everything")
    print("  - ~50 lines of code")
    print()
    print("When to use Lightning:")
    print("  + Complex training setups")
    print("  + Multi-GPU/distributed training")
    print("  + Research projects")
    print("  + Production deployment")
    print()
    print("When to use pure PyTorch:")
    print("  + Learning PyTorch basics")
    print("  + Custom training logic")
    print("  + Very simple projects")
    print("  + Maximum control needed")
    print()


def main():
    print("=" * 70)
    print("PYTORCH LIGHTNING FRAMEWORK")
    print("=" * 70)
    print()

    print(f"PyTorch version: {torch.__version__}")
    if LIGHTNING_AVAILABLE:
        print(f"Lightning version: {pl.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    if not LIGHTNING_AVAILABLE:
        print("Install PyTorch Lightning:")
        print("  pip install pytorch-lightning")
        print()

    demonstrate_basic_lightning()
    demonstrate_callbacks()
    demonstrate_multi_gpu()
    demonstrate_lightning_benefits()
    demonstrate_comparison()

    print("=" * 70)


if __name__ == "__main__":
    main()
