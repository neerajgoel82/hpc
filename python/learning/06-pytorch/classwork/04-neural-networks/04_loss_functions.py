"""
Loss Functions
MSE, CrossEntropy, and custom losses

This module demonstrates:
- Common loss functions
- Regression losses (MSE, MAE, Huber)
- Classification losses (CrossEntropy, NLL, BCE)
- Custom loss functions
- Loss reduction modes

Run: python 04_loss_functions.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Custom Focal Loss for imbalanced classification"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


def demonstrate_mse_loss():
    """Mean Squared Error loss for regression"""
    print("\n1. MSE Loss (Regression)")
    print("-" * 60)

    mse_loss = nn.MSELoss()

    # Predictions and targets
    predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
    targets = torch.tensor([3.0, -0.5, 2.0, 7.0])

    loss = mse_loss(predictions, targets)
    print(f"Predictions: {predictions}")
    print(f"Targets:     {targets}")
    print(f"MSE Loss:    {loss.item():.4f}")

    # Manual calculation
    manual_loss = ((predictions - targets) ** 2).mean()
    print(f"Manual MSE:  {manual_loss.item():.4f}")


def demonstrate_mae_loss():
    """Mean Absolute Error loss"""
    print("\n2. MAE Loss (L1 Loss)")
    print("-" * 60)

    mae_loss = nn.L1Loss()

    predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
    targets = torch.tensor([3.0, -0.5, 2.0, 7.0])

    loss = mae_loss(predictions, targets)
    print(f"Predictions: {predictions}")
    print(f"Targets:     {targets}")
    print(f"MAE Loss:    {loss.item():.4f}")

    print("\nMAE is less sensitive to outliers than MSE")


def demonstrate_huber_loss():
    """Huber loss (smooth L1)"""
    print("\n3. Huber Loss (Smooth L1)")
    print("-" * 60)

    huber_loss = nn.SmoothL1Loss()

    predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
    targets = torch.tensor([3.0, -0.5, 2.0, 7.0])

    loss = huber_loss(predictions, targets)
    print(f"Predictions: {predictions}")
    print(f"Targets:     {targets}")
    print(f"Huber Loss:  {loss.item():.4f}")

    print("\nHuber combines MSE (small errors) and MAE (large errors)")


def demonstrate_cross_entropy():
    """Cross Entropy loss for multi-class classification"""
    print("\n4. Cross Entropy Loss")
    print("-" * 60)

    ce_loss = nn.CrossEntropyLoss()

    # Raw logits (before softmax)
    logits = torch.tensor([[2.0, 1.0, 0.1],
                           [0.5, 2.5, 0.3]])
    targets = torch.tensor([0, 1])  # Class indices

    loss = ce_loss(logits, targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Targets: {targets}")
    print(f"CE Loss: {loss.item():.4f}")

    # Show probabilities
    probs = F.softmax(logits, dim=1)
    print(f"\nProbabilities:\n{probs}")
    print("\nNote: CrossEntropyLoss combines LogSoftmax + NLLLoss")


def demonstrate_nll_loss():
    """Negative Log Likelihood loss"""
    print("\n5. NLL Loss")
    print("-" * 60)

    nll_loss = nn.NLLLoss()

    # Log probabilities (after log_softmax)
    log_probs = torch.tensor([[-0.5, -1.5, -2.5],
                               [-2.0, -0.5, -2.5]])
    targets = torch.tensor([0, 1])

    loss = nll_loss(log_probs, targets)
    print(f"Log probs: {log_probs}")
    print(f"Targets: {targets}")
    print(f"NLL Loss: {loss.item():.4f}")

    print("\nNLL expects log probabilities as input")


def demonstrate_bce_loss():
    """Binary Cross Entropy loss"""
    print("\n6. Binary Cross Entropy Loss")
    print("-" * 60)

    bce_loss = nn.BCELoss()

    # Predictions (after sigmoid, range [0, 1])
    predictions = torch.tensor([0.9, 0.2, 0.7, 0.1])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss = bce_loss(predictions, targets)
    print(f"Predictions: {predictions}")
    print(f"Targets:     {targets}")
    print(f"BCE Loss:    {loss.item():.4f}")

    print("\nBCE expects probabilities (after sigmoid)")


def demonstrate_bce_with_logits():
    """BCE with logits (numerically stable)"""
    print("\n7. BCE with Logits")
    print("-" * 60)

    bce_logits_loss = nn.BCEWithLogitsLoss()

    # Raw logits (before sigmoid)
    logits = torch.tensor([2.0, -1.5, 1.0, -2.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss = bce_logits_loss(logits, targets)
    print(f"Logits:  {logits}")
    print(f"Targets: {targets}")
    print(f"BCE with Logits Loss: {loss.item():.4f}")

    print("\nMore numerically stable than BCE(sigmoid(x))")


def demonstrate_reduction_modes():
    """Loss reduction modes"""
    print("\n8. Reduction Modes")
    print("-" * 60)

    predictions = torch.tensor([2.0, 0.0, 1.0])
    targets = torch.tensor([3.0, -0.5, 2.0])

    # Mean (default)
    loss_mean = nn.MSELoss(reduction='mean')(predictions, targets)
    print(f"Mean reduction: {loss_mean.item():.4f}")

    # Sum
    loss_sum = nn.MSELoss(reduction='sum')(predictions, targets)
    print(f"Sum reduction:  {loss_sum.item():.4f}")

    # None (per-element)
    loss_none = nn.MSELoss(reduction='none')(predictions, targets)
    print(f"None reduction: {loss_none}")


def demonstrate_weighted_loss():
    """Weighted loss for imbalanced classes"""
    print("\n9. Weighted Loss")
    print("-" * 60)

    # Class weights for imbalanced dataset
    weights = torch.tensor([1.0, 2.0, 5.0])  # Class 2 is rare
    ce_weighted = nn.CrossEntropyLoss(weight=weights)

    logits = torch.tensor([[2.0, 1.0, 0.1],
                           [0.5, 2.5, 0.3],
                           [0.1, 0.5, 2.0]])
    targets = torch.tensor([0, 1, 2])

    loss_weighted = ce_weighted(logits, targets)
    loss_unweighted = nn.CrossEntropyLoss()(logits, targets)

    print(f"Unweighted loss: {loss_unweighted.item():.4f}")
    print(f"Weighted loss:   {loss_weighted.item():.4f}")
    print("\nWeighting helps with class imbalance")


def demonstrate_focal_loss():
    """Custom Focal Loss"""
    print("\n10. Focal Loss (Custom)")
    print("-" * 60)

    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    ce_loss = nn.CrossEntropyLoss()

    logits = torch.tensor([[3.0, 0.5, 0.1],
                           [0.5, 3.0, 0.3]])
    targets = torch.tensor([0, 1])

    loss_focal = focal_loss(logits, targets)
    loss_ce = ce_loss(logits, targets)

    print(f"Cross Entropy: {loss_ce.item():.4f}")
    print(f"Focal Loss:    {loss_focal.item():.4f}")
    print("\nFocal Loss focuses on hard examples")


def demonstrate_dice_loss():
    """Dice Loss for segmentation"""
    print("\n11. Dice Loss (Custom)")
    print("-" * 60)

    dice_loss = DiceLoss()

    # Binary segmentation (simplified)
    predictions = torch.tensor([2.0, -1.0, 1.5, -0.5])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss = dice_loss(predictions, targets)
    print(f"Dice Loss: {loss.item():.4f}")
    print("\nDice Loss measures overlap between prediction and target")


def main():
    print("=" * 60)
    print("Loss Functions")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    demonstrate_mse_loss()
    demonstrate_mae_loss()
    demonstrate_huber_loss()
    demonstrate_cross_entropy()
    demonstrate_nll_loss()
    demonstrate_bce_loss()
    demonstrate_bce_with_logits()
    demonstrate_reduction_modes()
    demonstrate_weighted_loss()
    demonstrate_focal_loss()
    demonstrate_dice_loss()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Choosing loss functions:")
    print("- Regression: MSELoss, L1Loss, SmoothL1Loss")
    print("- Multi-class: CrossEntropyLoss (includes softmax)")
    print("- Binary: BCEWithLogitsLoss (includes sigmoid)")
    print("- Imbalanced: Weighted loss or Focal Loss")
    print("- Segmentation: Dice Loss, IoU Loss")
    print("=" * 60)


if __name__ == "__main__":
    main()
