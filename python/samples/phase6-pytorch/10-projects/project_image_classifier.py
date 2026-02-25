"""
Project: Image Classifier
=========================
Build a complete image classification system.

Dataset: Custom image dataset (or CIFAR-10/CIFAR-100)
Goals:
- Load and preprocess images
- Build CNN architecture
- Train with data augmentation
- Evaluate on test set
- Visualize results and predictions

Skills: CNNs, Data loaders, Training pipelines
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # TODO: Define architecture
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass

def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    # TODO: Implement training loop
    pass

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    # TODO: Implement evaluation
    pass

def main():
    print("=" * 60)
    print("Image Classifier Project")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # TODO: Complete implementation
    print("\nTODO: Implement image classification pipeline")
    print("=" * 60)

if __name__ == "__main__":
    main()
