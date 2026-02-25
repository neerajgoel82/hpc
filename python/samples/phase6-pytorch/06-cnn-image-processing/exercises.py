"""
CNN Image Processing - Exercises

Complete these exercises to practice CNN concepts:
1. Custom CNN Architecture
2. Data Augmentation Effects
3. Batch Normalization Impact
4. Receptive Field Calculation
5. Feature Map Visualization
6. Transfer Learning Comparison
7. Model Ensemble
8. Grad-CAM Visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# EXERCISE 1: Custom CNN Architecture
# ============================================================================

def exercise1_custom_cnn():
    """
    Exercise 1: Design a custom CNN for CIFAR-10

    Requirements:
    - At least 3 convolutional blocks
    - Each block: Conv -> BatchNorm -> ReLU -> MaxPool
    - Use dropout for regularization
    - Final accuracy should be > 70% on test set

    TODO: Complete the CustomCNN class
    """
    print("=" * 60)
    print("Exercise 1: Custom CNN Architecture")
    print("=" * 60)

    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()

            # TODO: Implement your architecture here
            # Block 1: 3 -> 32 channels
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)

            # Block 2: 32 -> 64 channels
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)

            # Block 3: 64 -> 128 channels
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(2, 2)

            # Fully connected layers
            self.fc1 = nn.Linear(128 * 4 * 4, 256)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            # TODO: Implement forward pass
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)

            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)

            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool3(x)

            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

            return x

    model = CustomCNN()
    print(f"\nModel created!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    print("\nTODO: Train this model on CIFAR-10 and achieve >70% accuracy")

    return model


# ============================================================================
# EXERCISE 2: Data Augmentation Effects
# ============================================================================

def exercise2_augmentation_effects():
    """
    Exercise 2: Compare model performance with different augmentation strategies

    TODO: Implement three different augmentation strategies:
    1. No augmentation (baseline)
    2. Light augmentation (flip only)
    3. Heavy augmentation (flip, crop, rotation, color jitter)

    Compare their effects on training and test accuracy
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Data Augmentation Effects")
    print("=" * 60)

    # TODO: Implement different transform pipelines
    transform_none = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_light = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_heavy = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("\nAugmentation strategies defined:")
    print("1. None: Just normalize")
    print("2. Light: Horizontal flip")
    print("3. Heavy: Flip + Crop + Rotation + ColorJitter")

    print("\nTODO: Train models with each strategy and compare results")
    print("Expected: Heavy > Light > None (on test set)")


# ============================================================================
# EXERCISE 3: Batch Normalization Impact
# ============================================================================

def exercise3_batchnorm_impact():
    """
    Exercise 3: Measure the impact of Batch Normalization

    TODO: Create two versions of the same network:
    1. With BatchNorm layers
    2. Without BatchNorm layers

    Compare:
    - Training speed (epochs to convergence)
    - Final accuracy
    - Training stability (loss variance)
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Batch Normalization Impact")
    print("=" * 60)

    class CNNWithBN(nn.Module):
        def __init__(self):
            super(CNNWithBN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
            x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    class CNNWithoutBN(nn.Module):
        def __init__(self):
            super(CNNWithoutBN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            # No BatchNorm
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            # No BatchNorm
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model_with_bn = CNNWithBN()
    model_without_bn = CNNWithoutBN()

    print("\nModels created:")
    print(f"With BN params: {sum(p.numel() for p in model_with_bn.parameters()):,}")
    print(f"Without BN params: {sum(p.numel() for p in model_without_bn.parameters()):,}")

    print("\nTODO: Train both models and compare:")
    print("- Convergence speed")
    print("- Final accuracy")
    print("- Loss curve smoothness")


# ============================================================================
# EXERCISE 4: Receptive Field Calculation
# ============================================================================

def exercise4_receptive_field():
    """
    Exercise 4: Calculate receptive field of CNN layers

    TODO: For a given CNN architecture, calculate:
    1. Receptive field at each layer
    2. Final receptive field size

    Formula: RF_out = RF_in + (kernel_size - 1) * stride_product
    """
    print("\n" + "=" * 60)
    print("Exercise 4: Receptive Field Calculation")
    print("=" * 60)

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            return x

    print("\nArchitecture:")
    print("Layer 1: Conv3x3 (stride=1)")
    print("Layer 2: MaxPool2x2 (stride=2)")
    print("Layer 3: Conv3x3 (stride=1)")
    print("Layer 4: MaxPool2x2 (stride=2)")
    print("Layer 5: Conv3x3 (stride=1)")

    # TODO: Calculate receptive field
    print("\nTODO: Calculate receptive field at each layer")
    print("Hint: Start with RF=1, accumulate using formula")

    # Solution (uncomment to check):
    # Layer 1: RF = 1 + (3-1)*1 = 3
    # Layer 2: RF = 3 + (2-1)*1 = 4
    # Layer 3: RF = 4 + (3-1)*2 = 8
    # Layer 4: RF = 8 + (2-1)*2 = 10
    # Layer 5: RF = 10 + (3-1)*4 = 18


# ============================================================================
# EXERCISE 5: Feature Map Visualization
# ============================================================================

def exercise5_feature_visualization():
    """
    Exercise 5: Visualize feature maps from different layers

    TODO: Extract and visualize feature maps from:
    1. First convolutional layer
    2. Middle convolutional layer
    3. Last convolutional layer

    Observe how features become more abstract in deeper layers
    """
    print("\n" + "=" * 60)
    print("Exercise 5: Feature Map Visualization")
    print("=" * 60)

    # Load pretrained model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Get a sample image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\nTODO: Extract feature maps using hooks")
    print("1. Register forward hooks on target layers")
    print("2. Forward pass an image")
    print("3. Visualize activations from each layer")

    # Hint: Use register_forward_hook
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks (example)
    model.layer1[0].conv1.register_forward_hook(get_activation('layer1'))
    model.layer2[0].conv1.register_forward_hook(get_activation('layer2'))
    model.layer3[0].conv1.register_forward_hook(get_activation('layer3'))

    print("\nHooks registered on layer1, layer2, layer3")
    print("TODO: Pass image and visualize activations")


# ============================================================================
# EXERCISE 6: Transfer Learning Comparison
# ============================================================================

def exercise6_transfer_comparison():
    """
    Exercise 6: Compare different transfer learning strategies

    TODO: Train three models on a small custom dataset:
    1. From scratch (random initialization)
    2. Feature extraction (freeze all, train FC only)
    3. Fine-tuning (train all with small LR)

    Compare:
    - Training time
    - Final accuracy
    - Number of epochs to converge
    """
    print("\n" + "=" * 60)
    print("Exercise 6: Transfer Learning Comparison")
    print("=" * 60)

    # Model 1: From scratch
    model_scratch = models.resnet18(pretrained=False)
    model_scratch.fc = nn.Linear(512, 10)

    # Model 2: Feature extraction
    model_feature = models.resnet18(pretrained=True)
    for param in model_feature.parameters():
        param.requires_grad = False
    model_feature.fc = nn.Linear(512, 10)

    # Model 3: Fine-tuning
    model_finetune = models.resnet18(pretrained=True)
    model_finetune.fc = nn.Linear(512, 10)

    print("\nThree models created:")
    print("1. From scratch: No pretrained weights")
    print("2. Feature extraction: Frozen backbone")
    print("3. Fine-tuning: All layers trainable")

    print("\nTODO: Train all three and compare:")
    print("- Accuracy after 5 epochs")
    print("- Time per epoch")
    print("- Convergence speed")


# ============================================================================
# EXERCISE 7: Model Ensemble
# ============================================================================

def exercise7_model_ensemble():
    """
    Exercise 7: Create an ensemble of CNNs

    TODO: Train multiple diverse models and combine predictions:
    1. Train 3-5 different CNN architectures
    2. Combine predictions using:
       - Majority voting
       - Average probabilities
       - Weighted average

    Compare ensemble vs single model performance
    """
    print("\n" + "=" * 60)
    print("Exercise 7: Model Ensemble")
    print("=" * 60)

    # Create diverse models
    models_list = [
        models.resnet18(pretrained=True),
        models.resnet34(pretrained=True),
        models.mobilenet_v2(pretrained=True),
    ]

    print(f"\nCreated ensemble of {len(models_list)} models")

    def ensemble_predict(models_list, x):
        """Combine predictions from multiple models"""
        predictions = []
        for model in models_list:
            model.eval()
            with torch.no_grad():
                output = model(x)
                pred = F.softmax(output, dim=1)
                predictions.append(pred)

        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred

    print("\nTODO: Implement and compare:")
    print("1. Majority voting")
    print("2. Average probabilities")
    print("3. Weighted average (weight by validation accuracy)")


# ============================================================================
# EXERCISE 8: Grad-CAM Visualization
# ============================================================================

def exercise8_gradcam():
    """
    Exercise 8: Implement Grad-CAM for visual explanations

    TODO: Implement Grad-CAM (Gradient-weighted Class Activation Mapping):
    1. Forward pass to get predictions
    2. Backward pass to get gradients
    3. Weight feature maps by gradients
    4. Create heatmap showing important regions

    Visualize which parts of the image the model focuses on
    """
    print("\n" + "=" * 60)
    print("Exercise 8: Grad-CAM Visualization")
    print("=" * 60)

    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None

            # Register hooks
            target_layer.register_forward_hook(self.save_activation)
            target_layer.register_backward_hook(self.save_gradient)

        def save_activation(self, module, input, output):
            self.activations = output.detach()

        def save_gradient(self, module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def generate_cam(self, input_image, target_class):
            # TODO: Implement Grad-CAM algorithm
            # 1. Forward pass
            output = self.model(input_image)

            # 2. Backward pass for target class
            self.model.zero_grad()
            target = output[0, target_class]
            target.backward()

            # 3. Pool gradients
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

            # 4. Weight activations
            for i in range(self.activations.shape[1]):
                self.activations[:, i, :, :] *= pooled_gradients[i]

            # 5. Create heatmap
            heatmap = torch.mean(self.activations, dim=1).squeeze()
            heatmap = F.relu(heatmap)
            heatmap /= torch.max(heatmap)

            return heatmap

    print("\nGrad-CAM class defined")
    print("TODO: Use this to visualize important regions for predictions")
    print("Hint: Overlay heatmap on original image for interpretation")


def run_all_exercises():
    """Run all exercises"""
    print("\n" + "=" * 60)
    print("CNN IMAGE PROCESSING - EXERCISES")
    print("=" * 60)

    exercise1_custom_cnn()
    exercise2_augmentation_effects()
    exercise3_batchnorm_impact()
    exercise4_receptive_field()
    exercise5_feature_visualization()
    exercise6_transfer_comparison()
    exercise7_model_ensemble()
    exercise8_gradcam()

    print("\n" + "=" * 60)
    print("Exercise Summary")
    print("=" * 60)
    print("1. Custom CNN: Design and train your architecture")
    print("2. Augmentation: Compare augmentation strategies")
    print("3. BatchNorm: Measure BatchNorm impact")
    print("4. Receptive Field: Calculate field sizes")
    print("5. Feature Maps: Visualize learned features")
    print("6. Transfer Learning: Compare strategies")
    print("7. Ensemble: Combine multiple models")
    print("8. Grad-CAM: Explain model predictions")
    print("=" * 60)


if __name__ == "__main__":
    run_all_exercises()
