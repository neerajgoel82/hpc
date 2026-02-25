"""
Transfer Learning Exercises
Practice problems for transfer learning with PyTorch

Complete these exercises to master transfer learning:
1. Load and modify pretrained models
2. Feature extraction setup
3. Fine-tuning implementation
4. Model comparison
5. Custom classifier design
6. Domain adaptation
7. Complete pipeline
8. Model selection

Run: python exercises.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models


def exercise_1_load_model():
    """
    Exercise 1: Load and Inspect Pretrained Model

    Task: Load a pretrained ResNet-34 model and print:
    - Total number of parameters
    - Number of layers
    - Input size of the final fully connected layer
    - Output size of the final fully connected layer
    """
    print("\n" + "=" * 60)
    print("Exercise 1: Load and Inspect Model")
    print("=" * 60)

    # TODO: Load pretrained ResNet-34
    # model = ...

    # TODO: Count total parameters
    # total_params = ...

    # TODO: Print model information
    # print(f"Total parameters: {total_params:,}")
    # print(f"FC input size: ...")
    # print(f"FC output size: ...")

    print("TODO: Implement this exercise")
    print()


def exercise_2_feature_extraction():
    """
    Exercise 2: Setup for Feature Extraction

    Task: Prepare a ResNet-18 for feature extraction:
    - Load pretrained ResNet-18
    - Freeze all layers
    - Replace final layer for 7 classes
    - Verify only the final layer is trainable
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Feature Extraction Setup")
    print("=" * 60)

    # TODO: Load pretrained ResNet-18
    # model = ...

    # TODO: Freeze all parameters
    # for param in model.parameters():
    #     ...

    # TODO: Replace final layer for 7 classes
    # model.fc = ...

    # TODO: Count trainable parameters
    # trainable = ...

    # TODO: Verify
    # print(f"Trainable parameters: {trainable:,}")

    print("TODO: Implement this exercise")
    print()


def exercise_3_fine_tuning():
    """
    Exercise 3: Fine-Tuning Strategy

    Task: Implement selective fine-tuning:
    - Load pretrained ResNet-50
    - Freeze layers 1-3
    - Unfreeze layer 4
    - Replace classifier for 15 classes
    - Use discriminative learning rates:
      - Layer 4: lr=1e-4
      - New classifier: lr=1e-3
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Fine-Tuning Strategy")
    print("=" * 60)

    # TODO: Load model
    # model = ...

    # TODO: Freeze layers 1-3
    # for name, param in model.named_parameters():
    #     if 'layer1' in name or 'layer2' in name or 'layer3' in name:
    #         ...

    # TODO: Replace classifier
    # model.fc = ...

    # TODO: Setup optimizer with discriminative learning rates
    # optimizer = optim.Adam([
    #     {'params': ..., 'lr': 1e-4},
    #     {'params': ..., 'lr': 1e-3}
    # ])

    print("TODO: Implement this exercise")
    print()


def exercise_4_model_comparison():
    """
    Exercise 4: Compare Model Architectures

    Task: Compare these models and print their specs:
    - ResNet-18
    - MobileNet V2
    - VGG-11

    For each model, print:
    - Number of parameters
    - Model size in MB (assume float32)
    - Recommended use case
    """
    print("\n" + "=" * 60)
    print("Exercise 4: Model Comparison")
    print("=" * 60)

    # TODO: Load models and compare
    # models_dict = {
    #     'ResNet-18': models.resnet18(pretrained=False),
    #     ...
    # }

    # TODO: For each model, calculate and print:
    # - Parameters
    # - Size in MB
    # - Use case recommendation

    print("TODO: Implement this exercise")
    print()


def exercise_5_custom_classifier():
    """
    Exercise 5: Custom Classifier Design

    Task: Design a custom classifier for ResNet-50:
    - 3 fully connected layers: 2048 -> 1024 -> 512 -> 20
    - ReLU activation after each hidden layer
    - Dropout (0.5) after first layer
    - Dropout (0.3) after second layer
    - Replace the ResNet-50 fc layer with this classifier
    """
    print("\n" + "=" * 60)
    print("Exercise 5: Custom Classifier Design")
    print("=" * 60)

    # TODO: Load ResNet-50
    # model = ...

    # TODO: Design custom classifier
    # model.fc = nn.Sequential(
    #     ...
    # )

    # TODO: Print classifier architecture
    # print(model.fc)

    print("TODO: Implement this exercise")
    print()


def exercise_6_domain_adaptation():
    """
    Exercise 6: Domain Adaptation

    Task: Adapt ResNet-18 for grayscale medical images:
    - Load pretrained ResNet-18
    - Modify conv1 to accept 1 input channel (grayscale)
    - Initialize new conv1 weights as mean of RGB weights
    - Replace classifier for 3 classes (Normal, Benign, Malignant)
    - Freeze layers 1-2, unfreeze layers 3-4
    """
    print("\n" + "=" * 60)
    print("Exercise 6: Domain Adaptation")
    print("=" * 60)

    # TODO: Load model
    # model = ...

    # TODO: Modify conv1 for grayscale
    # original_conv = model.conv1
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # TODO: Initialize with mean of RGB weights
    # with torch.no_grad():
    #     model.conv1.weight = ...

    # TODO: Replace classifier for 3 classes
    # model.fc = ...

    # TODO: Selective freezing
    # ...

    print("TODO: Implement this exercise")
    print()


def exercise_7_training_pipeline():
    """
    Exercise 7: Complete Training Pipeline

    Task: Implement a complete training function:
    - Accept model, train_loader, val_loader, device, epochs
    - Use CrossEntropyLoss
    - Use Adam optimizer (lr=0.001)
    - Include validation after each epoch
    - Print train/val loss and accuracy
    - Return trained model and history
    """
    print("\n" + "=" * 60)
    print("Exercise 7: Training Pipeline")
    print("=" * 60)

    def train_model(model, train_loader, val_loader, device, epochs=5):
        """Training function to implement."""
        # TODO: Implement complete training loop
        # - Setup loss and optimizer
        # - Training loop with validation
        # - Track history
        # - Return model and history
        pass

    # TODO: Test with synthetic data
    # Create small dataset
    # images = torch.randn(100, 3, 224, 224)
    # labels = torch.randint(0, 5, (100,))
    # dataset = TensorDataset(images, labels)
    # loader = DataLoader(dataset, batch_size=16)

    # Load model
    # model = models.resnet18(pretrained=True)
    # model.fc = nn.Linear(512, 5)

    # Train
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # trained_model, history = train_model(model, loader, loader, device, epochs=2)

    print("TODO: Implement this exercise")
    print()


def exercise_8_model_selection():
    """
    Exercise 8: Model Selection Challenge

    Task: Given these scenarios, choose the best model and justify:

    Scenario A: Mobile app for plant identification
    - 10 plant species
    - Must run on smartphones
    - Real-time inference required

    Scenario B: Medical imaging diagnosis
    - 5 disease classes
    - Grayscale X-ray images
    - Accuracy is most important
    - GPU available for training and inference

    Scenario C: Satellite land cover classification
    - 15 land cover types
    - 4-channel images (RGB + IR)
    - Dataset: 10,000 images
    - Must balance accuracy and speed

    For each scenario, decide:
    - Which pretrained model to use?
    - Feature extraction or fine-tuning?
    - What modifications needed?
    """
    print("\n" + "=" * 60)
    print("Exercise 8: Model Selection Challenge")
    print("=" * 60)

    scenarios = {
        'A': {
            'task': 'Mobile plant identification',
            'constraints': 'Real-time on smartphone',
            'data': '10 classes, RGB images',
        },
        'B': {
            'task': 'Medical X-ray diagnosis',
            'constraints': 'Best accuracy, GPU available',
            'data': '5 classes, grayscale images',
        },
        'C': {
            'task': 'Satellite land cover',
            'constraints': 'Balance accuracy/speed',
            'data': '15 classes, 4-channel images, 10k samples',
        }
    }

    print("Analyze each scenario and choose:")
    print("  1. Best model architecture")
    print("  2. Training strategy")
    print("  3. Required modifications")
    print()

    for scenario, info in scenarios.items():
        print(f"Scenario {scenario}: {info['task']}")
        print(f"  Constraints: {info['constraints']}")
        print(f"  Data: {info['data']}")
        print(f"  TODO: Your recommendation:")
        print(f"    Model: ???")
        print(f"    Strategy: ???")
        print(f"    Modifications: ???")
        print()


def bonus_exercise_progressive_unfreezing():
    """
    Bonus Exercise: Progressive Unfreezing

    Task: Implement progressive unfreezing:
    - Phase 1 (3 epochs): Train only classifier
    - Phase 2 (3 epochs): Unfreeze layer4, train with lr=1e-4
    - Phase 3 (4 epochs): Unfreeze all, train with lr=1e-5

    Use synthetic data for demonstration.
    """
    print("\n" + "=" * 60)
    print("Bonus: Progressive Unfreezing")
    print("=" * 60)

    # TODO: Implement three-phase training
    # model = models.resnet50(pretrained=True)
    # model.fc = nn.Linear(2048, 10)

    # Phase 1: Train classifier only
    # ...

    # Phase 2: Unfreeze layer4
    # ...

    # Phase 3: Unfreeze all
    # ...

    print("TODO: Implement this bonus exercise")
    print()


def run_all_exercises():
    """Run all exercises."""
    print("=" * 60)
    print("TRANSFER LEARNING EXERCISES")
    print("=" * 60)
    print()
    print("Complete these exercises to master transfer learning.")
    print("Uncomment the TODO sections and implement the solutions.")
    print()

    exercise_1_load_model()
    exercise_2_feature_extraction()
    exercise_3_fine_tuning()
    exercise_4_model_comparison()
    exercise_5_custom_classifier()
    exercise_6_domain_adaptation()
    exercise_7_training_pipeline()
    exercise_8_model_selection()
    bonus_exercise_progressive_unfreezing()

    print("=" * 60)
    print("EXERCISE SOLUTIONS CHECKLIST")
    print("=" * 60)
    print()
    print("[ ] Exercise 1: Load and inspect model")
    print("[ ] Exercise 2: Feature extraction setup")
    print("[ ] Exercise 3: Fine-tuning strategy")
    print("[ ] Exercise 4: Model comparison")
    print("[ ] Exercise 5: Custom classifier design")
    print("[ ] Exercise 6: Domain adaptation")
    print("[ ] Exercise 7: Training pipeline")
    print("[ ] Exercise 8: Model selection challenge")
    print("[ ] Bonus: Progressive unfreezing")
    print()
    print("=" * 60)


def main():
    """Main function."""
    run_all_exercises()


if __name__ == "__main__":
    main()
