"""
Project: Simple Object Detection
================================
Build a simple object detection system with bounding boxes.

Dataset: Synthetic dataset with geometric shapes
Goals:
- Generate synthetic images with bounding boxes
- Build detection model (classification + localization)
- Train with combined loss (classification + bbox regression)
- Evaluate detection accuracy and IoU
- Visualize predictions with bounding boxes
- Understand multi-task learning

Skills: Object detection, Bounding boxes, Multi-task loss, IoU
Run: python project_object_detection.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Shape classes
CLASSES = ['circle', 'square', 'triangle']
NUM_CLASSES = len(CLASSES)


class ShapeDataset(Dataset):
    """Generate synthetic images with shapes and bounding boxes."""

    def __init__(self, num_samples=5000, img_size=64, transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image with shape
        img, label, bbox = self.generate_shape()

        if self.transform:
            img = self.transform(img)

        return img, label, bbox

    def generate_shape(self):
        """Generate image with a random shape and return bbox."""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

        # Random shape type
        shape_type = np.random.randint(0, NUM_CLASSES)

        # Random size (20-40% of image)
        size = np.random.randint(
            int(self.img_size * 0.2),
            int(self.img_size * 0.4)
        )

        # Random position (ensure shape fits in image)
        margin = size
        center_x = np.random.randint(margin, self.img_size - margin)
        center_y = np.random.randint(margin, self.img_size - margin)

        # Random color
        color = np.random.rand(3)

        if shape_type == 0:  # Circle
            bbox = self.draw_circle(img, center_x, center_y, size // 2, color)
        elif shape_type == 1:  # Square
            bbox = self.draw_square(img, center_x, center_y, size, color)
        else:  # Triangle
            bbox = self.draw_triangle(img, center_x, center_y, size, color)

        # Add noise
        img += np.random.randn(*img.shape) * 0.05
        img = np.clip(img, 0, 1)

        # Convert to CHW format
        img = np.transpose(img, (2, 0, 1))

        return torch.FloatTensor(img), shape_type, torch.FloatTensor(bbox)

    def draw_circle(self, img, cx, cy, radius, color):
        """Draw circle and return bounding box."""
        y, x = np.ogrid[:self.img_size, :self.img_size]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        img[mask] = color

        # Bounding box: [x_min, y_min, x_max, y_max] normalized to [0, 1]
        x_min, y_min = cx - radius, cy - radius
        x_max, y_max = cx + radius, cy + radius

        return [x_min / self.img_size, y_min / self.img_size,
                x_max / self.img_size, y_max / self.img_size]

    def draw_square(self, img, cx, cy, size, color):
        """Draw square and return bounding box."""
        half = size // 2
        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half

        y1, y2 = max(0, y1), min(self.img_size, y2)
        x1, x2 = max(0, x1), min(self.img_size, x2)

        img[y1:y2, x1:x2] = color

        return [x1 / self.img_size, y1 / self.img_size,
                x2 / self.img_size, y2 / self.img_size]

    def draw_triangle(self, img, cx, cy, size, color):
        """Draw triangle and return bounding box."""
        half = size // 2

        # Triangle vertices
        vertices = [
            (cx, cy - half),  # Top
            (cx - half, cy + half),  # Bottom left
            (cx + half, cy + half)  # Bottom right
        ]

        # Draw filled triangle (simplified rasterization)
        for y in range(max(0, cy - half), min(self.img_size, cy + half + 1)):
            for x in range(max(0, cx - half), min(self.img_size, cx + half + 1)):
                if self.point_in_triangle((x, y), vertices):
                    img[y, x] = color

        # Bounding box
        x_min, y_min = cx - half, cy - half
        x_max, y_max = cx + half, cy + half

        return [x_min / self.img_size, y_min / self.img_size,
                x_max / self.img_size, y_max / self.img_size]

    def point_in_triangle(self, pt, vertices):
        """Check if point is inside triangle."""
        x, y = pt
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]

        denominator = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        if abs(denominator) < 1e-10:
            return False

        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
        c = 1 - a - b

        return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1


class ObjectDetector(nn.Module):
    """CNN for object detection (classification + localization)."""

    def __init__(self, num_classes=3):
        super().__init__()

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8 -> 4
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 4)  # [x_min, y_min, x_max, y_max]
        )

    def forward(self, x):
        features = self.features(x)
        class_logits = self.classifier(features)
        bbox_pred = self.bbox_regressor(features)
        return class_logits, bbox_pred


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    # box format: [x_min, y_min, x_max, y_max]

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    criterion_class = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()

    running_loss = 0.0
    running_class_loss = 0.0
    running_bbox_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels, bboxes) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)

        # Forward pass
        optimizer.zero_grad()
        class_logits, bbox_pred = model(images)

        # Combined loss
        loss_class = criterion_class(class_logits, labels)
        loss_bbox = criterion_bbox(bbox_pred, bboxes)
        loss = loss_class + loss_bbox

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        running_class_loss += loss_class.item()
        running_bbox_loss += loss_bbox.item()

        _, predicted = class_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 50 == 49:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {running_loss/(batch_idx+1):.4f} "
                  f"Class: {running_class_loss/(batch_idx+1):.4f} "
                  f"BBox: {running_bbox_loss/(batch_idx+1):.4f} "
                  f"Acc: {100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()

    criterion_class = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()

    test_loss = 0.0
    correct = 0
    total = 0
    total_iou = 0.0

    with torch.no_grad():
        for images, labels, bboxes in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            # Forward pass
            class_logits, bbox_pred = model(images)

            # Loss
            loss_class = criterion_class(class_logits, labels)
            loss_bbox = criterion_bbox(bbox_pred, bboxes)
            loss = loss_class + loss_bbox
            test_loss += loss.item()

            # Classification accuracy
            _, predicted = class_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # IoU calculation
            for i in range(len(bboxes)):
                iou = calculate_iou(
                    bbox_pred[i].cpu().numpy(),
                    bboxes[i].cpu().numpy()
                )
                total_iou += iou

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    avg_iou = total_iou / total

    return test_loss, test_acc, avg_iou


def train_model(model, train_loader, test_loader, num_epochs, device, lr=0.001):
    """Complete training loop."""
    print("\n" + "=" * 60)
    print("TRAINING OBJECT DETECTOR")
    print("=" * 60)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'test_iou': []
    }

    best_iou = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()

        print(f"\nEpoch [{epoch+1}/{num_epochs}] LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # Evaluate
        test_loss, test_acc, test_iou = evaluate_model(
            model, test_loader, device
        )

        scheduler.step()

        # Save best model
        if test_iou > best_iou:
            best_iou = test_iou
            torch.save(model.state_dict(), 'best_detector.pth')
            print(f"  Saved best model (IoU: {best_iou:.4f})")

        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_iou'].append(test_iou)

        epoch_time = time.time() - start_time

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Test IoU:   {test_iou:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

    return history, best_iou


def visualize_training(history):
    """Visualize training metrics."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    axes[1].plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Classification Accuracy', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True)

    # IoU
    axes[2].plot(epochs, history['test_iou'], 'g-', label='Test IoU')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].set_title('Bounding Box IoU', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    print("Training visualizations created!")
    print("Close the plot window to continue...")
    plt.show()


def visualize_predictions(model, test_loader, device, num_samples=16):
    """Visualize predictions with bounding boxes."""
    model.eval()

    # Get a batch
    dataiter = iter(test_loader)
    images, labels, true_bboxes = next(dataiter)

    # Predict
    with torch.no_grad():
        images_gpu = images.to(device)
        class_logits, pred_bboxes = model(images_gpu)
        _, predicted_labels = class_logits.max(1)

    # Move to CPU
    images = images.cpu()
    labels = labels.cpu()
    true_bboxes = true_bboxes.cpu()
    predicted_labels = predicted_labels.cpu()
    pred_bboxes = pred_bboxes.cpu()

    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.flatten()

    for idx in range(min(num_samples, len(images))):
        ax = axes[idx]

        # Get image
        img = images[idx].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        ax.imshow(img)

        img_size = img.shape[0]

        # Ground truth bbox (green)
        true_box = true_bboxes[idx].numpy()
        true_rect = patches.Rectangle(
            (true_box[0] * img_size, true_box[1] * img_size),
            (true_box[2] - true_box[0]) * img_size,
            (true_box[3] - true_box[1]) * img_size,
            linewidth=2, edgecolor='green', facecolor='none', label='True'
        )
        ax.add_patch(true_rect)

        # Predicted bbox (red)
        pred_box = pred_bboxes[idx].numpy()
        pred_rect = patches.Rectangle(
            (pred_box[0] * img_size, pred_box[1] * img_size),
            (pred_box[2] - pred_box[0]) * img_size,
            (pred_box[3] - pred_box[1]) * img_size,
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Pred'
        )
        ax.add_patch(pred_rect)

        # Calculate IoU
        iou = calculate_iou(pred_box, true_box)

        # Title
        true_class = CLASSES[labels[idx]]
        pred_class = CLASSES[predicted_labels[idx]]
        color = 'green' if labels[idx] == predicted_labels[idx] else 'red'

        ax.set_title(f'T: {true_class}, P: {pred_class}\nIoU: {iou:.3f}',
                     color=color, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    print("\nSample predictions with bounding boxes displayed")
    print("Green box = Ground truth, Red dashed = Prediction")
    print("Close the plot window to continue...")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("OBJECT DETECTION PROJECT")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    num_epochs = 30
    learning_rate = 0.001
    num_train = 5000
    num_test = 1000

    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Training samples: {num_train}")
    print(f"  Test samples: {num_test}")

    # Create datasets
    print("\nGenerating synthetic datasets...")
    train_dataset = ShapeDataset(num_samples=num_train, img_size=64)
    test_dataset = ShapeDataset(num_samples=num_test, img_size=64)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nInitializing model...")
    model = ObjectDetector(num_classes=NUM_CLASSES)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train model
    history, best_iou = train_model(
        model, train_loader, test_loader, num_epochs, device, learning_rate
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    test_loss, test_acc, test_iou = evaluate_model(model, test_loader, device)

    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print(f"Final Test IoU: {test_iou:.4f}")
    print(f"Best Test IoU: {best_iou:.4f}")

    # Visualizations
    visualize_training(history)
    visualize_predictions(model, test_loader, device)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)
    print("\nKey Findings:")
    print(f"1. Achieved {test_acc:.2f}% classification accuracy")
    print(f"2. Average IoU of {test_iou:.4f} for bounding box predictions")
    print("3. Multi-task learning successfully combines classification and localization")
    print("4. SmoothL1Loss works well for bbox regression")
    print("5. Model saved to 'best_detector.pth'")
    print("=" * 60)


if __name__ == "__main__":
    main()
