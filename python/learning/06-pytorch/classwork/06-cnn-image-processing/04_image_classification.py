"""
Image Classification - Complete CNN Training Pipeline

This module demonstrates:
- Complete image classification pipeline
- Training on MNIST and CIFAR-10
- Data augmentation techniques
- Model evaluation and metrics
- Confusion matrix and per-class accuracy
- Saving and loading models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import time


class MNISTCNN(nn.Module):
    """CNN for MNIST digit classification"""

    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CIFAR10CNN(nn.Module):
    """CNN for CIFAR-10 image classification"""

    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


def prepare_mnist_data(batch_size=64, augment=False):
    """Prepare MNIST dataset with optional augmentation"""
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def prepare_cifar10_data(batch_size=64, augment=True):
    """Prepare CIFAR-10 dataset with augmentation"""
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')

    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, device, test_loader, criterion):
    """Evaluate model"""
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy, all_preds, all_targets


def train_classifier(model, train_loader, test_loader, device, epochs=10, lr=0.001):
    """Complete training pipeline"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc, _, _ = evaluate(model, device, test_loader, criterion)

        scheduler.step(test_loss)

        epoch_time = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")

    return model, history


def plot_confusion_matrix(true_labels, pred_labels, class_names, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, pred_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    return fig


def main():
    """Main demonstration function"""
    print("\n" + "=" * 60)
    print("IMAGE CLASSIFICATION WITH CNNS")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Train on MNIST
    print("\n" + "=" * 60)
    print("Training on MNIST")
    print("=" * 60)

    mnist_train_loader, mnist_test_loader = prepare_mnist_data(batch_size=128, augment=True)
    mnist_model = MNISTCNN().to(device)

    print(f"Model parameters: {sum(p.numel() for p in mnist_model.parameters()):,}")

    mnist_model, mnist_history = train_classifier(
        mnist_model, mnist_train_loader, mnist_test_loader, device, epochs=5, lr=0.001
    )

    # Evaluate MNIST
    _, test_acc, preds, targets = evaluate(mnist_model, device, mnist_test_loader, nn.CrossEntropyLoss())

    print(f"\n Final MNIST Test Accuracy: {test_acc:.2f}%")

    # Confusion matrix for MNIST
    mnist_classes = [str(i) for i in range(10)]
    fig = plot_confusion_matrix(targets, preds, mnist_classes, 'MNIST Confusion Matrix')
    plt.savefig('/tmp/mnist_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved to /tmp/mnist_confusion_matrix.png")
    plt.close()

    # Classification report
    print("\nMNIST Classification Report:")
    print(classification_report(targets, preds, target_names=mnist_classes))

    # Save model
    torch.save(mnist_model.state_dict(), '/tmp/mnist_cnn.pth')
    print("\nModel saved to /tmp/mnist_cnn.pth")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Data augmentation improves generalization")
    print("2. Learning rate scheduling helps convergence")
    print("3. Confusion matrix reveals per-class performance")
    print("4. BatchNorm and Dropout prevent overfitting")
    print("5. Deeper networks handle complex datasets (CIFAR-10)")
    print("=" * 60)


if __name__ == "__main__":
    main()
