"""
Project: Sentiment Analysis
===========================
Classify text sentiment using RNN/LSTM.

Dataset: Synthetic movie reviews (positive/negative)
Goals:
- Generate and preprocess text data
- Build vocabulary and embeddings
- Implement LSTM/GRU sentiment classifier
- Train with proper text processing
- Evaluate on test set with detailed metrics
- Make predictions on new text
- Visualize attention weights

Skills: RNNs, LSTMs, Text processing, Classification, Embeddings
Run: python project_sentiment_analysis.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


# Sample vocabulary and sentiment-related words
POSITIVE_WORDS = [
    'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'love',
    'brilliant', 'perfect', 'outstanding', 'superb', 'beautiful', 'best',
    'awesome', 'impressive', 'enjoyable', 'delightful', 'terrific'
]

NEGATIVE_WORDS = [
    'terrible', 'awful', 'horrible', 'bad', 'worst', 'boring', 'disappointing',
    'poor', 'waste', 'dull', 'tedious', 'mediocre', 'pathetic', 'useless',
    'frustrating', 'annoying', 'unpleasant'
]

NEUTRAL_WORDS = [
    'movie', 'film', 'story', 'plot', 'character', 'scene', 'actor', 'director',
    'watch', 'show', 'time', 'part', 'end', 'start', 'see', 'think', 'make',
    'really', 'very', 'much', 'just', 'like', 'get', 'one', 'would', 'even'
]


class TextDataset(Dataset):
    """Generate synthetic movie reviews with sentiment labels."""

    def __init__(self, num_samples=5000, max_len=50, vocab=None):
        self.num_samples = num_samples
        self.max_len = max_len

        if vocab is None:
            self.build_vocab()
        else:
            self.vocab = vocab
            self.word2idx = {word: idx for idx, word in enumerate(vocab)}

        self.samples = self.generate_samples()

    def build_vocab(self):
        """Build vocabulary."""
        self.vocab = ['<PAD>', '<UNK>'] + POSITIVE_WORDS + NEGATIVE_WORDS + NEUTRAL_WORDS
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}

    def generate_samples(self):
        """Generate synthetic reviews."""
        samples = []

        for _ in range(self.num_samples):
            # Random sentiment
            sentiment = np.random.randint(0, 2)  # 0: negative, 1: positive

            # Generate review
            if sentiment == 1:
                # Positive review
                sentiment_words = np.random.choice(POSITIVE_WORDS, size=np.random.randint(3, 6))
                neutral_words = np.random.choice(NEUTRAL_WORDS, size=np.random.randint(5, 10))
            else:
                # Negative review
                sentiment_words = np.random.choice(NEGATIVE_WORDS, size=np.random.randint(3, 6))
                neutral_words = np.random.choice(NEUTRAL_WORDS, size=np.random.randint(5, 10))

            # Combine and shuffle
            words = list(sentiment_words) + list(neutral_words)
            np.random.shuffle(words)

            # Truncate to max_len
            words = words[:self.max_len]

            # Convert to indices
            indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]

            samples.append((indices, sentiment))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        indices, label = self.samples[idx]
        return torch.LongTensor(indices), label


def collate_fn(batch):
    """Custom collate function to pad sequences."""
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return texts_padded, labels


class SentimentLSTM(nn.Module):
    """LSTM-based sentiment classifier."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.5, bidirectional=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Fully connected layers
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary classification
        )

    def forward(self, text):
        # text: [batch_size, seq_len]

        # Embedding
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: [batch_size, seq_len, hidden_dim * num_directions]

        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # Fully connected
        output = self.fc(hidden)  # [batch_size, 2]

        return output


class SentimentGRU(nn.Module):
    """GRU-based sentiment classifier for comparison."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, text):
        embedded = self.embedding(text)
        gru_out, hidden = self.gru(embedded)

        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        output = self.fc(hidden)
        return output


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (texts, labels) in enumerate(train_loader):
        texts, labels = texts.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 50 == 49:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {running_loss/(batch_idx+1):.4f} "
                  f"Acc: {100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    # Per-class metrics
    true_positives = [0, 0]
    false_positives = [0, 0]
    false_negatives = [0, 0]

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class metrics
            for i in range(len(labels)):
                pred = predicted[i].item()
                label = labels[i].item()

                if pred == label:
                    true_positives[label] += 1
                else:
                    false_positives[pred] += 1
                    false_negatives[label] += 1

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total

    # Calculate precision, recall, F1
    metrics = {}
    for i in range(2):
        precision = true_positives[i] / (true_positives[i] + false_positives[i] + 1e-10)
        recall = true_positives[i] / (true_positives[i] + false_negatives[i] + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        class_name = 'Negative' if i == 0 else 'Positive'
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return test_loss, test_acc, metrics, all_predictions, all_labels


def train_model(model, train_loader, test_loader, num_epochs, device, lr=0.001):
    """Complete training loop."""
    print("\n" + "=" * 60)
    print("TRAINING SENTIMENT CLASSIFIER")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    best_acc = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()

        print(f"\nEpoch [{epoch+1}/{num_epochs}] LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Evaluate
        test_loss, test_acc, metrics, _, _ = evaluate_model(
            model, test_loader, criterion, device
        )

        scheduler.step()

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_sentiment.pth')
            print(f"  Saved best model (Acc: {best_acc:.2f}%)")

        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        epoch_time = time.time() - start_time

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

    return history, best_acc


def visualize_results(history, metrics, predictions, labels):
    """Create visualizations."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    fig = plt.figure(figsize=(14, 5))

    # 1. Training curves - Loss
    ax1 = plt.subplot(1, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True)

    # 2. Training curves - Accuracy
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy', fontweight='bold')
    ax2.legend()
    ax2.grid(True)

    # 3. Confusion matrix
    ax3 = plt.subplot(1, 3, 3)
    confusion = np.zeros((2, 2), dtype=int)
    for pred, label in zip(predictions, labels):
        confusion[label][pred] += 1

    im = ax3.imshow(confusion, cmap='Blues', aspect='auto')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Negative', 'Positive'])
    ax3.set_yticklabels(['Negative', 'Positive'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Confusion Matrix', fontweight='bold')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax3.text(j, i, confusion[i, j],
                           ha="center", va="center", color="black", fontsize=14)

    plt.colorbar(im, ax=ax3)

    plt.tight_layout()
    print("Visualizations created!")
    print("Close the plot window to continue...")
    plt.show()


def predict_sentiment(model, text, vocab, word2idx, device, max_len=50):
    """Predict sentiment for a single text."""
    model.eval()

    # Tokenize (simple word split)
    words = text.lower().split()[:max_len]

    # Convert to indices
    indices = [word2idx.get(word, word2idx['<UNK>']) for word in words]

    # Convert to tensor
    text_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(text_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = output.max(1)

    sentiment = 'Positive' if predicted.item() == 1 else 'Negative'
    confidence = probabilities[0][predicted.item()].item()

    return sentiment, confidence, probabilities[0].cpu().numpy()


def main():
    """Main execution function."""
    print("=" * 60)
    print("SENTIMENT ANALYSIS PROJECT")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    num_epochs = 25
    learning_rate = 0.001
    num_train = 5000
    num_test = 1000

    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")

    # Create datasets
    print("\nGenerating text datasets...")
    train_dataset = TextDataset(num_samples=num_train, max_len=50)
    test_dataset = TextDataset(num_samples=num_test, max_len=50, vocab=train_dataset.vocab)

    vocab = train_dataset.vocab
    word2idx = train_dataset.word2idx
    vocab_size = len(vocab)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Training samples: {num_train}")
    print(f"Test samples: {num_test}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Create model
    print("\nInitializing LSTM model...")
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5,
        bidirectional=True
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train model
    history, best_acc = train_model(
        model, train_loader, test_loader, num_epochs, device, learning_rate
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, metrics, predictions, labels = evaluate_model(
        model, test_loader, criterion, device
    )

    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print(f"Best Test Accuracy: {best_acc:.2f}%")

    print("\nPer-Class Metrics:")
    for class_name, class_metrics in metrics.items():
        print(f"\n{class_name}:")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  Recall:    {class_metrics['recall']:.4f}")
        print(f"  F1 Score:  {class_metrics['f1']:.4f}")

    # Visualizations
    visualize_results(history, metrics, predictions, labels)

    # Test predictions on sample texts
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    sample_texts = [
        "excellent movie really love amazing story wonderful",
        "terrible film boring waste awful disappointing bad",
        "great plot fantastic actor brilliant perfect outstanding",
        "horrible story worst movie poor terrible disappointing"
    ]

    for text in sample_texts:
        sentiment, confidence, probs = predict_sentiment(
            model, text, vocab, word2idx, device
        )
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
        print(f"Probabilities: Negative={probs[0]:.4f}, Positive={probs[1]:.4f}")

    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)
    print("\nKey Findings:")
    print(f"1. Achieved {best_acc:.2f}% accuracy on sentiment classification")
    print("2. Bidirectional LSTM captures context from both directions")
    print("3. Gradient clipping prevents exploding gradients")
    print("4. Model effectively learns sentiment-bearing words")
    print("5. Model saved to 'best_sentiment.pth'")
    print("=" * 60)


if __name__ == "__main__":
    main()
