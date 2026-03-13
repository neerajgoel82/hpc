"""
Sequence Classification - Text and Sequence Classification Tasks

This module demonstrates:
1. Sentiment analysis with LSTM
2. Text classification architectures
3. Handling variable-length sequences
4. Padding and packing sequences
5. Multi-class sequence classification

Run: python 04_sequence_classification.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np


class SentimentLSTM(nn.Module):
    """LSTM for sentiment classification."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use last hidden state for classification
        return self.fc(self.dropout(hidden[-1]))


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with max pooling."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Max pooling over sequence
        pooled = torch.max(lstm_out, dim=1)[0]
        return self.fc(self.dropout(pooled))


class GRUTextClassifier(nn.Module):
    """Multi-layer GRU with attention-like mechanism."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=2,
                          batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        # Use final layer's last hidden state
        return self.fc(hidden[-1])


def create_synthetic_text_data(num_samples: int, vocab_size: int, max_len: int):
    """Create synthetic text classification dataset."""
    sequences = []
    labels = []

    for _ in range(num_samples):
        # Create sequences with patterns for different classes
        seq_len = np.random.randint(5, max_len + 1)
        label = np.random.randint(0, 3)

        if label == 0:
            # Class 0: sequences starting with small numbers
            seq = [np.random.randint(1, vocab_size // 3) for _ in range(seq_len)]
        elif label == 1:
            # Class 1: sequences with medium numbers
            seq = [np.random.randint(vocab_size // 3, 2 * vocab_size // 3)
                   for _ in range(seq_len)]
        else:
            # Class 2: sequences with large numbers
            seq = [np.random.randint(2 * vocab_size // 3, vocab_size)
                   for _ in range(seq_len)]

        sequences.append(torch.tensor(seq))
        labels.append(label)

    return sequences, torch.tensor(labels)


def train_sentiment_classifier():
    """Train sentiment classification model."""
    print("=" * 70)
    print("Sentiment Classification with LSTM")
    print("=" * 70)

    vocab_size = 1000
    embedding_dim = 64
    hidden_size = 128
    num_classes = 3
    max_len = 20
    num_samples = 500

    # Create dataset
    sequences, labels = create_synthetic_text_data(num_samples, vocab_size, max_len)

    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    # Split train/test
    split = int(0.8 * num_samples)
    X_train, X_test = padded_sequences[:split], padded_sequences[split:]
    y_train, y_test = labels[:split], labels[split:]

    print(f"\nDataset:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Max sequence length: {max_len}")

    # Create model
    model = SentimentLSTM(vocab_size, embedding_dim, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    print("\nTraining LSTM classifier...")
    batch_size = 32
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()

        if (epoch + 1) % 10 == 0:
            accuracy = 100 * correct / len(X_train)
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Loss: {total_loss/len(X_train):.4f}, "
                  f"Accuracy: {accuracy:.2f}%")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        test_accuracy = 100 * (predicted == y_test).float().mean()
        print(f"\nTest Accuracy: {test_accuracy:.2f}%")


def compare_architectures():
    """Compare different sequence classification architectures."""
    print("\n" + "=" * 70)
    print("Architecture Comparison")
    print("=" * 70)

    vocab_size = 500
    embedding_dim = 32
    hidden_size = 64
    num_classes = 3
    max_len = 15
    num_samples = 300

    sequences, labels = create_synthetic_text_data(num_samples, vocab_size, max_len)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    models = {
        'LSTM': SentimentLSTM(vocab_size, embedding_dim, hidden_size, num_classes),
        'BiLSTM': BiLSTMClassifier(vocab_size, embedding_dim, hidden_size, num_classes),
        'GRU': GRUTextClassifier(vocab_size, embedding_dim, hidden_size, num_classes)
    }

    results = {}

    for name, model in models.items():
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Quick training
        for _ in range(30):
            optimizer.zero_grad()
            outputs = model(padded_sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        with torch.no_grad():
            outputs = model(padded_sequences)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).float().mean()

        num_params = sum(p.numel() for p in model.parameters())
        results[name] = {'accuracy': accuracy.item(), 'params': num_params}

        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy.item():.2%}")
        print(f"  Parameters: {num_params:,}")


def handle_variable_lengths():
    """Demonstrate handling variable-length sequences efficiently."""
    print("\n" + "=" * 70)
    print("Variable-Length Sequence Handling")
    print("=" * 70)

    vocab_size = 100
    embedding_dim = 16
    hidden_size = 32

    # Create sequences of different lengths
    sequences = [
        torch.randint(1, vocab_size, (5,)),
        torch.randint(1, vocab_size, (10,)),
        torch.randint(1, vocab_size, (3,)),
        torch.randint(1, vocab_size, (8,)),
    ]
    lengths = torch.tensor([len(s) for s in sequences])

    print(f"Original sequence lengths: {lengths.tolist()}")

    # Pad sequences
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    print(f"Padded sequences shape: {padded.shape}")

    # Pack sequences for efficient processing
    model = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
    embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    embedded = embedding(padded)

    # Sort by length (required for pack_padded_sequence)
    sorted_lengths, sort_idx = lengths.sort(descending=True)
    sorted_embedded = embedded[sort_idx]

    # Pack
    packed = pack_padded_sequence(sorted_embedded, sorted_lengths.cpu(),
                                   batch_first=True)

    # Process with LSTM
    packed_output, (hidden, cell) = model(packed)

    # Unpack
    output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

    print(f"Output shape: {output.shape}")
    print(f"Output lengths: {output_lengths.tolist()}")
    print("\nPacking allows efficient processing without computing padding!")


def main():
    """Run all sequence classification demonstrations."""
    print("\n" + "=" * 70)
    print(" " * 15 + "SEQUENCE CLASSIFICATION TUTORIAL")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    train_sentiment_classifier()
    compare_architectures()
    handle_variable_lengths()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. LSTMs/GRUs work well for text classification")
    print("2. Use last hidden state or pooling for classification")
    print("3. Bidirectional models capture context from both directions")
    print("4. Padding handles variable-length sequences")
    print("5. Packing improves efficiency for variable lengths")
    print("=" * 70)


if __name__ == "__main__":
    main()
