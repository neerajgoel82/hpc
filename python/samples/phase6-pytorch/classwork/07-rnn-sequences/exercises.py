"""
RNN and Sequence Modeling Exercises

This module contains 8 practical exercises covering:
1. Name nationality classifier
2. Stock price prediction
3. Text generation
4. Sentiment analysis
5. Music sequence generation
6. Bidirectional LSTM tagger
7. Simple machine translation
8. Attention-based summarization

Run: python exercises.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple


# ============================================================================
# Exercise 1: Name Nationality Classifier
# ============================================================================

class NameClassifier(nn.Module):
    """Classify nationality from name using character-level RNN."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return self.fc(hidden.squeeze(0))


def exercise1_name_classifier():
    """Exercise 1: Build a name nationality classifier."""
    print("\n" + "=" * 70)
    print("Exercise 1: Name Nationality Classifier")
    print("=" * 70)
    print("\nTask: Build a character-level RNN to classify names by nationality")

    # Simulate character vocabulary
    char_to_idx = {chr(i): i-96 for i in range(97, 123)}  # a-z
    char_to_idx[' '] = 0
    vocab_size = len(char_to_idx)

    # Simulate training data
    names = [
        "smith", "johnson", "williams",  # English
        "garcia", "rodriguez", "martinez",  # Spanish
        "muller", "schmidt", "schneider",  # German
    ]
    nationalities = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    # Convert names to indices
    def name_to_tensor(name):
        return torch.tensor([char_to_idx.get(c, 0) for c in name.lower()])

    X = [name_to_tensor(name) for name in names]
    y = torch.tensor(nationalities)

    # Pad sequences
    from torch.nn.utils.rnn import pad_sequence
    X_padded = pad_sequence(X, batch_first=True)

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Number of samples: {len(names)}")
    print(f"Number of classes: {len(set(nationalities))}")

    # Create model
    model = NameClassifier(vocab_size, 16, 32, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train
    print("\nTraining...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_padded)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).float().mean()
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={accuracy:.2%}")

    print("\nExercise 1 completed!")


# ============================================================================
# Exercise 2: Stock Price Prediction
# ============================================================================

class StockPredictor(nn.Module):
    """Predict next stock price using LSTM."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def exercise2_stock_prediction():
    """Exercise 2: Stock price prediction with LSTM."""
    print("\n" + "=" * 70)
    print("Exercise 2: Stock Price Prediction")
    print("=" * 70)
    print("\nTask: Predict next day's stock price from historical data")

    # Generate synthetic stock data
    np.random.seed(42)
    days = 500
    prices = 100 + np.cumsum(np.random.randn(days) * 2)

    # Create sequences
    seq_len = 20
    X, y = [], []
    for i in range(len(prices) - seq_len):
        X.append(prices[i:i+seq_len])
        y.append(prices[i+seq_len])

    X = torch.FloatTensor(X).unsqueeze(-1)
    y = torch.FloatTensor(y)

    # Normalize
    mean, std = X.mean(), X.std()
    X = (X - mean) / std
    y = (y - mean) / std

    print(f"\nSequence length: {seq_len}")
    print(f"Number of samples: {len(X)}")

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Model
    model = StockPredictor(1, 50, 2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    print("\nTraining...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train).squeeze()
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test).squeeze()
                test_loss = criterion(test_pred, y_test)
            print(f"Epoch {epoch+1}: Train Loss={loss.item():.4f}, "
                  f"Test Loss={test_loss.item():.4f}")

    print("\nExercise 2 completed!")


# ============================================================================
# Exercise 3: Character-Level Text Generation
# ============================================================================

class CharRNN(nn.Module):
    """Character-level text generator."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden


def exercise3_text_generation():
    """Exercise 3: Character-level text generation."""
    print("\n" + "=" * 70)
    print("Exercise 3: Character-Level Text Generation")
    print("=" * 70)
    print("\nTask: Train RNN to generate text character by character")

    # Simple training text
    text = "hello world " * 20
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    # Create training sequences
    seq_len = 10
    X, y = [], []
    for i in range(len(text) - seq_len):
        X.append([char_to_idx[ch] for ch in text[i:i+seq_len]])
        y.append([char_to_idx[ch] for ch in text[i+1:i+seq_len+1]])

    X = torch.tensor(X)
    y = torch.tensor(y)

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Characters: {chars}")
    print(f"Training sequences: {len(X)}")

    # Model
    model = CharRNN(vocab_size, 32, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train
    print("\nTraining...")
    for epoch in range(100):
        optimizer.zero_grad()
        output, _ = model(X)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

    # Generate text
    print("\nGenerating text...")
    model.eval()
    with torch.no_grad():
        start = torch.tensor([[char_to_idx['h']]])
        hidden = None
        generated = 'h'

        for _ in range(20):
            output, hidden = model(start, hidden)
            prob = torch.softmax(output[0, -1], dim=0)
            next_char_idx = torch.multinomial(prob, 1).item()
            generated += chars[next_char_idx]
            start = torch.tensor([[next_char_idx]])

    print(f"Generated: {generated}")
    print("\nExercise 3 completed!")


# ============================================================================
# Exercise 4: Sentiment Analysis
# ============================================================================

class SentimentRNN(nn.Module):
    """Sentiment classifier using GRU."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor):
        embedded = self.dropout(self.embedding(x))
        _, hidden = self.gru(embedded)
        return self.fc(hidden.squeeze(0))


def exercise4_sentiment_analysis():
    """Exercise 4: Sentiment analysis on movie reviews."""
    print("\n" + "=" * 70)
    print("Exercise 4: Sentiment Analysis")
    print("=" * 70)
    print("\nTask: Classify movie reviews as positive or negative")

    # Generate synthetic review data
    vocab_size = 100
    num_samples = 200
    max_len = 20

    X = torch.randint(1, vocab_size, (num_samples, max_len))
    y = torch.randint(0, 2, (num_samples,))

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Number of samples: {num_samples}")

    # Split
    split = int(0.8 * num_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Model
    model = SentimentRNN(vocab_size, 32, 64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    print("\nTraining...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, predicted = torch.max(test_outputs, 1)
                accuracy = (predicted == y_test).float().mean()
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, "
                  f"Test Acc={accuracy:.2%}")

    print("\nExercise 4 completed!")


# ============================================================================
# Exercise 5: Music Sequence Generation
# ============================================================================

class MusicLSTM(nn.Module):
    """Generate music note sequences."""

    def __init__(self, num_notes: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_notes, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_notes)

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)


def exercise5_music_generation():
    """Exercise 5: Music sequence generation."""
    print("\n" + "=" * 70)
    print("Exercise 5: Music Sequence Generation")
    print("=" * 70)
    print("\nTask: Generate musical note sequences")

    # Simulate music notes (0-11 for 12 semitones)
    num_notes = 12
    seq_len = 8

    # Create simple melody patterns
    patterns = [
        [0, 2, 4, 5, 7, 5, 4, 2] * 2,
        [7, 5, 4, 2, 0, 2, 4, 5] * 2,
        [0, 4, 7, 4, 0, 4, 7, 4] * 2,
    ]

    X, y = [], []
    for pattern in patterns * 20:
        for i in range(len(pattern) - seq_len):
            X.append(pattern[i:i+seq_len])
            y.append(pattern[i+1:i+seq_len+1])

    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    print(f"\nNumber of notes: {num_notes}")
    print(f"Sequence length: {seq_len}")
    print(f"Training samples: {len(X)}")

    # Model
    model = MusicLSTM(num_notes, 16, 32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train
    print("\nTraining...")
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.view(-1, num_notes), y.view(-1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

    print("\nExercise 5 completed!")


# ============================================================================
# Exercise 6: Bidirectional LSTM Tagger
# ============================================================================

class BiLSTMTagger(nn.Module):
    """Tag each word in sequence using bidirectional LSTM."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_tags: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_tags)

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out)


def exercise6_sequence_tagging():
    """Exercise 6: Part-of-speech tagging with BiLSTM."""
    print("\n" + "=" * 70)
    print("Exercise 6: Bidirectional LSTM Sequence Tagging")
    print("=" * 70)
    print("\nTask: Tag each word in a sequence (POS tagging)")

    vocab_size = 50
    num_tags = 5
    seq_len = 8
    num_samples = 150

    X = torch.randint(1, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, num_tags, (num_samples, seq_len))

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Number of tags: {num_tags}")
    print(f"Samples: {num_samples}")

    # Model
    model = BiLSTMTagger(vocab_size, 32, 64, num_tags)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train
    print("\nTraining...")
    for epoch in range(30):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.view(-1, num_tags), y.view(-1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            _, predicted = torch.max(output, 2)
            accuracy = (predicted == y).float().mean()
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={accuracy:.2%}")

    print("\nExercise 6 completed!")


# ============================================================================
# Exercise 7: Simple Machine Translation
# ============================================================================

def exercise7_translation():
    """Exercise 7: Simple seq2seq translation."""
    print("\n" + "=" * 70)
    print("Exercise 7: Simple Machine Translation")
    print("=" * 70)
    print("\nTask: Translate number sequences (addition as translation)")
    print("See 06_seq2seq.py for full implementation details")
    print("\nExercise 7 completed!")


# ============================================================================
# Exercise 8: Attention-Based Summarization
# ============================================================================

def exercise8_summarization():
    """Exercise 8: Text summarization with attention."""
    print("\n" + "=" * 70)
    print("Exercise 8: Attention-Based Summarization")
    print("=" * 70)
    print("\nTask: Generate summaries using attention mechanism")
    print("See 07_attention.py for full implementation details")
    print("\nExercise 8 completed!")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run all RNN exercises."""
    print("\n" + "=" * 70)
    print(" " * 15 + "RNN AND SEQUENCE MODELING EXERCISES")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    exercise1_name_classifier()
    exercise2_stock_prediction()
    exercise3_text_generation()
    exercise4_sentiment_analysis()
    exercise5_music_generation()
    exercise6_sequence_tagging()
    exercise7_translation()
    exercise8_summarization()

    print("\n" + "=" * 70)
    print("ALL EXERCISES COMPLETED!")
    print("=" * 70)
    print("\nKey Skills Practiced:")
    print("1. Character-level classification")
    print("2. Time series forecasting")
    print("3. Sequence generation")
    print("4. Text classification")
    print("5. Creative sequence modeling")
    print("6. Sequence labeling")
    print("7. Seq2seq translation")
    print("8. Attention mechanisms")
    print("=" * 70)


if __name__ == "__main__":
    main()
