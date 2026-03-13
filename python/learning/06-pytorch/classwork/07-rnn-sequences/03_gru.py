"""
GRU Networks - Gated Recurrent Units

This module demonstrates:
1. GRU architecture and components
2. Manual GRU cell implementation
3. Comparison with LSTM and RNN
4. GRU efficiency advantages
5. Practical GRU applications

Run: python 03_gru.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time


class SimpleGRUCell(nn.Module):
    """Manual implementation of GRU cell to understand gates."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Reset gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        # New candidate hidden state
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch_size, input_size)
            hidden: Previous hidden state (batch_size, hidden_size)
        Returns:
            New hidden state (batch_size, hidden_size)
        """
        # Concatenate input and previous hidden
        combined = torch.cat([x, hidden], dim=1)

        # Reset gate: decides what to forget
        r_t = torch.sigmoid(self.W_r(combined))

        # Update gate: decides how much to update
        z_t = torch.sigmoid(self.W_z(combined))

        # Candidate hidden state (with reset applied)
        combined_reset = torch.cat([x, r_t * hidden], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))

        # Final hidden state (interpolation between old and new)
        h_t = (1 - z_t) * hidden + z_t * h_tilde

        return h_t


class ManualGRU(nn.Module):
    """GRU network using manual cell implementation."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_cell = SimpleGRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hidden=None):
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size)

        outputs = []
        for t in range(seq_len):
            hidden = self.gru_cell(x[:, t, :], hidden)
            output = self.fc(hidden)
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1), hidden


class GRULanguageModel(nn.Module):
    """Character-level language model using GRU."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        output = self.fc(output)
        return output, hidden


def demonstrate_gru_gates():
    """Visualize GRU gate activations."""
    print("=" * 70)
    print("GRU Gates Demonstration")
    print("=" * 70)

    input_size = 5
    hidden_size = 8
    batch_size = 2

    gru_cell = SimpleGRUCell(input_size, hidden_size)

    x = torch.randn(batch_size, input_size)
    h = torch.zeros(batch_size, hidden_size)

    print(f"\nInput shape: {x.shape}")
    print(f"Hidden state shape: {h.shape}")

    # Forward pass
    h_new = gru_cell(x, h)

    print(f"\nNew hidden state shape: {h_new.shape}")
    print(f"Hidden state change: {torch.abs(h_new - h).mean().item():.4f}")
    print(f"Hidden activation range: [{h_new.min():.3f}, {h_new.max():.3f}]")

    print("\nGRU has 2 gates (reset, update) vs LSTM's 3 gates")
    print("This makes GRU computationally more efficient")


def compare_rnn_lstm_gru():
    """Compare RNN, LSTM, and GRU on sequence task."""
    print("\n" + "=" * 70)
    print("RNN vs LSTM vs GRU Comparison")
    print("=" * 70)

    input_size = 10
    hidden_size = 64
    output_size = 5
    seq_len = 30
    batch_size = 32
    num_samples = 200

    # Generate synthetic data
    X = torch.randn(num_samples, seq_len, input_size)
    Y = torch.randn(num_samples, seq_len, output_size)

    models = {
        'RNN': nn.RNN(input_size, hidden_size, batch_first=True),
        'LSTM': nn.LSTM(input_size, hidden_size, batch_first=True),
        'GRU': nn.GRU(input_size, hidden_size, batch_first=True)
    }

    results = {}

    for name, model in models.items():
        fc = nn.Linear(hidden_size, output_size)
        optimizer = optim.Adam(list(model.parameters()) + list(fc.parameters()), lr=0.01)
        criterion = nn.MSELoss()

        print(f"\nTraining {name}...")
        start_time = time.time()

        for epoch in range(50):
            optimizer.zero_grad()
            output, _ = model(X)
            pred = fc(output)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()

        training_time = time.time() - start_time
        final_loss = loss.item()

        results[name] = {
            'loss': final_loss,
            'time': training_time,
            'params': sum(p.numel() for p in model.parameters())
        }

        print(f"{name} - Loss: {final_loss:.4f}, Time: {training_time:.2f}s, "
              f"Params: {results[name]['params']}")

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY:")
    print("=" * 70)
    for name, result in results.items():
        print(f"{name:6s} - Params: {result['params']:6d}, Time: {result['time']:5.2f}s")


def gru_sequence_generation():
    """Train GRU to generate sequences."""
    print("\n" + "=" * 70)
    print("GRU Sequence Generation Example")
    print("=" * 70)

    # Simple sequence: learn to repeat pattern
    seq_len = 10
    vocab_size = 5
    embedding_dim = 16
    hidden_size = 32

    model = GRULanguageModel(vocab_size, embedding_dim, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Create training data: repeating pattern
    num_samples = 100
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    # Target is input shifted by 1
    Y = torch.cat([X[:, 1:], torch.randint(0, vocab_size, (num_samples, 1))], dim=1)

    print("\nTraining GRU for sequence generation...")
    for epoch in range(100):
        optimizer.zero_grad()
        output, _ = model(X)
        loss = criterion(output.view(-1, vocab_size), Y.view(-1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            # Calculate accuracy
            _, predicted = torch.max(output, 2)
            accuracy = (predicted == Y).float().mean()
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}, "
                  f"Accuracy: {accuracy.item():.2%}")

    # Generate sequence
    with torch.no_grad():
        test_input = torch.randint(0, vocab_size, (1, seq_len))
        output, _ = model(test_input)
        _, predicted = torch.max(output, 2)
        print(f"\nInput sequence:     {test_input[0].tolist()}")
        print(f"Generated sequence: {predicted[0].tolist()}")


def stacked_gru_example():
    """Demonstrate stacked (multi-layer) GRU."""
    print("\n" + "=" * 70)
    print("Stacked GRU Example")
    print("=" * 70)

    input_size = 10
    hidden_size = 32
    num_layers = 3
    seq_len = 20
    batch_size = 16

    # Multi-layer GRU
    gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    x = torch.randn(batch_size, seq_len, input_size)
    output, hidden = gru(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"  (num_layers={num_layers}, batch_size={batch_size}, hidden_size={hidden_size})")

    total_params = sum(p.numel() for p in gru.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Stacked layers allow learning hierarchical representations")


def main():
    """Run all GRU demonstrations."""
    print("\n" + "=" * 70)
    print(" " * 20 + "GRU NETWORKS TUTORIAL")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    demonstrate_gru_gates()
    compare_rnn_lstm_gru()
    gru_sequence_generation()
    stacked_gru_example()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. GRU uses 2 gates (reset, update) vs LSTM's 3 gates")
    print("2. GRU is computationally more efficient than LSTM")
    print("3. GRU often performs similarly to LSTM")
    print("4. GRU has fewer parameters, trains faster")
    print("5. Choose GRU for efficiency, LSTM for max performance")
    print("=" * 70)


if __name__ == "__main__":
    main()
