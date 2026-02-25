"""
LSTM Networks - Long Short-Term Memory Architecture

This module demonstrates:
1. LSTM architecture and components (gates)
2. Manual LSTM cell implementation
3. Solving vanishing gradient problem
4. Bidirectional LSTM
5. Comparison with basic RNN

Run: python 02_lstm.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SimpleLSTMCell(nn.Module):
    """Manual implementation of LSTM cell to understand gates."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Forget gate
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        # Input gate
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        # Candidate cell state
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        # Output gate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, hidden: tuple):
        """
        Args:
            x: Input (batch_size, input_size)
            hidden: Tuple of (h, c) each (batch_size, hidden_size)
        Returns:
            New (h, c) tuple
        """
        h_prev, c_prev = hidden

        # Concatenate input and previous hidden
        combined = torch.cat([x, h_prev], dim=1)

        # Forget gate: decides what to forget from cell state
        f_t = torch.sigmoid(self.W_f(combined))

        # Input gate: decides what new information to store
        i_t = torch.sigmoid(self.W_i(combined))

        # Candidate cell state
        c_tilde = torch.tanh(self.W_c(combined))

        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde

        # Output gate: decides what to output
        o_t = torch.sigmoid(self.W_o(combined))

        # Update hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class ManualLSTM(nn.Module):
    """LSTM network using manual cell implementation."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = SimpleLSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hidden=None):
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size)
            c = torch.zeros(batch_size, self.hidden_size)
            hidden = (h, c)

        outputs = []
        for t in range(seq_len):
            h, c = self.lstm_cell(x[:, t, :], hidden)
            hidden = (h, c)
            output = self.fc(h)
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1), hidden


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for sequence classification."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor):
        # LSTM output: (batch, seq, hidden*2)
        lstm_out, _ = self.lstm(x)
        # Use last output for classification
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


def demonstrate_lstm_gates():
    """Visualize LSTM gate activations."""
    print("=" * 70)
    print("LSTM Gates Demonstration")
    print("=" * 70)

    input_size = 5
    hidden_size = 8
    batch_size = 2

    lstm_cell = SimpleLSTMCell(input_size, hidden_size)

    x = torch.randn(batch_size, input_size)
    h = torch.zeros(batch_size, hidden_size)
    c = torch.zeros(batch_size, hidden_size)

    print(f"\nInput shape: {x.shape}")
    print(f"Hidden state shape: {h.shape}")
    print(f"Cell state shape: {c.shape}")

    # Forward pass
    h_new, c_new = lstm_cell(x, (h, c))

    print(f"\nNew hidden state shape: {h_new.shape}")
    print(f"New cell state shape: {c_new.shape}")
    print(f"\nCell state change: {torch.abs(c_new - c).mean().item():.4f}")
    print(f"Hidden state activation range: [{h_new.min():.3f}, {h_new.max():.3f}]")


def compare_rnn_lstm():
    """Compare RNN vs LSTM on long sequence task."""
    print("\n" + "=" * 70)
    print("RNN vs LSTM on Long Sequences")
    print("=" * 70)

    input_size = 1
    hidden_size = 32
    output_size = 1
    seq_len = 50

    # Create models
    rnn_model = nn.RNN(input_size, hidden_size, batch_first=True)
    rnn_fc = nn.Linear(hidden_size, output_size)

    lstm_model = nn.LSTM(input_size, hidden_size, batch_first=True)
    lstm_fc = nn.Linear(hidden_size, output_size)

    # Generate data with long-term dependency
    num_samples = 100
    X = torch.randn(num_samples, seq_len, input_size)
    # Target depends on first value
    Y = X[:, 0, :].repeat(1, seq_len).reshape(num_samples, seq_len, output_size)

    # Train RNN
    print("\nTraining RNN...")
    rnn_optimizer = optim.Adam(
        list(rnn_model.parameters()) + list(rnn_fc.parameters()), lr=0.01
    )
    criterion = nn.MSELoss()

    for epoch in range(100):
        rnn_optimizer.zero_grad()
        rnn_out, _ = rnn_model(X)
        rnn_pred = rnn_fc(rnn_out)
        rnn_loss = criterion(rnn_pred, Y)
        rnn_loss.backward()
        rnn_optimizer.step()

    print(f"RNN Final Loss: {rnn_loss.item():.4f}")

    # Train LSTM
    print("\nTraining LSTM...")
    lstm_optimizer = optim.Adam(
        list(lstm_model.parameters()) + list(lstm_fc.parameters()), lr=0.01
    )

    for epoch in range(100):
        lstm_optimizer.zero_grad()
        lstm_out, _ = lstm_model(X)
        lstm_pred = lstm_fc(lstm_out)
        lstm_loss = criterion(lstm_pred, Y)
        lstm_loss.backward()
        lstm_optimizer.step()

    print(f"LSTM Final Loss: {lstm_loss.item():.4f}")
    print(f"\nLSTM performs better on long-term dependencies!")


def bidirectional_lstm_example():
    """Demonstrate bidirectional LSTM."""
    print("\n" + "=" * 70)
    print("Bidirectional LSTM Example")
    print("=" * 70)

    input_size = 10
    hidden_size = 20
    num_classes = 3
    seq_len = 15
    batch_size = 32

    model = BiLSTMClassifier(input_size, hidden_size, num_classes)

    # Generate random data
    X = torch.randn(batch_size, seq_len, input_size)
    y = torch.randint(0, num_classes, (batch_size,))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining bidirectional LSTM classifier...")
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).float().mean()
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}, "
                  f"Accuracy: {accuracy.item():.2%}")


def main():
    """Run all LSTM demonstrations."""
    print("\n" + "=" * 70)
    print(" " * 20 + "LSTM NETWORKS TUTORIAL")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    demonstrate_lstm_gates()
    compare_rnn_lstm()
    bidirectional_lstm_example()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. LSTMs use gates to control information flow")
    print("2. Cell state helps maintain long-term memory")
    print("3. LSTMs solve vanishing gradient problem")
    print("4. Bidirectional LSTMs process sequences both ways")
    print("5. LSTMs outperform RNNs on long sequences")
    print("=" * 70)


if __name__ == "__main__":
    main()
