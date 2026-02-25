"""
RNN Basics - Simple Recurrent Neural Network Architecture

This module demonstrates:
1. Basic RNN architecture and components
2. Sequence processing and hidden states
3. Manual RNN implementation
4. PyTorch's built-in RNN
5. Handling variable-length sequences

Run: python 01_rnn_basics.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SimpleRNNCell(nn.Module):
    """Manual implementation of a single RNN cell."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, input_size)
            hidden: Previous hidden state (batch_size, hidden_size)
        Returns:
            New hidden state (batch_size, hidden_size)
        """
        hidden = torch.tanh(self.W_ih(x) + self.W_hh(hidden))
        return hidden


class ManualRNN(nn.Module):
    """Manual RNN implementation processing full sequences."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = SimpleRNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Args:
            x: Input (batch_size, seq_len, input_size)
            hidden: Initial hidden state (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size)

        outputs = []
        for t in range(seq_len):
            hidden = self.rnn_cell(x[:, t, :], hidden)
            output = self.fc(hidden)
            outputs.append(output.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden


class CharRNN(nn.Module):
    """Character-level RNN using PyTorch's built-in RNN."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size: int):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


def demonstrate_rnn_unrolling():
    """Visualize how RNN processes sequences over time."""
    print("=" * 70)
    print("RNN Sequence Processing Demonstration")
    print("=" * 70)

    input_size = 3
    hidden_size = 4
    seq_len = 5
    batch_size = 2

    rnn_cell = SimpleRNNCell(input_size, hidden_size)
    x = torch.randn(batch_size, seq_len, input_size)
    hidden = torch.zeros(batch_size, hidden_size)

    print(f"\nInput shape: {x.shape}")
    print(f"Initial hidden state shape: {hidden.shape}")

    for t in range(seq_len):
        hidden = rnn_cell(x[:, t, :], hidden)
        print(f"\nTimestep {t+1}:")
        print(f"  Input[0]: [{x[0, t, 0]:.3f}, {x[0, t, 1]:.3f}, {x[0, t, 2]:.3f}]")
        print(f"  Hidden[0]: [{hidden[0, 0]:.3f}, {hidden[0, 1]:.3f}, ...]")


def compare_implementations():
    """Compare manual vs PyTorch RNN implementations."""
    print("\n" + "=" * 70)
    print("Manual vs PyTorch RNN Comparison")
    print("=" * 70)

    input_size = 5
    hidden_size = 10
    output_size = 3
    seq_len = 7
    batch_size = 4

    manual_rnn = ManualRNN(input_size, hidden_size, output_size)
    pytorch_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    pytorch_fc = nn.Linear(hidden_size, output_size)

    x = torch.randn(batch_size, seq_len, input_size)

    manual_output, manual_hidden = manual_rnn(x)
    pytorch_output, pytorch_hidden = pytorch_rnn(x)
    pytorch_output = pytorch_fc(pytorch_output)

    print(f"\nInput shape: {x.shape}")
    print(f"Manual RNN output: {manual_output.shape}")
    print(f"PyTorch RNN output: {pytorch_output.shape}")
    print(f"Manual hidden: {manual_hidden.shape}")
    print(f"PyTorch hidden: {pytorch_hidden.shape}")


def sequence_to_sequence_example():
    """Simple sequence-to-sequence prediction task."""
    print("\n" + "=" * 70)
    print("Sequence-to-Sequence Learning Example")
    print("=" * 70)

    input_size = 1
    hidden_size = 20
    output_size = 1
    seq_len = 10

    model = ManualRNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training data: sine wave prediction
    num_samples = 100
    X = torch.zeros(num_samples, seq_len, input_size)
    Y = torch.zeros(num_samples, seq_len, output_size)

    for i in range(num_samples):
        phase = np.random.rand() * 2 * np.pi
        freq = np.random.rand() * 2 + 1
        t = np.linspace(0, 2*np.pi, seq_len+1)
        signal = np.sin(freq * t + phase)
        X[i, :, 0] = torch.FloatTensor(signal[:-1])
        Y[i, :, 0] = torch.FloatTensor(signal[1:])

    print("\nTraining RNN to predict next value in sine wave...")
    for epoch in range(200):
        optimizer.zero_grad()
        outputs, _ = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")

    # Test
    test_input = torch.zeros(1, seq_len, input_size)
    t = np.linspace(0, 2*np.pi, seq_len+1)
    test_signal = np.sin(2 * t)
    test_input[0, :, 0] = torch.FloatTensor(test_signal[:-1])
    test_target = torch.FloatTensor(test_signal[1:])

    with torch.no_grad():
        prediction, _ = model(test_input)

    test_loss = nn.MSELoss()(prediction[0, :, 0], test_target)
    print(f"\nTest prediction error: {test_loss.item():.4f}")


def main():
    """Run all RNN basics demonstrations."""
    print("\n" + "=" * 70)
    print(" " * 20 + "RNN BASICS TUTORIAL")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    demonstrate_rnn_unrolling()
    compare_implementations()
    sequence_to_sequence_example()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. RNNs process sequences one element at a time")
    print("2. Hidden state carries information across time steps")
    print("3. Same weights are shared across all time steps")
    print("4. RNNs can learn temporal dependencies")
    print("5. PyTorch provides optimized RNN implementations")
    print("=" * 70)


if __name__ == "__main__":
    main()
