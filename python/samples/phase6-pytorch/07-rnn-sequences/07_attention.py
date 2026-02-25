"""
Attention Mechanism - Introduction to Attention for Sequence Models

This module demonstrates:
1. Attention mechanism basics
2. Additive (Bahdanau) attention
3. Multiplicative (Luong) attention
4. Attention visualization
5. Seq2seq with attention

Run: python 07_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


class BahdanauAttention(nn.Module):
    """Additive attention mechanism (Bahdanau et al., 2015)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_key = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        """
        Args:
            query: Decoder hidden state (batch, hidden_size)
            keys: Encoder outputs (batch, seq_len, hidden_size)
        Returns:
            context: Weighted sum of keys (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        # query: (batch, hidden_size) -> (batch, 1, hidden_size)
        query = query.unsqueeze(1)

        # Compute attention scores
        # (batch, seq_len, hidden_size)
        score = torch.tanh(self.W_query(query) + self.W_key(keys))

        # (batch, seq_len, 1) -> (batch, seq_len)
        attention_weights = F.softmax(self.v(score).squeeze(2), dim=1)

        # Compute context vector
        # (batch, seq_len, 1) * (batch, seq_len, hidden_size)
        context = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)

        return context, attention_weights


class LuongAttention(nn.Module):
    """Multiplicative attention mechanism (Luong et al., 2015)."""

    def __init__(self, hidden_size: int, method: str = 'dot'):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size

        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.W = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        """
        Args:
            query: Decoder hidden state (batch, hidden_size)
            keys: Encoder outputs (batch, seq_len, hidden_size)
        Returns:
            context, attention_weights
        """
        if self.method == 'dot':
            # Dot product attention
            scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
        elif self.method == 'general':
            # General attention
            scores = torch.bmm(keys, self.W(query).unsqueeze(2)).squeeze(2)
        elif self.method == 'concat':
            # Concat attention
            query_expanded = query.unsqueeze(1).expand(-1, keys.size(1), -1)
            concat = torch.cat([keys, query_expanded], dim=2)
            scores = self.v(torch.tanh(self.W(concat))).squeeze(2)

        attention_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)

        return context, attention_weights


class AttentionDecoder(nn.Module):
    """Decoder with attention mechanism."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = attention
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: tuple, encoder_outputs: torch.Tensor):
        """
        Args:
            x: Input token (batch, 1)
            hidden: (hidden, cell)
            encoder_outputs: (batch, src_len, hidden_size)
        Returns:
            output, (hidden, cell), attention_weights
        """
        embedded = self.embedding(x)

        # Compute attention
        context, attention_weights = self.attention(hidden[0][-1], encoder_outputs)

        # Concatenate embedded input and context
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)

        # LSTM step
        output, (hidden, cell) = self.lstm(lstm_input, hidden)

        # Prediction
        prediction = self.fc(output)

        return prediction, (hidden, cell), attention_weights


class Encoder(nn.Module):
    """Simple encoder for attention seq2seq."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)


def demonstrate_attention_basics():
    """Demonstrate basic attention computation."""
    print("=" * 70)
    print("Attention Mechanism Basics")
    print("=" * 70)

    batch_size = 2
    seq_len = 5
    hidden_size = 8

    # Simulated encoder outputs (keys)
    encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)

    # Simulated decoder hidden state (query)
    decoder_hidden = torch.randn(batch_size, hidden_size)

    print(f"\nEncoder outputs shape: {encoder_outputs.shape}")
    print(f"Decoder hidden shape: {decoder_hidden.shape}")

    # Bahdanau attention
    bahdanau_attn = BahdanauAttention(hidden_size)
    context_b, weights_b = bahdanau_attn(decoder_hidden, encoder_outputs)

    print(f"\nBahdanau Attention:")
    print(f"  Context shape: {context_b.shape}")
    print(f"  Weights shape: {weights_b.shape}")
    print(f"  Weights sum: {weights_b[0].sum().item():.4f}")
    print(f"  Weights: {weights_b[0].detach().numpy()}")

    # Luong attention
    luong_attn = LuongAttention(hidden_size, method='dot')
    context_l, weights_l = luong_attn(decoder_hidden, encoder_outputs)

    print(f"\nLuong Attention (dot):")
    print(f"  Context shape: {context_l.shape}")
    print(f"  Weights: {weights_l[0].detach().numpy()}")


def compare_attention_methods():
    """Compare different attention mechanisms."""
    print("\n" + "=" * 70)
    print("Attention Methods Comparison")
    print("=" * 70)

    hidden_size = 16
    seq_len = 8
    batch_size = 4

    encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)
    query = torch.randn(batch_size, hidden_size)

    methods = {
        'Bahdanau': BahdanauAttention(hidden_size),
        'Luong (dot)': LuongAttention(hidden_size, 'dot'),
        'Luong (general)': LuongAttention(hidden_size, 'general'),
        'Luong (concat)': LuongAttention(hidden_size, 'concat')
    }

    print("\nAttention weight distributions:")
    for name, attention in methods.items():
        _, weights = attention(query, encoder_outputs)
        print(f"\n{name}:")
        print(f"  Mean: {weights.mean().item():.4f}")
        print(f"  Std:  {weights.std().item():.4f}")
        print(f"  Max:  {weights.max().item():.4f}")
        print(f"  Example: {weights[0].detach().numpy()}")


def train_seq2seq_with_attention():
    """Train seq2seq with attention on sequence reversal."""
    print("\n" + "=" * 70)
    print("Seq2Seq with Attention Training")
    print("=" * 70)

    vocab_size = 20
    embedding_dim = 32
    hidden_size = 64
    seq_len = 6
    num_samples = 300
    device = torch.device('cpu')

    # Generate data (sequence reversal task)
    src_sequences = []
    trg_sequences = []

    for _ in range(num_samples):
        seq = torch.randint(1, vocab_size, (seq_len,))
        reversed_seq = torch.cat([torch.tensor([0]), seq.flip(0)])
        src_sequences.append(seq)
        trg_sequences.append(reversed_seq)

    src = torch.stack(src_sequences)
    trg = torch.stack(trg_sequences)

    print(f"\nDataset:")
    print(f"  Samples: {num_samples}")
    print(f"  Sequence length: {seq_len}")

    # Create models
    encoder = Encoder(vocab_size, embedding_dim, hidden_size)
    attention = BahdanauAttention(hidden_size)
    decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_size, attention)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=0.001
    )

    # Training
    print("\nTraining seq2seq with attention...")
    num_epochs = 50
    batch_size = 32

    for epoch in range(num_epochs):
        total_loss = 0

        for i in range(0, len(src), batch_size):
            batch_src = src[i:i+batch_size].to(device)
            batch_trg = trg[i:i+batch_size].to(device)

            optimizer.zero_grad()

            # Encode
            encoder_outputs, (hidden, cell) = encoder(batch_src)

            # Decode with attention
            decoder_input = batch_trg[:, 0].unsqueeze(1)
            loss = 0

            for t in range(1, batch_trg.size(1)):
                output, (hidden, cell), _ = decoder(
                    decoder_input, (hidden, cell), encoder_outputs
                )
                loss += criterion(output.squeeze(1), batch_trg[:, t])
                decoder_input = batch_trg[:, t].unsqueeze(1)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / (len(src) // batch_size)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Visualize attention
    print("\nGenerating attention visualization...")
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        test_src = src[0:1].to(device)
        test_trg = trg[0:1].to(device)

        encoder_outputs, (hidden, cell) = encoder(test_src)
        decoder_input = test_trg[:, 0].unsqueeze(1)

        attention_weights_list = []
        for t in range(1, test_trg.size(1)):
            output, (hidden, cell), attention_weights = decoder(
                decoder_input, (hidden, cell), encoder_outputs
            )
            attention_weights_list.append(attention_weights[0].cpu().numpy())
            decoder_input = output.argmax(2)

        print(f"\nAttention weights shape: ({len(attention_weights_list)}, {len(attention_weights_list[0])})")
        print("Each row shows attention over input sequence for one output timestep")


def main():
    """Run all attention demonstrations."""
    print("\n" + "=" * 70)
    print(" " * 20 + "ATTENTION MECHANISM TUTORIAL")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    demonstrate_attention_basics()
    compare_attention_methods()
    train_seq2seq_with_attention()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Attention allows decoder to focus on relevant input parts")
    print("2. Attention weights sum to 1 (probability distribution)")
    print("3. Bahdanau uses additive, Luong uses multiplicative")
    print("4. Attention improves long sequence performance")
    print("5. Attention weights are interpretable")
    print("=" * 70)


if __name__ == "__main__":
    main()
