"""
Sequence-to-Sequence Models - Encoder-Decoder Architecture

This module demonstrates:
1. Basic encoder-decoder architecture
2. Sequence-to-sequence learning
3. Simple machine translation example
4. Teacher forcing
5. Inference with trained seq2seq models

Run: python 06_seq2seq.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class Encoder(nn.Module):
    """Encoder: Processes input sequence into context vector."""

    def __init__(self, input_size: int, embedding_dim: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input sequence (batch, seq_len)
        Returns:
            outputs, (hidden, cell): LSTM outputs and final states
        """
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    """Decoder: Generates output sequence from context vector."""

    def __init__(self, output_size: int, embedding_dim: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hidden: tuple):
        """
        Args:
            x: Input token (batch, 1)
            hidden: (hidden, cell) from encoder or previous step
        Returns:
            output, (hidden, cell)
        """
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)


class Seq2Seq(nn.Module):
    """Complete Sequence-to-Sequence model."""

    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        """
        Args:
            src: Source sequence (batch, src_len)
            trg: Target sequence (batch, trg_len)
            teacher_forcing_ratio: Probability of using teacher forcing
        Returns:
            outputs: Predictions (batch, trg_len, vocab_size)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features

        # Store outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode
        encoder_outputs, hidden = self.encoder(src)

        # First input to decoder is <sos> token
        decoder_input = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            # Decode
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = output.squeeze(1)

            # Teacher forcing: use actual next token
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2)

            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs


class SimpleSeq2Seq(nn.Module):
    """Simplified seq2seq for demonstration."""

    def __init__(self, input_vocab: int, output_vocab: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.LSTM(input_vocab, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(output_vocab, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_vocab)

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        # Encode
        _, (hidden, cell) = self.encoder(src)

        # Decode
        decoder_out, _ = self.decoder(trg, (hidden, cell))
        output = self.output_layer(decoder_out)

        return output


def create_reverse_sequence_data(num_samples: int, seq_len: int, vocab_size: int):
    """Create dataset for sequence reversal task."""
    src_sequences = []
    trg_sequences = []

    for _ in range(num_samples):
        # Random sequence
        seq = torch.randint(1, vocab_size, (seq_len,))
        # Target is reversed sequence with <sos>=0 at start
        reversed_seq = torch.cat([torch.tensor([0]), seq.flip(0)])

        src_sequences.append(seq)
        trg_sequences.append(reversed_seq)

    return torch.stack(src_sequences), torch.stack(trg_sequences)


def train_sequence_reversal():
    """Train seq2seq model to reverse sequences."""
    print("=" * 70)
    print("Sequence Reversal Task")
    print("=" * 70)

    vocab_size = 20
    embedding_dim = 32
    hidden_size = 64
    num_layers = 1
    seq_len = 8
    num_samples = 500

    device = torch.device('cpu')

    # Generate data
    src, trg = create_reverse_sequence_data(num_samples, seq_len, vocab_size)

    print(f"\nDataset:")
    print(f"  Samples: {num_samples}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"\nExample:")
    print(f"  Input:  {src[0].tolist()}")
    print(f"  Target: {trg[0].tolist()}")

    # Create model
    encoder = Encoder(vocab_size, embedding_dim, hidden_size, num_layers)
    decoder = Decoder(vocab_size, embedding_dim, hidden_size, num_layers)
    model = Seq2Seq(encoder, decoder, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    print("\nTraining seq2seq model...")
    num_epochs = 100
    batch_size = 32

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i in range(0, len(src), batch_size):
            batch_src = src[i:i+batch_size].to(device)
            batch_trg = trg[i:i+batch_size].to(device)

            optimizer.zero_grad()
            output = model(batch_src, batch_trg, teacher_forcing_ratio=0.5)

            # Reshape for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg_reshaped = batch_trg[:, 1:].reshape(-1)

            loss = criterion(output, trg_reshaped)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / (len(src) // batch_size)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Test inference
    model.eval()
    with torch.no_grad():
        test_src = src[:5].to(device)
        test_trg = trg[:5].to(device)

        # Inference without teacher forcing
        output = model(test_src, test_trg, teacher_forcing_ratio=0)
        predictions = output.argmax(2)

        print("\nInference Examples:")
        for i in range(5):
            print(f"\nExample {i+1}:")
            print(f"  Input:     {test_src[i].tolist()}")
            print(f"  Target:    {test_trg[i].tolist()}")
            print(f"  Predicted: {predictions[i].tolist()}")


def demonstrate_teacher_forcing():
    """Demonstrate effect of teacher forcing ratio."""
    print("\n" + "=" * 70)
    print("Teacher Forcing Demonstration")
    print("=" * 70)

    vocab_size = 15
    embedding_dim = 16
    hidden_size = 32
    seq_len = 6
    num_samples = 200
    device = torch.device('cpu')

    # Generate data
    src, trg = create_reverse_sequence_data(num_samples, seq_len, vocab_size)

    # Train with different teacher forcing ratios
    ratios = [0.0, 0.5, 1.0]
    results = {}

    for ratio in ratios:
        encoder = Encoder(vocab_size, embedding_dim, hidden_size)
        decoder = Decoder(vocab_size, embedding_dim, hidden_size)
        model = Seq2Seq(encoder, decoder, device).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            output = model(src.to(device), trg.to(device),
                          teacher_forcing_ratio=ratio)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg_reshaped = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg_reshaped.to(device))
            loss.backward()
            optimizer.step()

        results[ratio] = loss.item()
        print(f"\nTeacher forcing ratio {ratio:.1f}: Final loss = {loss.item():.4f}")

    print("\nTeacher forcing helps during training by providing correct context!")


def main():
    """Run all seq2seq demonstrations."""
    print("\n" + "=" * 70)
    print(" " * 15 + "SEQUENCE-TO-SEQUENCE TUTORIAL")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train_sequence_reversal()
    demonstrate_teacher_forcing()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Seq2seq uses encoder-decoder architecture")
    print("2. Encoder compresses input into context vector")
    print("3. Decoder generates output sequence from context")
    print("4. Teacher forcing improves training convergence")
    print("5. Seq2seq is foundation for translation, summarization")
    print("=" * 70)


if __name__ == "__main__":
    main()
