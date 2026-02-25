"""
Project: Sentiment Analysis
===========================
Classify text sentiment using RNN/LSTM.

Dataset: IMDB reviews or Twitter sentiment
Goals:
- Preprocess text data
- Build LSTM model
- Train sentiment classifier
- Evaluate on test set
- Make predictions on new text

Skills: RNNs, LSTMs, Text processing, Classification
"""

import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        # TODO: Define LSTM architecture
        pass
    
    def forward(self, text):
        # TODO: Implement forward pass
        pass

def main():
    print("Sentiment Analysis Project")
    print("=" * 60)
    print("TODO: Implement sentiment analysis")
    print("=" * 60)

if __name__ == "__main__":
    main()
