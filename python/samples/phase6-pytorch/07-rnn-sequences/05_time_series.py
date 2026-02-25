"""
Time Series Prediction - Forecasting Future Values in Sequences

This module demonstrates:
1. Univariate time series forecasting
2. Multivariate time series prediction
3. Multi-step ahead prediction
4. LSTM for time series
5. Handling trends and seasonality

Run: python 05_time_series.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class TimeSeriesLSTM(nn.Module):
    """LSTM for univariate time series prediction."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        # Use last time step for prediction
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions


class MultivariateLSTM(nn.Module):
    """LSTM for multivariate time series prediction."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class Seq2SeqLSTM(nn.Module):
    """LSTM for multi-step ahead prediction."""

    def __init__(self, input_size: int, hidden_size: int, output_steps: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_steps = output_steps

        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor):
        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Decode
        decoder_input = x[:, -1:, :]  # Last time step
        predictions = []

        for _ in range(self.output_steps):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(decoder_output)
            predictions.append(pred)
            decoder_input = pred

        return torch.cat(predictions, dim=1)


def create_time_series_data(num_samples: int, seq_len: int, forecast_horizon: int):
    """Generate synthetic time series data."""
    X, y = [], []

    for i in range(num_samples):
        # Create time series with trend and seasonality
        t = np.arange(seq_len + forecast_horizon)
        trend = 0.05 * t
        seasonality = np.sin(2 * np.pi * t / 12)
        noise = np.random.randn(len(t)) * 0.1

        series = trend + seasonality + noise

        X.append(series[:seq_len])
        y.append(series[seq_len:seq_len+forecast_horizon])

    return torch.FloatTensor(X).unsqueeze(-1), torch.FloatTensor(y)


def univariate_forecasting():
    """Single-step univariate time series forecasting."""
    print("=" * 70)
    print("Univariate Time Series Forecasting")
    print("=" * 70)

    seq_len = 24
    hidden_size = 64
    num_layers = 2
    num_samples = 500

    # Generate data
    X, y = create_time_series_data(num_samples, seq_len, 1)
    y = y.squeeze(-1)

    # Split train/test
    split = int(0.8 * num_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nDataset:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Forecast horizon: 1 step")

    # Create model
    model = TimeSeriesLSTM(input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    print("\nTraining LSTM for time series forecasting...")
    num_epochs = 100
    batch_size = 32

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / (len(X_train) // batch_size)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train)
        train_mse = criterion(train_pred, y_train)

        test_pred = model(X_test)
        test_mse = criterion(test_pred, y_test)

    print(f"\nResults:")
    print(f"  Train MSE: {train_mse.item():.4f}")
    print(f"  Test MSE: {test_mse.item():.4f}")

    # Calculate RMSE
    train_rmse = np.sqrt(train_mse.item())
    test_rmse = np.sqrt(test_mse.item())
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")


def multivariate_forecasting():
    """Multivariate time series prediction."""
    print("\n" + "=" * 70)
    print("Multivariate Time Series Forecasting")
    print("=" * 70)

    seq_len = 20
    num_features = 3
    hidden_size = 64
    num_samples = 400

    # Generate multivariate data
    X = []
    y = []

    for _ in range(num_samples):
        # Create correlated time series
        t = np.arange(seq_len + 1)
        feature1 = np.sin(2 * np.pi * t / 10) + np.random.randn(len(t)) * 0.1
        feature2 = np.cos(2 * np.pi * t / 10) + np.random.randn(len(t)) * 0.1
        feature3 = (feature1 + feature2) / 2 + np.random.randn(len(t)) * 0.1

        series = np.stack([feature1, feature2, feature3], axis=1)
        X.append(series[:-1])
        y.append(series[-1])

    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y))

    # Split
    split = int(0.8 * num_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nDataset:")
    print(f"  Input shape: {X_train.shape}")
    print(f"  Output shape: {y_train.shape}")
    print(f"  Number of features: {num_features}")

    # Model
    model = MultivariateLSTM(num_features, hidden_size, num_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    print("\nTraining multivariate LSTM...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_mse = criterion(test_pred, y_test)
        print(f"\nTest MSE: {test_mse.item():.4f}")


def multi_step_forecasting():
    """Multi-step ahead prediction."""
    print("\n" + "=" * 70)
    print("Multi-Step Ahead Forecasting")
    print("=" * 70)

    seq_len = 20
    forecast_steps = 5
    hidden_size = 64
    num_samples = 300

    # Generate data
    X, y = create_time_series_data(num_samples, seq_len, forecast_steps)

    split = int(0.8 * num_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nDataset:")
    print(f"  Input sequence length: {seq_len}")
    print(f"  Forecast horizon: {forecast_steps} steps")

    # Model
    model = Seq2SeqLSTM(input_size=1, hidden_size=hidden_size,
                        output_steps=forecast_steps)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    print("\nTraining seq2seq LSTM for multi-step forecasting...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions.squeeze(-1), y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_mse = criterion(test_pred.squeeze(-1), y_test)
        print(f"\nTest MSE: {test_mse.item():.4f}")

        # Show example prediction
        example_pred = test_pred[0].squeeze().numpy()
        example_true = y_test[0].numpy()

        print(f"\nExample prediction:")
        print(f"  Predicted: {example_pred}")
        print(f"  Actual:    {example_true}")


def main():
    """Run all time series prediction demonstrations."""
    print("\n" + "=" * 70)
    print(" " * 15 + "TIME SERIES PREDICTION TUTORIAL")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    univariate_forecasting()
    multivariate_forecasting()
    multi_step_forecasting()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. LSTMs capture temporal dependencies in time series")
    print("2. Use sliding windows for sequence-to-value prediction")
    print("3. Multivariate models handle multiple related features")
    print("4. Seq2seq models enable multi-step forecasting")
    print("5. Proper scaling and normalization are crucial")
    print("=" * 70)


if __name__ == "__main__":
    main()
