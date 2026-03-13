"""
Project: Time Series Forecasting
================================
Predict future values in time series data using RNNs.

Dataset: Synthetic multivariate time series (temperature, humidity, etc.)
Goals:
- Generate realistic time series data with trends and seasonality
- Prepare windowed sequences for training
- Build LSTM/GRU forecasting models
- Train with multi-step ahead prediction
- Evaluate forecast accuracy (MAE, RMSE, MAPE)
- Visualize predictions vs actual values
- Compare different architectures

Skills: LSTMs, Sequence modeling, Time series, Feature scaling
Run: python project_time_series.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


def generate_time_series(n_samples=2000, n_features=3):
    """Generate synthetic multivariate time series with trends and seasonality."""
    print("Generating time series data...")

    time_steps = np.arange(n_samples)

    # Feature 1: Temperature (with daily and weekly seasonality + trend)
    daily_cycle = 10 * np.sin(2 * np.pi * time_steps / 24)
    weekly_cycle = 5 * np.sin(2 * np.pi * time_steps / (24 * 7))
    trend = 0.01 * time_steps
    noise = np.random.randn(n_samples) * 2
    temperature = 20 + daily_cycle + weekly_cycle + trend + noise

    # Feature 2: Humidity (inverse correlation with temperature)
    humidity = 70 - 0.5 * daily_cycle - 0.3 * weekly_cycle + np.random.randn(n_samples) * 3
    humidity = np.clip(humidity, 30, 95)

    # Feature 3: Pressure (slow changes)
    pressure = 1013 + 5 * np.sin(2 * np.pi * time_steps / (24 * 14)) + np.random.randn(n_samples) * 2

    # Combine features
    data = np.stack([temperature, humidity, pressure], axis=1)

    print(f"Generated {n_samples} time steps with {n_features} features")
    print(f"Temperature range: [{temperature.min():.2f}, {temperature.max():.2f}]")
    print(f"Humidity range: [{humidity.min():.2f}, {humidity.max():.2f}]")
    print(f"Pressure range: [{pressure.min():.2f}, {pressure.max():.2f}]")

    return data


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting with sliding windows."""

    def __init__(self, data, seq_len=24, pred_len=6):
        """
        Args:
            data: numpy array of shape (n_samples, n_features)
            seq_len: length of input sequence
            pred_len: length of prediction horizon
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx:idx + self.seq_len]
        # Target sequence (predict next pred_len steps)
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return x, y


class TimeSeriesLSTM(nn.Module):
    """LSTM model for time series forecasting."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 output_dim=3, pred_len=6, dropout=0.2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.output_dim = output_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim * pred_len)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_dim]

        # Predict multiple steps ahead
        output = self.fc(last_hidden)  # [batch_size, output_dim * pred_len]

        # Reshape to [batch_size, pred_len, output_dim]
        output = output.view(-1, self.pred_len, self.output_dim)

        return output


class TimeSeriesGRU(nn.Module):
    """GRU model for time series forecasting."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 output_dim=3, pred_len=6, dropout=0.2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.output_dim = output_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim * pred_len)
        )

    def forward(self, x):
        gru_out, hidden = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        output = self.fc(last_hidden)
        output = output.view(-1, self.pred_len, self.output_dim)
        return output


def normalize_data(train_data, test_data):
    """Normalize data using training statistics."""
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    train_normalized = (train_data - mean) / (std + 1e-8)
    test_normalized = (test_data - mean) / (std + 1e-8)

    return train_normalized, test_normalized, mean, std


def denormalize_data(data, mean, std):
    """Denormalize data."""
    return data * std + mean


def calculate_metrics(predictions, targets):
    """Calculate forecasting metrics."""
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100

    return mae, rmse, mape


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()

    running_loss = 0.0

    for batch_idx, (sequences, targets) in enumerate(train_loader):
        sequences, targets = sequences.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 50 == 49:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {running_loss/(batch_idx+1):.4f}")

    epoch_loss = running_loss / len(train_loader)

    return epoch_loss


def evaluate_model(model, test_loader, criterion, device, mean, std):
    """Evaluate model on test set."""
    model.eval()

    test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)

            predictions = model(sequences)
            loss = criterion(predictions, targets)

            test_loss += loss.item()

            # Denormalize for metric calculation
            pred_denorm = denormalize_data(
                predictions.cpu().numpy(),
                mean, std
            )
            target_denorm = denormalize_data(
                targets.cpu().numpy(),
                mean, std
            )

            all_predictions.append(pred_denorm)
            all_targets.append(target_denorm)

    test_loss = test_loss / len(test_loader)

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    mae, rmse, mape = calculate_metrics(all_predictions, all_targets)

    return test_loss, mae, rmse, mape, all_predictions, all_targets


def train_model(model, train_loader, test_loader, num_epochs, device, mean, std, lr=0.001):
    """Complete training loop."""
    print("\n" + "=" * 60)
    print("TRAINING TIME SERIES MODEL")
    print("=" * 60)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    history = {
        'train_loss': [],
        'test_loss': [],
        'test_mae': [],
        'test_rmse': [],
        'test_mape': []
    }

    best_mae = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        print(f"\nEpoch [{epoch+1}/{num_epochs}] LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        test_loss, test_mae, test_rmse, test_mape, _, _ = evaluate_model(
            model, test_loader, criterion, device, mean, std
        )

        scheduler.step()

        # Save best model
        if test_mae < best_mae:
            best_mae = test_mae
            torch.save(model.state_dict(), 'best_timeseries.pth')
            print(f"  Saved best model (MAE: {best_mae:.4f})")

        # Record metrics
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_mae'].append(test_mae)
        history['test_rmse'].append(test_rmse)
        history['test_mape'].append(test_mape)

        epoch_time = time.time() - start_time

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}")
        print(f"  Test MAE:   {test_mae:.4f}")
        print(f"  Test RMSE:  {test_rmse:.4f}")
        print(f"  Test MAPE:  {test_mape:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

    return history, best_mae


def visualize_results(history, predictions, targets, data, feature_names):
    """Create comprehensive visualizations."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 10))

    # 1. Training curves - Loss
    ax1 = plt.subplot(2, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Test Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True)

    # 2. Test metrics over epochs
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['test_mae'], 'g-', label='MAE')
    ax2.plot(epochs, history['test_rmse'], 'b-', label='RMSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.set_title('Test Metrics (MAE & RMSE)', fontweight='bold')
    ax2.legend()
    ax2.grid(True)

    # 3. MAPE over epochs
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, history['test_mape'], 'r-')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAPE (%)')
    ax3.set_title('Test MAPE', fontweight='bold')
    ax3.grid(True)

    # 4-6. Predictions vs Actual for each feature
    sample_idx = np.random.randint(0, len(predictions) - 50)
    sample_preds = predictions[sample_idx:sample_idx + 50]
    sample_targets = targets[sample_idx:sample_idx + 50]

    for feat_idx, feat_name in enumerate(feature_names):
        ax = plt.subplot(2, 3, 4 + feat_idx)

        # Flatten predictions for this feature
        pred_values = sample_preds[:, :, feat_idx].flatten()
        target_values = sample_targets[:, :, feat_idx].flatten()

        time_steps = np.arange(len(pred_values))
        ax.plot(time_steps, target_values, 'b-', label='Actual', alpha=0.7)
        ax.plot(time_steps, pred_values, 'r--', label='Predicted', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel(feat_name)
        ax.set_title(f'{feat_name} Predictions', fontweight='bold')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    print("Visualizations created!")
    print("Close the plot window to continue...")
    plt.show()


def visualize_forecast(model, test_data, seq_len, pred_len, mean, std, device):
    """Visualize a single forecast example."""
    model.eval()

    # Select a random sequence
    idx = np.random.randint(0, len(test_data) - seq_len - pred_len)

    input_seq = test_data[idx:idx + seq_len]
    actual_future = test_data[idx + seq_len:idx + seq_len + pred_len]

    # Predict
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
        prediction = model(input_tensor)
        prediction = prediction.squeeze(0).cpu().numpy()

    # Denormalize
    input_seq_denorm = denormalize_data(input_seq, mean, std)
    prediction_denorm = denormalize_data(prediction, mean, std)
    actual_future_denorm = denormalize_data(actual_future, mean, std)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    feature_names = ['Temperature', 'Humidity', 'Pressure']

    for feat_idx, (ax, feat_name) in enumerate(zip(axes, feature_names)):
        # Time steps
        input_time = np.arange(seq_len)
        future_time = np.arange(seq_len, seq_len + pred_len)

        # Plot input sequence
        ax.plot(input_time, input_seq_denorm[:, feat_idx], 'b-',
                label='Input', linewidth=2)

        # Plot actual future
        ax.plot(future_time, actual_future_denorm[:, feat_idx], 'g-',
                label='Actual Future', linewidth=2)

        # Plot predicted future
        ax.plot(future_time, prediction_denorm[:, feat_idx], 'r--',
                label='Predicted Future', linewidth=2)

        ax.axvline(x=seq_len, color='gray', linestyle=':', label='Forecast Start')
        ax.set_xlabel('Time Step')
        ax.set_ylabel(feat_name)
        ax.set_title(f'{feat_name} Forecast', fontweight='bold')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    print("\nForecast visualization displayed")
    print("Close the plot window to continue...")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("TIME SERIES FORECASTING PROJECT")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    seq_len = 24  # Use 24 time steps to predict
    pred_len = 6  # Predict next 6 time steps
    batch_size = 32
    num_epochs = 40
    learning_rate = 0.001

    print(f"\nHyperparameters:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Prediction length: {pred_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")

    # Generate data
    data = generate_time_series(n_samples=2000, n_features=3)

    # Train/test split (80/20)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    print(f"\nTrain samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Normalize
    train_norm, test_norm, mean, std = normalize_data(train_data, test_data)

    # Create datasets
    train_dataset = TimeSeriesDataset(train_norm, seq_len=seq_len, pred_len=pred_len)
    test_dataset = TimeSeriesDataset(test_norm, seq_len=seq_len, pred_len=pred_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training windows: {len(train_dataset)}")
    print(f"Test windows: {len(test_dataset)}")

    # Create model
    print("\nInitializing LSTM model...")
    model = TimeSeriesLSTM(
        input_dim=3,
        hidden_dim=128,
        num_layers=2,
        output_dim=3,
        pred_len=pred_len,
        dropout=0.2
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train model
    history, best_mae = train_model(
        model, train_loader, test_loader, num_epochs, device, mean, std, learning_rate
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    criterion = nn.MSELoss()
    test_loss, test_mae, test_rmse, test_mape, predictions, targets = evaluate_model(
        model, test_loader, criterion, device, mean, std
    )

    print(f"\nFinal Test Metrics:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAPE: {test_mape:.2f}%")
    print(f"\nBest MAE: {best_mae:.4f}")

    # Visualizations
    feature_names = ['Temperature', 'Humidity', 'Pressure']
    visualize_results(history, predictions, targets, data, feature_names)
    visualize_forecast(model, test_norm, seq_len, pred_len, mean, std, device)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)
    print("\nKey Findings:")
    print(f"1. Achieved MAE of {best_mae:.4f} on multi-step forecasting")
    print(f"2. MAPE of {test_mape:.2f}% indicates good relative accuracy")
    print("3. LSTM effectively captures temporal dependencies")
    print("4. Multi-step ahead prediction is challenging but achievable")
    print("5. Model saved to 'best_timeseries.pth'")
    print("=" * 60)


if __name__ == "__main__":
    main()
