"""
Model Ensembles
Combine multiple models for better predictions

This module demonstrates:
- Voting ensembles (hard and soft voting)
- Averaging predictions
- Stacking models
- Weighted ensembles
- Ensemble training strategies

Run: python 02_model_ensembles.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SimpleClassifier(nn.Module):
    """Simple classifier for ensemble demonstration."""

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class VotingEnsemble(nn.Module):
    """Voting ensemble combining multiple models."""

    def __init__(self, models, voting="soft"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.voting = voting  # 'soft' or 'hard'

    def forward(self, x):
        predictions = [model(x) for model in self.models]

        if self.voting == "soft":
            # Average probabilities
            probs = [F.softmax(pred, dim=1) for pred in predictions]
            avg_probs = torch.stack(probs).mean(dim=0)
            return torch.log(avg_probs + 1e-8)  # Convert back to log-probs
        else:
            # Hard voting: majority class
            hard_preds = [pred.argmax(dim=1) for pred in predictions]
            stacked = torch.stack(hard_preds)
            # Mode (most common class)
            mode_pred = torch.mode(stacked, dim=0)[0]
            return mode_pred


class WeightedEnsemble(nn.Module):
    """Weighted ensemble with learnable weights."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        # Learnable weights for each model
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))

    def forward(self, x):
        predictions = [model(x) for model in self.models]
        probs = [F.softmax(pred, dim=1) for pred in predictions]

        # Normalize weights with softmax
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted average
        weighted_probs = sum(w * p for w, p in zip(norm_weights, probs))
        return torch.log(weighted_probs + 1e-8)


class StackingEnsemble(nn.Module):
    """Stacking ensemble with meta-learner."""

    def __init__(self, base_models, meta_model):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model

    def forward(self, x):
        # Get predictions from base models
        base_predictions = [model(x) for model in self.base_models]

        # Concatenate base predictions as features for meta-model
        meta_features = torch.cat(base_predictions, dim=1)

        # Meta-model makes final prediction
        return self.meta_model(meta_features)


def create_sample_data():
    """Create sample data for demonstration."""
    torch.manual_seed(42)
    X = torch.randn(1000, 20)
    y = torch.randint(0, 3, (1000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader


def train_single_model(model, train_loader, epochs=5):
    """Train a single model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def demonstrate_voting_ensemble():
    """Demonstrate voting ensemble."""
    print("1. Voting Ensemble")
    print("-" * 60)

    # Create and train multiple models
    train_loader = create_sample_data()

    print("Training 3 individual models...")
    models = []
    for i in range(3):
        model = SimpleClassifier(20, 64, 3, dropout=0.3)
        model = train_single_model(model, train_loader, epochs=5)
        models.append(model)
        print(f"Model {i+1} trained")

    # Soft voting ensemble
    soft_ensemble = VotingEnsemble(models, voting="soft")

    # Test ensemble
    X_test = torch.randn(10, 20)
    with torch.no_grad():
        individual_preds = [model(X_test).argmax(dim=1) for model in models]
        ensemble_pred = soft_ensemble(X_test).argmax(dim=1)

    print("\nSample predictions:")
    for i in range(3):
        print(f"Model {i+1}: {individual_preds[i][:5].tolist()}")
    print(f"Ensemble: {ensemble_pred[:5].tolist()}")
    print()


def demonstrate_weighted_ensemble():
    """Demonstrate weighted ensemble."""
    print("2. Weighted Ensemble")
    print("-" * 60)

    # Create models
    train_loader = create_sample_data()
    models = [
        train_single_model(SimpleClassifier(20, 64, 3), train_loader, epochs=5)
        for _ in range(3)
    ]

    ensemble = WeightedEnsemble(models)

    print(f"Initial weights: {F.softmax(ensemble.weights, dim=0).detach()}")

    # Train ensemble weights
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    ensemble.train()
    for epoch in range(10):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = ensemble(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    print(f"Learned weights: {F.softmax(ensemble.weights, dim=0).detach()}")
    print()


def demonstrate_stacking_ensemble():
    """Demonstrate stacking ensemble."""
    print("3. Stacking Ensemble")
    print("-" * 60)

    # Create base models
    train_loader = create_sample_data()
    base_models = [
        train_single_model(SimpleClassifier(20, 64, 3), train_loader, epochs=5)
        for _ in range(3)
    ]

    # Meta-model takes concatenated predictions (3 models * 3 classes = 9 features)
    meta_model = nn.Sequential(nn.Linear(9, 16), nn.ReLU(), nn.Linear(16, 3))

    ensemble = StackingEnsemble(base_models, meta_model)

    print("Base models trained")
    print("Training meta-model...")

    # Train meta-model
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    ensemble.train()
    for epoch in range(10):
        for X_batch, y_batch in train_loader:
            # Freeze base models
            for model in base_models:
                model.eval()

            optimizer.zero_grad()
            outputs = ensemble(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    print("Meta-model trained")
    print(f"Parameters in meta-model: {sum(p.numel() for p in meta_model.parameters())}")
    print()


def demonstrate_ensemble_diversity():
    """Demonstrate importance of ensemble diversity."""
    print("4. Ensemble Diversity")
    print("-" * 60)

    train_loader = create_sample_data()

    # Create diverse models with different architectures
    models = [
        SimpleClassifier(20, 64, 3, dropout=0.2),
        SimpleClassifier(20, 128, 3, dropout=0.4),
        SimpleClassifier(20, 32, 3, dropout=0.3),
    ]

    for i, model in enumerate(models):
        train_single_model(model, train_loader, epochs=5)

    # Test diversity
    X_test = torch.randn(100, 20)
    with torch.no_grad():
        preds = [model(X_test).argmax(dim=1) for model in models]

    # Calculate agreement between models
    agreements = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            agreement = (preds[i] == preds[j]).float().mean().item()
            agreements.append(agreement)
            print(f"Model {i+1} vs Model {j+1} agreement: {agreement:.2%}")

    print(f"Average agreement: {sum(agreements) / len(agreements):.2%}")
    print("(Lower agreement often means better ensemble diversity)")
    print()


def main():
    print("=" * 70)
    print("MODEL ENSEMBLES")
    print("=" * 70)
    print()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    demonstrate_voting_ensemble()
    demonstrate_weighted_ensemble()
    demonstrate_stacking_ensemble()
    demonstrate_ensemble_diversity()

    print("=" * 70)


if __name__ == "__main__":
    main()
