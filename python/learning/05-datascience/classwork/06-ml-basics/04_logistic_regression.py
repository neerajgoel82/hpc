"""
Logistic Regression
===================
Binary and multiclass classification with logistic regression.

Topics:
- Binary classification (two classes)
- Multiclass classification (more than two)
- Probability predictions
- Decision boundaries
- Model coefficients
- Class balancing

Run: python 04_logistic_regression.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import datasets

def main():
    print("=" * 60)
    print("Logistic Regression")
    print("=" * 60)

    # 1. Logistic vs Linear Regression
    print("\n1. Logistic vs Linear Regression")
    print("-" * 40)

    print("Linear Regression:")
    print("  - Predicts continuous values")
    print("  - Output: any real number")
    print("  - Example: Predict house price, temperature")

    print("\nLogistic Regression:")
    print("  - Predicts class probabilities")
    print("  - Output: probability between 0 and 1")
    print("  - Example: Spam/not spam, disease/healthy")
    print("  - Uses sigmoid function: σ(x) = 1 / (1 + e^(-x))")

    # 2. Binary classification with iris (2 classes)
    print("\n2. Binary Classification")
    print("-" * 40)

    # Load iris, use only first 2 classes for binary classification
    iris = datasets.load_iris()
    X_binary = iris.data[iris.target != 2]  # Remove class 2
    y_binary = iris.target[iris.target != 2]  # Classes 0 and 1 only

    print(f"Dataset: Iris (binary)")
    print(f"  Total samples: {len(X_binary)}")
    print(f"  Features: {X_binary.shape[1]}")
    print(f"  Classes: {iris.target_names[0]} vs {iris.target_names[1]}")

    # Class distribution
    unique, counts = np.unique(y_binary, return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls} ({iris.target_names[cls]}): {cnt} samples")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y_binary, test_size=0.2, stratify=y_binary, random_state=42
    )

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")

    # 3. Training binary classifier
    print("\n3. Training Binary Classifier")
    print("-" * 40)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    print("Model trained")
    print(f"  Classes: {model.classes_}")
    print(f"  Intercept: {model.intercept_[0]:.4f}")
    print("\nCoefficients:")
    for name, coef in zip(iris.feature_names, model.coef_[0]):
        print(f"  {name}: {coef:7.4f}")

    # 4. Making predictions
    print("\n4. Making Predictions")
    print("-" * 40)

    predictions = model.predict(X_test)

    print("Predictions (first 10):")
    for i in range(min(10, len(X_test))):
        pred_name = iris.target_names[predictions[i]]
        actual_name = iris.target_names[y_test[i]]
        match = "✓" if predictions[i] == y_test[i] else "✗"
        print(f"  {match} Predicted: {pred_name:10s}, Actual: {actual_name:10s}")

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.2%}")

    # 5. Probability predictions
    print("\n5. Probability Predictions")
    print("-" * 40)

    probabilities = model.predict_proba(X_test)

    print("Probability predictions (first 5):")
    print(f"  {'Class 0':>12s} {'Class 1':>12s} {'Predicted':>12s} {'Actual':>10s}")
    for i in range(min(5, len(X_test))):
        prob0, prob1 = probabilities[i]
        pred = predictions[i]
        actual = y_test[i]
        print(f"  {prob0:12.4f} {prob1:12.4f} {pred:12d} {actual:10d}")

    print("\nNote: Prediction is class with highest probability")
    print("  Default threshold: 0.5 (50%)")

    # 6. Decision boundary and confidence
    print("\n6. Decision Confidence")
    print("-" * 40)

    max_probs = probabilities.max(axis=1)

    print("Prediction confidence (first 10):")
    for i in range(min(10, len(X_test))):
        confidence = max_probs[i]
        pred_name = iris.target_names[predictions[i]]
        status = "High" if confidence > 0.9 else "Medium" if confidence > 0.7 else "Low"
        print(f"  {pred_name:10s}: {confidence:.2%} ({status} confidence)")

    print(f"\nAverage confidence: {max_probs.mean():.2%}")
    print(f"Min confidence: {max_probs.min():.2%}")
    print(f"Max confidence: {max_probs.max():.2%}")

    # 7. Multiclass classification
    print("\n7. Multiclass Classification")
    print("-" * 40)

    # Use full iris dataset (3 classes)
    X = iris.data
    y = iris.target

    print(f"Dataset: Iris (multiclass)")
    print(f"  Total samples: {len(X)}")
    print(f"  Classes: {len(iris.target_names)}")
    for i, name in enumerate(iris.target_names):
        count = np.sum(y == i)
        print(f"    {i}. {name}: {count} samples")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train multiclass model
    model_multi = LogisticRegression(random_state=42, max_iter=200)
    model_multi.fit(X_train, y_train)

    print("\nModel trained (multiclass)")
    print(f"  Classes: {model_multi.classes_}")
    print(f"  Strategy: One-vs-Rest (OvR)")

    # 8. Multiclass predictions
    print("\n8. Multiclass Predictions")
    print("-" * 40)

    predictions = model_multi.predict(X_test)
    probabilities = model_multi.predict_proba(X_test)

    print("Sample predictions:")
    print(f"  {'Setosa Prob':>12s} {'Versicolor':>12s} {'Virginica':>12s} "
          f"{'Pred':>10s} {'Actual':>10s}")
    for i in range(min(5, len(X_test))):
        p0, p1, p2 = probabilities[i]
        pred_name = iris.target_names[predictions[i]]
        actual_name = iris.target_names[y_test[i]]
        print(f"  {p0:12.4f} {p1:12.4f} {p2:12.4f} "
              f"{pred_name:>10s} {actual_name:>10s}")

    # 9. Evaluation metrics
    print("\n9. Detailed Evaluation")
    print("-" * 40)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Overall Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        target_names=iris.target_names,
        digits=3
    ))

    # 10. Model parameters and tuning
    print("\n10. Model Parameters")
    print("-" * 40)

    print("Common LogisticRegression parameters:")
    print("  - C: Regularization strength (default=1.0)")
    print("    Smaller C = stronger regularization")
    print("  - penalty: 'l1', 'l2' (default='l2')")
    print("  - solver: 'lbfgs', 'liblinear', 'saga', etc.")
    print("  - max_iter: Maximum iterations (default=100)")

    # Compare different C values
    print("\nEffect of regularization (C parameter):")
    for c_val in [0.1, 1.0, 10.0]:
        model = LogisticRegression(C=c_val, random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"  C={c_val:4.1f}: Accuracy = {score:.4f}")

    # 11. Handling imbalanced data
    print("\n11. Handling Imbalanced Classes")
    print("-" * 40)

    # Create imbalanced dataset
    imbalanced_idx = np.concatenate([
        np.where(y == 0)[0],  # All class 0
        np.where(y == 1)[0][:10],  # Only 10 from class 1
        np.where(y == 2)[0][:5]   # Only 5 from class 2
    ])

    X_imb = X[imbalanced_idx]
    y_imb = y[imbalanced_idx]

    print("Imbalanced dataset:")
    unique, counts = np.unique(y_imb, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples")

    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
        X_imb, y_imb, test_size=0.2, stratify=y_imb, random_state=42
    )

    # Without balancing
    model_no_balance = LogisticRegression(random_state=42, max_iter=200)
    model_no_balance.fit(X_train_imb, y_train_imb)
    score_no_balance = model_no_balance.score(X_test_imb, y_test_imb)

    # With class_weight='balanced'
    model_balanced = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=200
    )
    model_balanced.fit(X_train_imb, y_train_imb)
    score_balanced = model_balanced.score(X_test_imb, y_test_imb)

    print("\nResults:")
    print(f"  Without balancing: {score_no_balance:.4f}")
    print(f"  With balancing:    {score_balanced:.4f}")

    print("\n" + "=" * 60)
    print("Summary - Logistic Regression:")
    print("  - Binary: 2 classes (0/1, yes/no, spam/ham)")
    print("  - Multiclass: 3+ classes (one-vs-rest strategy)")
    print("  - Output: Probabilities (0 to 1)")
    print("  - predict(): Returns class with highest probability")
    print("  - predict_proba(): Returns all class probabilities")
    print("  - C parameter: Controls regularization strength")
    print("  - class_weight='balanced': Handles imbalanced data")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Train on breast cancer dataset (binary classification)")
    print("2. Experiment with different C values")
    print("3. Use custom probability threshold (not 0.5)")
    print("4. Compare L1 vs L2 regularization")
    print("=" * 60)

if __name__ == "__main__":
    main()
