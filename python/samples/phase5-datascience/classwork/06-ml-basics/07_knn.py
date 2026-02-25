"""
K-Nearest Neighbors (KNN)
=========================
Instance-based learning for classification and regression.

Topics:
- How KNN works
- KNN for classification
- KNN for regression
- Choosing k value
- Distance metrics
- Effect of feature scaling
- Computational considerations

Run: python 07_knn.py
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import datasets

def main():
    print("=" * 60)
    print("K-Nearest Neighbors (KNN)")
    print("=" * 60)

    # 1. How KNN works
    print("\n1. How K-Nearest Neighbors Works")
    print("-" * 40)

    print("KNN is a 'lazy' learning algorithm:")
    print("  - No training phase (just stores data)")
    print("  - At prediction time:")
    print("    1. Find K closest training samples to new point")
    print("    2. Classification: Vote among K neighbors")
    print("    3. Regression: Average of K neighbors")

    print("\nExample (k=3):")
    print("  New point: [5.1, 3.5, 1.4, 0.2]")
    print("  Find 3 nearest neighbors")
    print("  If 2 are 'setosa' and 1 is 'versicolor'")
    print("  Predict: 'setosa' (majority vote)")

    print("\nDistance metric (default: Euclidean):")
    print("  d = sqrt((x1-x2)² + (y1-y2)² + ...)")

    # 2. KNN classifier
    print("\n2. KNN Classifier")
    print("-" * 40)

    # Load iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    print(f"Dataset: Iris")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(iris.target_names)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train KNN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    print("\nKNN Classifier trained (k=5)")
    print(f"  Training samples stored: {len(X_train)}")

    # 3. Making predictions
    print("\n3. Making Predictions")
    print("-" * 40)

    predictions = knn.predict(X_test)

    print("Sample predictions:")
    for i in range(min(6, len(X_test))):
        pred_name = iris.target_names[predictions[i]]
        actual_name = iris.target_names[y_test[i]]
        match = "✓" if predictions[i] == y_test[i] else "✗"
        print(f"  {match} Predicted: {pred_name:10s}, Actual: {actual_name:10s}")

    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)

    print(f"\nTraining accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # 4. Finding neighbors
    print("\n4. Finding Nearest Neighbors")
    print("-" * 40)

    # Get a test sample
    sample = X_test[0:1]
    print(f"Test sample: {sample[0]}")
    print(f"Actual class: {iris.target_names[y_test[0]]}")

    # Find k nearest neighbors
    distances, indices = knn.kneighbors(sample)

    print(f"\n5 Nearest neighbors:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        neighbor_class = iris.target_names[y_train[idx]]
        print(f"  {i+1}. Distance: {dist:.3f}, Class: {neighbor_class}")

    prediction = knn.predict(sample)
    print(f"\nPrediction: {iris.target_names[prediction[0]]}")

    # 5. Effect of k value
    print("\n5. Effect of k Value")
    print("-" * 40)

    print("Testing different k values:")
    print(f"  {'k':>3s} {'Train Acc':>12s} {'Test Acc':>12s}")

    for k in [1, 3, 5, 7, 10, 15, 20]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        train_acc = knn.score(X_train, y_train)
        test_acc = knn.score(X_test, y_test)

        print(f"  {k:>3d} {train_acc:>12.4f} {test_acc:>12.4f}")

    print("\nObservations:")
    print("  - k=1: Perfect training accuracy (memorization)")
    print("  - Small k: Low bias, high variance (overfitting)")
    print("  - Large k: High bias, low variance (underfitting)")
    print("  - Best k: Balance between bias and variance")
    print("  - Tip: Try odd k for binary classification (avoid ties)")

    # 6. Cross-validation to find best k
    print("\n6. Finding Best k with Cross-Validation")
    print("-" * 40)

    k_values = range(1, 31)
    cv_scores = []

    print("Testing k from 1 to 30 with 5-fold CV...")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5)
        cv_scores.append(scores.mean())

    best_k = k_values[np.argmax(cv_scores)]
    best_score = max(cv_scores)

    print(f"\nBest k: {best_k}")
    print(f"Best CV score: {best_score:.4f}")

    print("\nTop 5 k values:")
    sorted_indices = np.argsort(cv_scores)[::-1][:5]
    for idx in sorted_indices:
        k = k_values[idx]
        score = cv_scores[idx]
        print(f"  k={k:2d}: {score:.4f}")

    # 7. Distance metrics
    print("\n7. Distance Metrics")
    print("-" * 40)

    print("Common distance metrics:")
    print("  - euclidean: sqrt(Σ(xi-yi)²)  [default]")
    print("  - manhattan: Σ|xi-yi|")
    print("  - minkowski: (Σ|xi-yi|^p)^(1/p)")
    print("  - chebyshev: max(|xi-yi|)")

    metrics = ['euclidean', 'manhattan', 'chebyshev']
    print(f"\nComparing distance metrics (k=5):")
    print(f"  {'Metric':<12s} {'Test Acc':>10s}")

    for metric in metrics:
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
        knn.fit(X_train, y_train)
        test_acc = knn.score(X_test, y_test)
        print(f"  {metric:<12s} {test_acc:>10.4f}")

    # 8. Feature scaling importance
    print("\n8. Importance of Feature Scaling")
    print("-" * 40)

    # Create dataset with different scales
    print("Feature ranges (before scaling):")
    for i, name in enumerate(iris.feature_names):
        print(f"  {name}: [{X[:, i].min():.1f}, {X[:, i].max():.1f}]")

    # Without scaling
    knn_no_scale = KNeighborsClassifier(n_neighbors=5)
    knn_no_scale.fit(X_train, y_train)
    acc_no_scale = knn_no_scale.score(X_test, y_test)

    # With scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_scaled.fit(X_train_scaled, y_train)
    acc_scaled = knn_scaled.score(X_test_scaled, y_test)

    print("\nPerformance comparison:")
    print(f"  Without scaling: {acc_no_scale:.4f}")
    print(f"  With scaling:    {acc_scaled:.4f}")

    print("\nNote: Scaling is crucial when features have different units")
    print("  (e.g., age in years vs income in dollars)")

    # 9. KNN for regression
    print("\n9. KNN for Regression")
    print("-" * 40)

    # Load diabetes dataset
    diabetes = datasets.load_diabetes()
    X_reg = diabetes.data
    y_reg = diabetes.target

    print(f"Dataset: Diabetes")
    print(f"  Samples: {len(X_reg)}")
    print(f"  Features: {X_reg.shape[1]}")

    # Split and scale
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_reg_scaled = scaler.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler.transform(X_test_reg)

    # Train KNN regressor
    knn_reg = KNeighborsRegressor(n_neighbors=5)
    knn_reg.fit(X_train_reg_scaled, y_train_reg)

    print("\nKNN Regressor trained (k=5)")

    predictions_reg = knn_reg.predict(X_test_reg_scaled)

    print("\nSample predictions:")
    for i in range(5):
        error = abs(predictions_reg[i] - y_test_reg[i])
        print(f"  Predicted: {predictions_reg[i]:6.1f}, "
              f"Actual: {y_test_reg[i]:6.1f}, Error: {error:5.1f}")

    test_mse = mean_squared_error(y_test_reg, predictions_reg)
    test_rmse = np.sqrt(test_mse)

    print(f"\nTest RMSE: {test_rmse:.2f}")

    # 10. Optimal k for regression
    print("\n10. Finding Optimal k for Regression")
    print("-" * 40)

    print(f"  {'k':>3s} {'Test RMSE':>12s}")
    for k in [1, 3, 5, 7, 10, 15, 20]:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train_reg_scaled, y_train_reg)
        pred = knn.predict(X_test_reg_scaled)
        rmse = np.sqrt(mean_squared_error(y_test_reg, pred))
        print(f"  {k:>3d} {rmse:>12.2f}")

    # 11. Weighted KNN
    print("\n11. Weighted KNN")
    print("-" * 40)

    print("Uniform weights: All neighbors vote equally")
    print("Distance weights: Closer neighbors have more influence")
    print("  weight = 1 / distance")

    print(f"\nComparison (k=10):")
    print(f"  {'Weights':<12s} {'Test Acc':>10s}")

    for weights in ['uniform', 'distance']:
        knn = KNeighborsClassifier(n_neighbors=10, weights=weights)
        knn.fit(X_train, y_train)
        test_acc = knn.score(X_test, y_test)
        print(f"  {weights:<12s} {test_acc:>10.4f}")

    print("\nDistance weights often work better with larger k")

    # 12. Computational considerations
    print("\n12. Computational Considerations")
    print("-" * 40)

    print("Training:")
    print("  - Very fast (just stores data)")
    print("  - O(1) complexity")

    print("\nPrediction:")
    print("  - Slow (compute distance to all training points)")
    print("  - O(n * d) where n=samples, d=features")
    print("  - Gets slower as dataset grows")

    print("\nMemory:")
    print("  - Must store all training data")
    print("  - Can be large for big datasets")

    print("\nSpeed improvements:")
    print("  - Use algorithm='ball_tree' or 'kd_tree'")
    print("  - Reduce features (dimensionality reduction)")
    print("  - Sample training data (if very large)")

    print("\n" + "=" * 60)
    print("Summary - K-Nearest Neighbors:")
    print("  Advantages:")
    print("    - Simple and intuitive")
    print("    - No training phase")
    print("    - Works for classification and regression")
    print("    - No assumptions about data distribution")
    print("  Disadvantages:")
    print("    - Slow prediction (especially with large data)")
    print("    - Memory intensive")
    print("    - Sensitive to feature scaling")
    print("    - Suffers in high dimensions (curse of dimensionality)")
    print("  Best practices:")
    print("    - Always scale features")
    print("    - Use cross-validation to find best k")
    print("    - Start with k=5, adjust based on data size")
    print("    - Consider distance weights for larger k")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Plot CV score vs k to visualize optimal k")
    print("2. Compare performance with and without outliers")
    print("3. Test on high-dimensional data (curse of dimensionality)")
    print("4. Implement weighted voting manually")
    print("=" * 60)

if __name__ == "__main__":
    main()
