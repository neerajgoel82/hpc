"""
Introduction to Scikit-Learn
=============================
Overview of scikit-learn's API, workflow, and built-in datasets.

Topics:
- Scikit-learn workflow and API design
- Loading built-in datasets (iris, digits)
- Data structure and exploration
- Basic estimator interface
- Fit-predict pattern

Run: python 01_sklearn_intro.py
"""

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

def main():
    print("=" * 60)
    print("Introduction to Scikit-Learn")
    print("=" * 60)

    # 1. The scikit-learn API philosophy
    print("\n1. Scikit-Learn API Philosophy")
    print("-" * 40)

    print("Key principles:")
    print("  - Consistency: All objects share a uniform interface")
    print("  - Inspection: All parameters are accessible")
    print("  - Limited object hierarchy: Algorithms are objects")
    print("  - Composition: Many tasks are sequences of transformations")
    print("  - Sensible defaults: Simple to get started")

    print("\nBasic workflow:")
    print("  1. Load data")
    print("  2. Choose a model")
    print("  3. Fit the model to training data")
    print("  4. Predict on new data")
    print("  5. Evaluate performance")

    # 2. Loading built-in datasets - Iris
    print("\n2. Built-in Datasets - Iris")
    print("-" * 40)

    # Load the famous iris dataset
    iris = datasets.load_iris()

    print("Iris dataset structure:")
    print(f"  Type: {type(iris)}")
    print(f"  Keys: {iris.keys()}")

    print("\nData array (features):")
    print(f"  Shape: {iris.data.shape}")
    print(f"  Type: {type(iris.data)}")
    print(f"  First 5 samples:\n{iris.data[:5]}")

    print("\nTarget array (labels):")
    print(f"  Shape: {iris.target.shape}")
    print(f"  Unique values: {np.unique(iris.target)}")
    print(f"  First 10 targets: {iris.target[:10]}")

    print("\nFeature names:")
    print(f"  {iris.feature_names}")

    print("\nTarget names (classes):")
    print(f"  {iris.target_names}")

    print("\nDataset description (first 200 chars):")
    print(iris.DESCR[:200] + "...")

    # 3. Exploring iris data
    print("\n3. Exploring Iris Data")
    print("-" * 40)

    print("Dataset statistics:")
    print(f"  Total samples: {len(iris.data)}")
    print(f"  Features per sample: {iris.data.shape[1]}")
    print(f"  Classes: {len(iris.target_names)}")

    # Class distribution
    unique, counts = np.unique(iris.target, return_counts=True)
    print("\nClass distribution:")
    for class_id, count in zip(unique, counts):
        print(f"  {iris.target_names[class_id]}: {count} samples")

    # Feature statistics
    print("\nFeature statistics:")
    for i, name in enumerate(iris.feature_names):
        col = iris.data[:, i]
        print(f"  {name}:")
        print(f"    Range: [{col.min():.2f}, {col.max():.2f}]")
        print(f"    Mean: {col.mean():.2f}, Std: {col.std():.2f}")

    # 4. Loading digits dataset
    print("\n4. Built-in Datasets - Digits")
    print("-" * 40)

    digits = datasets.load_digits()

    print("Digits dataset (handwritten digits 0-9):")
    print(f"  Data shape: {digits.data.shape}")
    print(f"  Images shape: {digits.images.shape}")
    print(f"  Number of classes: {len(digits.target_names)}")

    print("\nFirst digit:")
    print(f"  Label: {digits.target[0]}")
    print(f"  Image (8x8 pixels):")
    print(digits.images[0])

    print("\nFlattened representation (first digit):")
    print(f"  Shape: {digits.data[0].shape}")
    print(f"  Values: {digits.data[0][:20]}... (showing first 20)")

    # 5. The estimator interface
    print("\n5. The Estimator Interface")
    print("-" * 40)

    print("All estimators implement:")
    print("  .fit(X, y)      - Train the model")
    print("  .predict(X)     - Make predictions")
    print("  .score(X, y)    - Evaluate performance")
    print("  .get_params()   - Get parameters")
    print("  .set_params()   - Set parameters")

    # Simple example with KNN
    print("\nExample with K-Nearest Neighbors:")

    # Create a simple classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    print(f"  Model created: {knn}")

    # Use only first 100 samples for quick demo
    X_train = iris.data[:100]
    y_train = iris.target[:100]
    X_test = iris.data[100:110]
    y_test = iris.target[100:110]

    # Fit the model
    knn.fit(X_train, y_train)
    print(f"  Model trained on {len(X_train)} samples")

    # Make predictions
    predictions = knn.predict(X_test)
    print(f"  Predictions: {predictions}")
    print(f"  Actual:      {y_test}")

    # Score the model
    accuracy = knn.score(X_test, y_test)
    print(f"  Accuracy: {accuracy:.2%}")

    # 6. Supervised vs Unsupervised learning
    print("\n6. Supervised vs Unsupervised Learning")
    print("-" * 40)

    print("Supervised Learning:")
    print("  - Has labeled training data (X, y)")
    print("  - Goal: Learn mapping from X to y")
    print("  - Examples: Classification, Regression")
    print("  - Uses: fit(X, y) and predict(X)")

    print("\nUnsupervised Learning:")
    print("  - Only has input data (X)")
    print("  - Goal: Find patterns/structure in X")
    print("  - Examples: Clustering, Dimensionality Reduction")
    print("  - Uses: fit(X) and transform(X) or predict(X)")

    # 7. Regression example
    print("\n7. Quick Regression Example")
    print("-" * 40)

    # Create simple synthetic data
    np.random.seed(42)
    X_reg = np.array([[1], [2], [3], [4], [5]])
    y_reg = np.array([2, 4, 5, 4, 5])

    print("Training data:")
    for x, y in zip(X_reg, y_reg):
        print(f"  X={x[0]}, y={y}")

    # Fit linear regression
    lr = LinearRegression()
    lr.fit(X_reg, y_reg)

    print("\nModel learned:")
    print(f"  Coefficient: {lr.coef_[0]:.2f}")
    print(f"  Intercept: {lr.intercept_:.2f}")
    print(f"  Equation: y = {lr.coef_[0]:.2f}*x + {lr.intercept_:.2f}")

    # Make predictions
    X_new = np.array([[6], [7]])
    predictions = lr.predict(X_new)

    print("\nPredictions on new data:")
    for x, pred in zip(X_new, predictions):
        print(f"  X={x[0]} -> y={pred:.2f}")

    print("\n" + "=" * 60)
    print("Summary - Scikit-Learn Workflow:")
    print("  1. Import: from sklearn.xxx import Model")
    print("  2. Create: model = Model(parameters)")
    print("  3. Train:  model.fit(X_train, y_train)")
    print("  4. Predict: predictions = model.predict(X_test)")
    print("  5. Evaluate: score = model.score(X_test, y_test)")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Load the wine dataset and explore its structure")
    print("2. Train a KNN classifier on digits dataset")
    print("3. Compare predictions with different n_neighbors values")
    print("4. Calculate mean and range for each feature in iris")
    print("=" * 60)

if __name__ == "__main__":
    main()
