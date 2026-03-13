"""
Decision Trees
==============
Tree-based classification and regression models.

Topics:
- Decision tree classifiers
- Decision tree regressors
- Tree depth and overfitting
- Feature importance
- Tree visualization (text-based)
- Gini impurity and entropy

Run: python 05_decision_trees.py
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import datasets
from sklearn import tree

def main():
    print("=" * 60)
    print("Decision Trees")
    print("=" * 60)

    # 1. How decision trees work
    print("\n1. How Decision Trees Work")
    print("-" * 40)

    print("Decision trees make predictions by:")
    print("  1. Ask a question about a feature")
    print("  2. Split data based on answer (yes/no)")
    print("  3. Repeat at each branch until reaching a leaf")
    print("  4. Leaf contains the prediction")

    print("\nExample decision process:")
    print("  Is sepal length > 5.5?")
    print("    Yes -> Is petal width > 1.5?")
    print("      Yes -> Predict: Virginica")
    print("      No  -> Predict: Versicolor")
    print("    No  -> Predict: Setosa")

    # 2. Decision tree classifier
    print("\n2. Decision Tree Classifier")
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

    # Train decision tree
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)

    print("\nModel trained")
    print(f"  Tree depth: {dt_classifier.get_depth()}")
    print(f"  Number of leaves: {dt_classifier.get_n_leaves()}")
    print(f"  Number of features: {dt_classifier.n_features_in_}")

    # 3. Making predictions
    print("\n3. Making Predictions")
    print("-" * 40)

    predictions = dt_classifier.predict(X_test)

    print("Sample predictions:")
    for i in range(min(8, len(X_test))):
        pred_name = iris.target_names[predictions[i]]
        actual_name = iris.target_names[y_test[i]]
        match = "✓" if predictions[i] == y_test[i] else "✗"
        print(f"  {match} Predicted: {pred_name:10s}, Actual: {actual_name:10s}")

    train_accuracy = dt_classifier.score(X_train, y_train)
    test_accuracy = dt_classifier.score(X_test, y_test)

    print(f"\nTraining accuracy: {train_accuracy:.2%}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    # 4. Feature importance
    print("\n4. Feature Importance")
    print("-" * 40)

    importances = dt_classifier.feature_importances_

    print("Feature importance scores:")
    feature_importance = sorted(
        zip(iris.feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    for name, importance in feature_importance:
        bar = "█" * int(importance * 50)
        print(f"  {name:20s}: {importance:.4f} {bar}")

    print("\nInterpretation:")
    print("  - Higher score = more important for decisions")
    print("  - Scores sum to 1.0")
    print("  - Based on how much each feature reduces impurity")

    # 5. Tree depth and overfitting
    print("\n5. Tree Depth and Overfitting")
    print("-" * 40)

    print("Effect of max_depth on performance:")
    print(f"  {'Depth':>5s} {'Train Acc':>10s} {'Test Acc':>10s} {'Leaves':>8s}")

    for depth in [1, 2, 3, 5, 10, None]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)

        train_acc = dt.score(X_train, y_train)
        test_acc = dt.score(X_test, y_test)
        n_leaves = dt.get_n_leaves()

        depth_str = "None" if depth is None else str(depth)
        print(f"  {depth_str:>5s} {train_acc:>10.4f} {test_acc:>10.4f} {n_leaves:>8d}")

    print("\nObservations:")
    print("  - Shallow trees: High bias, low variance (underfitting)")
    print("  - Deep trees: Low bias, high variance (overfitting)")
    print("  - Best depth: Balances training and test performance")

    # 6. Text-based tree visualization
    print("\n6. Tree Structure (Small Tree)")
    print("-" * 40)

    # Train a small tree for visualization
    dt_small = DecisionTreeClassifier(max_depth=2, random_state=42)
    dt_small.fit(X_train, y_train)

    print(f"Tree with max_depth=2:")
    tree_text = tree.export_text(
        dt_small,
        feature_names=iris.feature_names,
        class_names=iris.target_names.tolist()
    )
    print(tree_text)

    # 7. Decision tree regressor
    print("\n7. Decision Tree Regressor")
    print("-" * 40)

    # Load diabetes dataset
    diabetes = datasets.load_diabetes()
    X_reg = diabetes.data
    y_reg = diabetes.target

    print(f"Dataset: Diabetes")
    print(f"  Samples: {len(X_reg)}")
    print(f"  Features: {X_reg.shape[1]}")
    print(f"  Target: Continuous (disease progression)")

    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Train regressor
    dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_regressor.fit(X_train_reg, y_train_reg)

    print("\nRegressor trained")
    print(f"  Tree depth: {dt_regressor.get_depth()}")
    print(f"  Number of leaves: {dt_regressor.get_n_leaves()}")

    # 8. Regression predictions
    print("\n8. Regression Predictions")
    print("-" * 40)

    predictions_reg = dt_regressor.predict(X_test_reg)

    print("Sample predictions:")
    for i in range(5):
        error = abs(predictions_reg[i] - y_test_reg[i])
        print(f"  Predicted: {predictions_reg[i]:6.1f}, "
              f"Actual: {y_test_reg[i]:6.1f}, Error: {error:5.1f}")

    train_mse = mean_squared_error(y_train_reg, dt_regressor.predict(X_train_reg))
    test_mse = mean_squared_error(y_test_reg, predictions_reg)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    print(f"\nTraining RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")

    # 9. Comparing with different depths
    print("\n9. Regressor Depth Comparison")
    print("-" * 40)

    print(f"  {'Depth':>5s} {'Train RMSE':>12s} {'Test RMSE':>12s}")
    for depth in [2, 3, 5, 10, None]:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt.fit(X_train_reg, y_train_reg)

        train_pred = dt.predict(X_train_reg)
        test_pred = dt.predict(X_test_reg)

        train_rmse = np.sqrt(mean_squared_error(y_train_reg, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_reg, test_pred))

        depth_str = "None" if depth is None else str(depth)
        print(f"  {depth_str:>5s} {train_rmse:>12.2f} {test_rmse:>12.2f}")

    # 10. Tree parameters
    print("\n10. Important Tree Parameters")
    print("-" * 40)

    print("Key parameters for controlling tree growth:")
    print("\nmax_depth:")
    print("  - Maximum depth of the tree")
    print("  - Smaller = simpler model, less overfitting")
    print("  - Default: None (grow until pure leaves)")

    print("\nmin_samples_split:")
    print("  - Minimum samples required to split a node")
    print("  - Default: 2")
    print("  - Larger = more conservative splitting")

    print("\nmin_samples_leaf:")
    print("  - Minimum samples required at a leaf")
    print("  - Default: 1")
    print("  - Larger = smoother predictions")

    print("\nmax_features:")
    print("  - Number of features to consider for best split")
    print("  - Can be int, float, 'sqrt', 'log2', None")
    print("  - Adds randomness, reduces overfitting")

    # Demonstrate min_samples_split
    print("\nEffect of min_samples_split:")
    print(f"  {'min_split':>10s} {'Test Acc':>10s} {'Leaves':>8s}")
    for min_split in [2, 5, 10, 20]:
        dt = DecisionTreeClassifier(
            min_samples_split=min_split,
            random_state=42
        )
        dt.fit(X_train, y_train)
        test_acc = dt.score(X_test, y_test)
        n_leaves = dt.get_n_leaves()
        print(f"  {min_split:>10d} {test_acc:>10.4f} {n_leaves:>8d}")

    print("\n" + "=" * 60)
    print("Summary - Decision Trees:")
    print("  Advantages:")
    print("    - Easy to understand and interpret")
    print("    - No feature scaling needed")
    print("    - Handles non-linear relationships")
    print("    - Works with numerical and categorical data")
    print("  Disadvantages:")
    print("    - Prone to overfitting (especially deep trees)")
    print("    - Can be unstable (small data changes -> different tree)")
    print("    - Biased toward features with many levels")
    print("  Best practices:")
    print("    - Limit tree depth (max_depth)")
    print("    - Require minimum samples per split/leaf")
    print("    - Use cross-validation to find best parameters")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Find optimal max_depth using cross-validation")
    print("2. Compare Gini vs Entropy criterion")
    print("3. Visualize feature importance with bar plot")
    print("4. Build tree on only 2 most important features")
    print("=" * 60)

if __name__ == "__main__":
    main()
