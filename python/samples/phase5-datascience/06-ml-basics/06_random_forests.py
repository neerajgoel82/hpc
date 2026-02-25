"""
Random Forests
==============
Ensemble learning with random forest classifiers and regressors.

Topics:
- Ensemble methods (bagging)
- Random forest classifier
- Random forest regressor
- Feature importance
- Out-of-bag error
- Comparison with single decision tree

Run: python 06_random_forests.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import datasets

def main():
    print("=" * 60)
    print("Random Forests")
    print("=" * 60)

    # 1. Understanding ensemble methods
    print("\n1. What are Ensemble Methods?")
    print("-" * 40)

    print("Ensemble: Combine multiple models for better predictions")

    print("\nKey idea:")
    print("  - Train many models (weak learners)")
    print("  - Combine their predictions")
    print("  - Ensemble often outperforms single model")

    print("\nRandom Forest = Ensemble of Decision Trees")
    print("  - Train many decision trees")
    print("  - Each tree sees random subset of data")
    print("  - Each split considers random subset of features")
    print("  - Combine by voting (classification) or averaging (regression)")

    print("\nBootstrap Aggregating (Bagging):")
    print("  1. Create bootstrap sample (random sample with replacement)")
    print("  2. Train tree on bootstrap sample")
    print("  3. Repeat N times")
    print("  4. Aggregate predictions")

    # 2. Random forest classifier
    print("\n2. Random Forest Classifier")
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

    # Train random forest
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_classifier.fit(X_train, y_train)

    print("\nRandom Forest trained")
    print(f"  Number of trees: {rf_classifier.n_estimators}")
    print(f"  Number of features: {rf_classifier.n_features_in_}")

    # 3. Making predictions
    print("\n3. Making Predictions")
    print("-" * 40)

    predictions = rf_classifier.predict(X_test)

    print("Sample predictions:")
    for i in range(min(6, len(X_test))):
        pred_name = iris.target_names[predictions[i]]
        actual_name = iris.target_names[y_test[i]]
        match = "✓" if predictions[i] == y_test[i] else "✗"
        print(f"  {match} Predicted: {pred_name:10s}, Actual: {actual_name:10s}")

    train_accuracy = rf_classifier.score(X_train, y_train)
    test_accuracy = rf_classifier.score(X_test, y_test)

    print(f"\nTraining accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # 4. Comparing with single decision tree
    print("\n4. Random Forest vs Single Decision Tree")
    print("-" * 40)

    # Train single decision tree
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)

    dt_train_acc = dt_classifier.score(X_train, y_train)
    dt_test_acc = dt_classifier.score(X_test, y_test)

    print(f"{'Model':<20s} {'Train Acc':>12s} {'Test Acc':>12s}")
    print(f"{'Decision Tree':<20s} {dt_train_acc:>12.4f} {dt_test_acc:>12.4f}")
    print(f"{'Random Forest':<20s} {train_accuracy:>12.4f} {test_accuracy:>12.4f}")

    print("\nObservations:")
    print("  - Random Forest typically has lower training accuracy")
    print("  - But better test accuracy (less overfitting)")
    print("  - More robust and generalizable")

    # 5. Feature importance
    print("\n5. Feature Importance")
    print("-" * 40)

    importances = rf_classifier.feature_importances_

    print("Feature importance (averaged over all trees):")
    feature_importance = sorted(
        zip(iris.feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    for name, importance in feature_importance:
        bar = "█" * int(importance * 50)
        print(f"  {name:20s}: {importance:.4f} {bar}")

    # Compare with single tree
    dt_importances = dt_classifier.feature_importances_

    print("\nComparison: Random Forest vs Decision Tree")
    print(f"  {'Feature':<20s} {'RF':>10s} {'DT':>10s}")
    for name, rf_imp, dt_imp in zip(iris.feature_names, importances, dt_importances):
        print(f"  {name:<20s} {rf_imp:>10.4f} {dt_imp:>10.4f}")

    # 6. Effect of number of trees
    print("\n6. Effect of Number of Trees")
    print("-" * 40)

    print("Testing different numbers of trees:")
    print(f"  {'n_trees':>8s} {'Train Acc':>12s} {'Test Acc':>12s}")

    for n_trees in [1, 10, 50, 100, 200]:
        rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rf.fit(X_train, y_train)

        train_acc = rf.score(X_train, y_train)
        test_acc = rf.score(X_test, y_test)

        print(f"  {n_trees:>8d} {train_acc:>12.4f} {test_acc:>12.4f}")

    print("\nNote: More trees = more stable but slower training")
    print("  - Typical values: 100-500 trees")
    print("  - Beyond certain point, improvement plateaus")

    # 7. Random forest regressor
    print("\n7. Random Forest Regressor")
    print("-" * 40)

    # Load diabetes dataset
    diabetes = datasets.load_diabetes()
    X_reg = diabetes.data
    y_reg = diabetes.target

    print(f"Dataset: Diabetes")
    print(f"  Samples: {len(X_reg)}")
    print(f"  Features: {X_reg.shape[1]}")

    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Train random forest regressor
    rf_regressor = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    rf_regressor.fit(X_train_reg, y_train_reg)

    print("\nRandom Forest Regressor trained")
    print(f"  Number of trees: {rf_regressor.n_estimators}")

    # 8. Regression predictions
    print("\n8. Regression Predictions")
    print("-" * 40)

    predictions_reg = rf_regressor.predict(X_test_reg)

    print("Sample predictions:")
    for i in range(5):
        error = abs(predictions_reg[i] - y_test_reg[i])
        print(f"  Predicted: {predictions_reg[i]:6.1f}, "
              f"Actual: {y_test_reg[i]:6.1f}, Error: {error:5.1f}")

    # Compare with single decision tree
    dt_regressor = DecisionTreeRegressor(random_state=42)
    dt_regressor.fit(X_train_reg, y_train_reg)

    rf_train_pred = rf_regressor.predict(X_train_reg)
    rf_test_pred = rf_regressor.predict(X_test_reg)
    dt_train_pred = dt_regressor.predict(X_train_reg)
    dt_test_pred = dt_regressor.predict(X_test_reg)

    rf_train_rmse = np.sqrt(mean_squared_error(y_train_reg, rf_train_pred))
    rf_test_rmse = np.sqrt(mean_squared_error(y_test_reg, rf_test_pred))
    dt_train_rmse = np.sqrt(mean_squared_error(y_train_reg, dt_train_pred))
    dt_test_rmse = np.sqrt(mean_squared_error(y_test_reg, dt_test_pred))

    print("\nComparison:")
    print(f"{'Model':<20s} {'Train RMSE':>12s} {'Test RMSE':>12s}")
    print(f"{'Decision Tree':<20s} {dt_train_rmse:>12.2f} {dt_test_rmse:>12.2f}")
    print(f"{'Random Forest':<20s} {rf_train_rmse:>12.2f} {rf_test_rmse:>12.2f}")

    # 9. Out-of-bag error
    print("\n9. Out-of-Bag (OOB) Error")
    print("-" * 40)

    print("OOB Error: Validation without separate test set")
    print("  - Each tree trained on bootstrap sample (~63% of data)")
    print("  - Remaining ~37% is 'out-of-bag' for that tree")
    print("  - Use OOB samples to estimate generalization error")

    rf_oob = RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        random_state=42
    )
    rf_oob.fit(X_train, y_train)

    print(f"\nOOB score: {rf_oob.oob_score_:.4f}")
    print(f"Test score: {rf_oob.score(X_test, y_test):.4f}")
    print("Note: OOB score approximates test score without using test set")

    # 10. Important parameters
    print("\n10. Important Random Forest Parameters")
    print("-" * 40)

    print("n_estimators:")
    print("  - Number of trees in forest")
    print("  - More trees = better but slower")
    print("  - Typical: 100-500")

    print("\nmax_depth:")
    print("  - Maximum depth of each tree")
    print("  - None = grow until pure leaves")
    print("  - Smaller = less overfitting")

    print("\nmax_features:")
    print("  - Number of features to consider per split")
    print("  - 'sqrt': sqrt(n_features) [default for classification]")
    print("  - 'log2': log2(n_features)")
    print("  - Adds diversity between trees")

    print("\nmin_samples_split:")
    print("  - Minimum samples required to split node")
    print("  - Default: 2")
    print("  - Larger = more conservative")

    print("\nbootstrap:")
    print("  - Whether to use bootstrap samples")
    print("  - Default: True")
    print("  - False = use whole dataset for each tree")

    # Demonstrate max_features effect
    print("\nEffect of max_features:")
    print(f"  {'max_features':>12s} {'Test Acc':>10s}")

    for max_feat in ['sqrt', 'log2', None]:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_features=max_feat,
            random_state=42
        )
        rf.fit(X_train, y_train)
        test_acc = rf.score(X_test, y_test)

        feat_str = str(max_feat) if max_feat else "None"
        print(f"  {feat_str:>12s} {test_acc:>10.4f}")

    # 11. Cross-validation
    print("\n11. Cross-Validation Score")
    print("-" * 40)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=5)

    print("5-Fold Cross-Validation:")
    print(f"  Scores: {cv_scores}")
    print(f"  Mean: {cv_scores.mean():.4f}")
    print(f"  Std: {cv_scores.std():.4f}")

    print("\n" + "=" * 60)
    print("Summary - Random Forests:")
    print("  Advantages:")
    print("    - Very good performance out of the box")
    print("    - Resistant to overfitting (vs single tree)")
    print("    - Provides feature importance")
    print("    - Handles non-linear relationships")
    print("    - Works well with default parameters")
    print("  Disadvantages:")
    print("    - Slower to train and predict than single tree")
    print("    - Less interpretable than single tree")
    print("    - Larger model size (memory)")
    print("  Best practices:")
    print("    - Start with 100 trees")
    print("    - Use OOB score for quick validation")
    print("    - Check feature importance to understand model")
    print("    - Consider max_depth to control complexity")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Find optimal n_estimators using cross-validation")
    print("2. Compare different max_features settings")
    print("3. Use feature importance to select top N features")
    print("4. Compare RF with gradient boosting (if time permits)")
    print("=" * 60)

if __name__ == "__main__":
    main()
