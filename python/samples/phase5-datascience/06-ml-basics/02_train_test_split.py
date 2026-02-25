"""
Train-Test Split and Cross-Validation
======================================
Splitting data for training and testing, cross-validation strategies.

Topics:
- Train-test split basics
- Random state and reproducibility
- Stratified splits for imbalanced data
- Cross-validation techniques
- K-fold and stratified K-fold

Run: python 02_train_test_split.py
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def main():
    print("=" * 60)
    print("Train-Test Split and Cross-Validation")
    print("=" * 60)

    # Load iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # 1. Basic train-test split
    print("\n1. Basic Train-Test Split")
    print("-" * 40)

    print(f"Total samples: {len(X)}")

    # Split into 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nAfter split:")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    print(f"\nTraining shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")

    print(f"\nTest shapes:")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")

    # 2. Random state for reproducibility
    print("\n2. Random State and Reproducibility")
    print("-" * 40)

    print("Without random_state (different each time):")
    for i in range(3):
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2)
        print(f"  Split {i+1}: First 5 training labels: {y_tr[:5]}")

    print("\nWith random_state=42 (same each time):")
    for i in range(3):
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"  Split {i+1}: First 5 training labels: {y_tr[:5]}")

    # 3. Class distribution in splits
    print("\n3. Class Distribution in Splits")
    print("-" * 40)

    # Regular split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Original distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")

    print("\nRegular split - Training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(y_train)*100:.1f}%)")

    print("\nRegular split - Test set:")
    unique, counts = np.unique(y_test, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(y_test)*100:.1f}%)")

    # 4. Stratified split
    print("\n4. Stratified Split")
    print("-" * 40)

    print("Stratified split ensures class proportions are preserved")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nStratified split - Training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(y_train)*100:.1f}%)")

    print("\nStratified split - Test set:")
    unique, counts = np.unique(y_test, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(y_test)*100:.1f}%)")

    # 5. Importance of test set
    print("\n5. Why Keep a Test Set?")
    print("-" * 40)

    print("Training set: Used to train the model")
    print("  - Model learns patterns from this data")
    print("  - Model will perform well on this set")

    print("\nTest set: Used to evaluate the model")
    print("  - Simulates unseen data")
    print("  - Provides unbiased performance estimate")
    print("  - Never use for training or tuning!")

    # Demonstrate overfitting risk
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)

    print(f"\nExample with KNN (k=1):")
    print(f"  Training accuracy: {train_score:.2%}")
    print(f"  Test accuracy: {test_score:.2%}")
    print(f"  Gap: {(train_score - test_score)*100:.1f}%")

    # 6. Cross-validation basics
    print("\n6. Cross-Validation Basics")
    print("-" * 40)

    print("Problem with single train-test split:")
    print("  - Results depend on which samples ended up in test set")
    print("  - Might get lucky or unlucky")

    print("\nSolution: K-Fold Cross-Validation")
    print("  - Split data into K folds")
    print("  - Train on K-1 folds, test on 1 fold")
    print("  - Repeat K times, each fold used as test once")
    print("  - Average the K scores")

    # 7. K-Fold cross-validation
    print("\n7. K-Fold Cross-Validation")
    print("-" * 40)

    # Use full dataset for CV
    X, y = iris.data, iris.target

    model = KNeighborsClassifier(n_neighbors=5)

    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)

    print("5-Fold Cross-Validation:")
    print(f"  Scores: {cv_scores}")
    print(f"  Mean: {cv_scores.mean():.4f}")
    print(f"  Std: {cv_scores.std():.4f}")
    print(f"  Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")

    # Compare with different K values
    print("\nComparing different K values:")
    for k in [3, 5, 10]:
        scores = cross_val_score(model, X, y, cv=k)
        print(f"  {k}-fold: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # 8. Manual K-Fold
    print("\n8. Manual K-Fold Implementation")
    print("-" * 40)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("Manual 5-fold cross-validation:")
    fold_num = 1
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train_fold = X[train_idx]
        X_test_fold = X[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_test_fold, y_test_fold)
        scores.append(score)

        print(f"  Fold {fold_num}: Train={len(train_idx)}, Test={len(test_idx)}, "
              f"Score={score:.4f}")
        fold_num += 1

    print(f"\nMean score: {np.mean(scores):.4f}")

    # 9. Stratified K-Fold
    print("\n9. Stratified K-Fold Cross-Validation")
    print("-" * 40)

    print("Regular K-Fold class distribution per fold:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold_num, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        y_test_fold = y[test_idx]
        unique, counts = np.unique(y_test_fold, return_counts=True)
        print(f"  Fold {fold_num}: {dict(zip(unique, counts))}")

    print("\nStratified K-Fold class distribution per fold:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        y_test_fold = y[test_idx]
        unique, counts = np.unique(y_test_fold, return_counts=True)
        print(f"  Fold {fold_num}: {dict(zip(unique, counts))}")

    # 10. Cross-validation for model selection
    print("\n10. Using Cross-Validation for Model Selection")
    print("-" * 40)

    print("Comparing different k values for KNN:")
    for k in [1, 3, 5, 7, 10]:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X, y, cv=5)
        print(f"  k={k:2d}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    print("\n" + "=" * 60)
    print("Summary - Data Splitting Best Practices:")
    print("  1. Always keep a separate test set")
    print("  2. Use stratified split for classification")
    print("  3. Set random_state for reproducibility")
    print("  4. Use cross-validation for model evaluation")
    print("  5. Typical split: 80% train, 20% test")
    print("  6. For small datasets, use k-fold CV (k=5 or k=10)")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Compare stratified vs non-stratified on imbalanced data")
    print("2. Implement leave-one-out cross-validation manually")
    print("3. Compare 5-fold vs 10-fold CV on different models")
    print("4. Create train/validation/test split (60/20/20)")
    print("=" * 60)

if __name__ == "__main__":
    main()
