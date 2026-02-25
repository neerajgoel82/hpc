"""
Machine Learning Basics - Practice Exercises
============================================
Practice problems covering all ML basics concepts.

Exercises:
1. Breast Cancer Classification
2. Wine Quality Prediction
3. Model Comparison
4. Hyperparameter Tuning
5. Feature Selection
6. Cross-Validation Study
7. Imbalanced Classification
8. Ensemble Comparison
9. Regression Analysis
10. Full ML Pipeline

Run: python exercises.py
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score
)


def exercise_1_breast_cancer():
    """
    Exercise 1: Breast Cancer Classification
    Load breast cancer dataset, train logistic regression, evaluate performance
    """
    print("\n" + "=" * 60)
    print("Exercise 1: Breast Cancer Classification")
    print("=" * 60)

    # TODO: Load breast cancer dataset
    # TODO: Split into train/test (80/20)
    # TODO: Train logistic regression
    # TODO: Calculate accuracy, precision, recall, F1
    # TODO: Print classification report

    print("\nTODO: Implement this exercise")
    print("Steps:")
    print("  1. Load datasets.load_breast_cancer()")
    print("  2. Split data (test_size=0.2, random_state=42)")
    print("  3. Train LogisticRegression")
    print("  4. Calculate and print all metrics")
    print("  5. Display confusion matrix")


def exercise_2_wine_quality():
    """
    Exercise 2: Wine Quality Prediction
    Use wine dataset for multiclass classification
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Wine Quality Classification")
    print("=" * 60)

    # TODO: Load wine dataset
    # TODO: Use stratified train/test split
    # TODO: Train multiple models (Logistic, Tree, KNN)
    # TODO: Compare accuracy of all models
    # TODO: Which model performs best?

    print("\nTODO: Implement this exercise")
    print("Steps:")
    print("  1. Load datasets.load_wine()")
    print("  2. Stratified split (test_size=0.3)")
    print("  3. Train 3 different classifiers")
    print("  4. Compare test accuracy")
    print("  5. Identify best model")


def exercise_3_model_comparison():
    """
    Exercise 3: Comprehensive Model Comparison
    Compare 5 models on iris dataset using cross-validation
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Model Comparison with Cross-Validation")
    print("=" * 60)

    # TODO: Load iris dataset
    # TODO: Define 5 different models
    # TODO: Use 5-fold cross-validation on each
    # TODO: Print mean CV score and std for each
    # TODO: Rank models by performance

    print("\nTODO: Implement this exercise")
    print("Steps:")
    print("  1. Load iris dataset")
    print("  2. Create list of 5 models")
    print("  3. Run cross_val_score(cv=5) on each")
    print("  4. Print mean ± std for each model")
    print("  5. Determine winner")


def exercise_4_hyperparameter_tuning():
    """
    Exercise 4: Hyperparameter Tuning for KNN
    Find optimal k value for KNN using cross-validation
    """
    print("\n" + "=" * 60)
    print("Exercise 4: Hyperparameter Tuning (KNN)")
    print("=" * 60)

    # TODO: Load digits dataset
    # TODO: Test k values from 1 to 30
    # TODO: Use cross-validation to evaluate each k
    # TODO: Plot CV score vs k
    # TODO: Report optimal k

    print("\nTODO: Implement this exercise")
    print("Steps:")
    print("  1. Load datasets.load_digits()")
    print("  2. Loop through k=1 to 30")
    print("  3. Cross-validate each k")
    print("  4. Find k with best CV score")
    print("  5. Test final model on held-out test set")


def exercise_5_feature_selection():
    """
    Exercise 5: Feature Importance and Selection
    Use Random Forest to identify important features
    """
    print("\n" + "=" * 60)
    print("Exercise 5: Feature Importance Analysis")
    print("=" * 60)

    # TODO: Load diabetes dataset (regression)
    # TODO: Train Random Forest Regressor
    # TODO: Extract feature importances
    # TODO: Sort features by importance
    # TODO: Train model with only top 5 features
    # TODO: Compare performance: all features vs top 5

    print("\nTODO: Implement this exercise")
    print("Steps:")
    print("  1. Load diabetes dataset")
    print("  2. Train RandomForestRegressor")
    print("  3. Get feature_importances_")
    print("  4. Select top 5 features")
    print("  5. Compare RMSE: all vs top 5")


def exercise_6_cross_validation_study():
    """
    Exercise 6: Cross-Validation Strategy Study
    Compare different CV strategies
    """
    print("\n" + "=" * 60)
    print("Exercise 6: Cross-Validation Strategy Comparison")
    print("=" * 60)

    # TODO: Load iris dataset
    # TODO: Try different k values for k-fold (3, 5, 10)
    # TODO: Compare stratified vs non-stratified
    # TODO: Compare regular vs stratified split
    # TODO: Which strategy is most stable?

    print("\nTODO: Implement this exercise")
    print("Steps:")
    print("  1. Load iris dataset")
    print("  2. Test k-fold with k=3,5,10")
    print("  3. Test StratifiedKFold")
    print("  4. Compare mean and std of scores")
    print("  5. Recommend best strategy")


def exercise_7_imbalanced_classification():
    """
    Exercise 7: Handling Imbalanced Data
    Create imbalanced dataset and handle it properly
    """
    print("\n" + "=" * 60)
    print("Exercise 7: Imbalanced Classification")
    print("=" * 60)

    # TODO: Load breast cancer, create 90-10 imbalance
    # TODO: Train model without balancing
    # TODO: Train model with class_weight='balanced'
    # TODO: Compare precision, recall, F1
    # TODO: Which approach is better?

    print("\nTODO: Implement this exercise")
    print("Steps:")
    print("  1. Create imbalanced dataset (90% class 0)")
    print("  2. Train LogisticRegression (default)")
    print("  3. Train with class_weight='balanced'")
    print("  4. Compare recall for minority class")
    print("  5. Explain which is better and why")


def exercise_8_ensemble_comparison():
    """
    Exercise 8: Single Model vs Ensemble
    Compare decision tree with random forest
    """
    print("\n" + "=" * 60)
    print("Exercise 8: Single Tree vs Random Forest")
    print("=" * 60)

    # TODO: Load wine dataset
    # TODO: Train DecisionTreeClassifier
    # TODO: Train RandomForestClassifier
    # TODO: Compare training vs test accuracy
    # TODO: Which shows more overfitting?

    print("\nTODO: Implement this exercise")
    print("Steps:")
    print("  1. Load wine dataset")
    print("  2. Train DecisionTreeClassifier")
    print("  3. Train RandomForestClassifier")
    print("  4. Calculate train and test accuracy for both")
    print("  5. Compare overfitting (train-test gap)")


def exercise_9_regression_analysis():
    """
    Exercise 9: Regression Model Comparison
    Compare multiple regression models
    """
    print("\n" + "=" * 60)
    print("Exercise 9: Regression Model Comparison")
    print("=" * 60)

    # TODO: Load diabetes dataset
    # TODO: Train Linear Regression
    # TODO: Train Decision Tree Regressor
    # TODO: Train Random Forest Regressor
    # TODO: Compare RMSE and R² for all three

    print("\nTODO: Implement this exercise")
    print("Steps:")
    print("  1. Load diabetes dataset")
    print("  2. Train 3 regressors")
    print("  3. Calculate RMSE for each")
    print("  4. Calculate R² for each")
    print("  5. Rank models by performance")


def exercise_10_full_pipeline():
    """
    Exercise 10: Complete ML Pipeline
    Build a full pipeline from raw data to evaluation
    """
    print("\n" + "=" * 60)
    print("Exercise 10: Full ML Pipeline")
    print("=" * 60)

    # TODO: Load breast cancer dataset
    # TODO: Exploratory data analysis (shape, stats)
    # TODO: Train-test split with stratification
    # TODO: Feature scaling
    # TODO: Train multiple models
    # TODO: Cross-validate best model
    # TODO: Evaluate on test set
    # TODO: Print comprehensive results

    print("\nTODO: Implement this exercise")
    print("Steps:")
    print("  1. Load and explore data")
    print("  2. Split data (stratified)")
    print("  3. Scale features")
    print("  4. Train 3+ models")
    print("  5. Select best via CV")
    print("  6. Evaluate on test set")
    print("  7. Print final metrics and insights")


# Bonus exercises
def bonus_exercise_1():
    """
    Bonus 1: ROC Curve Analysis
    Calculate and interpret ROC curve for binary classification
    """
    print("\n" + "=" * 60)
    print("Bonus Exercise 1: ROC Curve Analysis")
    print("=" * 60)

    # TODO: Load breast cancer dataset
    # TODO: Train logistic regression
    # TODO: Get probability predictions
    # TODO: Calculate ROC curve points
    # TODO: Calculate AUC score
    # TODO: Find optimal threshold

    print("\nTODO: Implement this bonus exercise")


def bonus_exercise_2():
    """
    Bonus 2: Feature Engineering
    Create new features and compare performance
    """
    print("\n" + "=" * 60)
    print("Bonus Exercise 2: Feature Engineering")
    print("=" * 60)

    # TODO: Load diabetes dataset
    # TODO: Create polynomial features (x²)
    # TODO: Create interaction features (x1*x2)
    # TODO: Compare model with/without new features

    print("\nTODO: Implement this bonus exercise")


def solution_example():
    """
    Example solution for Exercise 1
    """
    print("\n" + "=" * 60)
    print("Example Solution: Exercise 1")
    print("=" * 60)

    # Load data
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    print(f"Dataset: {data.DESCR.split(':', 1)[0]}")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(data.target_names)} {data.target_names}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")

    # Train model
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(f"  [[{cm[0,0]:3d} {cm[0,1]:3d}]")
    print(f"   [{cm[1,0]:3d} {cm[1,1]:3d}]]")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=data.target_names))


def main():
    print("=" * 60)
    print("Machine Learning Basics - Practice Exercises")
    print("=" * 60)

    print("\nThis file contains 10 exercises covering all ML basics topics.")
    print("Each exercise is a function you need to implement.")
    print("\nTopics covered:")
    print("  1. Binary classification")
    print("  2. Multiclass classification")
    print("  3. Model comparison")
    print("  4. Hyperparameter tuning")
    print("  5. Feature selection")
    print("  6. Cross-validation")
    print("  7. Imbalanced data")
    print("  8. Ensemble methods")
    print("  9. Regression")
    print("  10. Full ML pipeline")

    print("\n" + "=" * 60)
    print("Running Example Solution (Exercise 1)")
    print("=" * 60)
    solution_example()

    print("\n" + "=" * 60)
    print("Your Turn: Implement the Following Exercises")
    print("=" * 60)

    # Uncomment to see exercise descriptions
    exercise_1_breast_cancer()
    exercise_2_wine_quality()
    exercise_3_model_comparison()
    exercise_4_hyperparameter_tuning()
    exercise_5_feature_selection()
    exercise_6_cross_validation_study()
    exercise_7_imbalanced_classification()
    exercise_8_ensemble_comparison()
    exercise_9_regression_analysis()
    exercise_10_full_pipeline()

    print("\n" + "=" * 60)
    print("Bonus Challenges")
    print("=" * 60)
    bonus_exercise_1()
    bonus_exercise_2()

    print("\n" + "=" * 60)
    print("How to Complete These Exercises:")
    print("  1. Read each exercise description carefully")
    print("  2. Implement the TODO sections")
    print("  3. Run the code and verify results")
    print("  4. Compare with example solution")
    print("  5. Experiment with different parameters")
    print("\nTips:")
    print("  - Start with simpler exercises (1-3)")
    print("  - Use the example solution as a template")
    print("  - Print intermediate results to debug")
    print("  - Try to explain why certain models perform better")
    print("=" * 60)


if __name__ == "__main__":
    main()
