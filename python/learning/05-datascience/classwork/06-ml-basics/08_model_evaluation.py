"""
Model Evaluation
================
Comprehensive metrics for evaluating machine learning models.

Topics:
- Confusion matrix
- Accuracy, precision, recall, F1-score
- Classification report
- ROC curve and AUC
- Precision-recall curve
- Regression metrics (MSE, RMSE, R², MAE)
- When to use which metric

Run: python 08_model_evaluation.py
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, mean_squared_error, mean_absolute_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

def main():
    print("=" * 60)
    print("Model Evaluation Metrics")
    print("=" * 60)

    # 1. Binary classification setup
    print("\n1. Binary Classification Setup")
    print("-" * 40)

    # Load iris, use only first 2 classes
    iris = datasets.load_iris()
    X_binary = iris.data[iris.target != 2]
    y_binary = iris.target[iris.target != 2]

    print(f"Binary classification: {iris.target_names[0]} vs {iris.target_names[1]}")
    print(f"  Total samples: {len(X_binary)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y_binary, test_size=0.3, stratify=y_binary, random_state=42
    )

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")

    # 2. Confusion matrix
    print("\n2. Confusion Matrix")
    print("-" * 40)

    cm = confusion_matrix(y_test, predictions)

    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"              0         1")
    print(f"Actual  0   {cm[0,0]:3d}       {cm[0,1]:3d}")
    print(f"        1   {cm[1,0]:3d}       {cm[1,1]:3d}")

    tn, fp, fn, tp = cm.ravel()

    print("\nBreakdown:")
    print(f"  True Negatives (TN):  {tn} - Correctly predicted class 0")
    print(f"  False Positives (FP): {fp} - Wrongly predicted class 1")
    print(f"  False Negatives (FN): {fn} - Wrongly predicted class 0")
    print(f"  True Positives (TP):  {tp} - Correctly predicted class 1")

    # 3. Basic metrics
    print("\n3. Basic Metrics")
    print("-" * 40)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"  Formula: (TP + TN) / Total")
    print(f"  = ({tp} + {tn}) / {len(y_test)} = {accuracy:.4f}")
    print(f"  Interpretation: {accuracy*100:.1f}% of predictions are correct")

    print(f"\nPrecision: {precision:.4f}")
    print(f"  Formula: TP / (TP + FP)")
    print(f"  = {tp} / ({tp} + {fp}) = {precision:.4f}")
    print(f"  Interpretation: {precision*100:.1f}% of predicted positives are correct")

    print(f"\nRecall:    {recall:.4f}")
    print(f"  Formula: TP / (TP + FN)")
    print(f"  = {tp} / ({tp} + {fn}) = {recall:.4f}")
    print(f"  Interpretation: {recall*100:.1f}% of actual positives were found")

    print(f"\nF1-Score:  {f1:.4f}")
    print(f"  Formula: 2 * (Precision * Recall) / (Precision + Recall)")
    print(f"  = 2 * ({precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
    print(f"  = {f1:.4f}")
    print(f"  Interpretation: Harmonic mean of precision and recall")

    # 4. When to use which metric
    print("\n4. When to Use Which Metric")
    print("-" * 40)

    print("Accuracy:")
    print("  Use when: Classes are balanced")
    print("  Don't use: Imbalanced data (e.g., fraud detection)")

    print("\nPrecision:")
    print("  Use when: False positives are costly")
    print("  Example: Spam detection (don't mark good emails as spam)")

    print("\nRecall:")
    print("  Use when: False negatives are costly")
    print("  Example: Disease detection (don't miss sick patients)")

    print("\nF1-Score:")
    print("  Use when: Balance precision and recall")
    print("  Example: Search engines (relevant and complete results)")

    # 5. Classification report
    print("\n5. Classification Report")
    print("-" * 40)

    print("Comprehensive report for all classes:")
    print(classification_report(
        y_test, predictions,
        target_names=[iris.target_names[0], iris.target_names[1]],
        digits=4
    ))

    # 6. Imbalanced data example
    print("\n6. Metrics on Imbalanced Data")
    print("-" * 40)

    # Create imbalanced dataset (90% class 0, 10% class 1)
    np.random.seed(42)
    n_samples = 1000
    n_class1 = 100

    # Generate imbalanced data
    idx_class0 = np.random.choice(len(X_binary[y_binary==0]), n_samples-n_class1)
    idx_class1 = np.random.choice(len(X_binary[y_binary==1]), n_class1)

    X_imb = np.vstack([
        X_binary[y_binary==0][idx_class0],
        X_binary[y_binary==1][idx_class1]
    ])
    y_imb = np.concatenate([
        np.zeros(n_samples-n_class1),
        np.ones(n_class1)
    ])

    print(f"Imbalanced dataset:")
    print(f"  Class 0: {np.sum(y_imb==0)} samples ({np.sum(y_imb==0)/len(y_imb)*100:.1f}%)")
    print(f"  Class 1: {np.sum(y_imb==1)} samples ({np.sum(y_imb==1)/len(y_imb)*100:.1f}%)")

    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
        X_imb, y_imb, test_size=0.3, stratify=y_imb, random_state=42
    )

    model_imb = LogisticRegression(random_state=42)
    model_imb.fit(X_train_imb, y_train_imb)
    pred_imb = model_imb.predict(X_test_imb)

    acc_imb = accuracy_score(y_test_imb, pred_imb)
    prec_imb = precision_score(y_test_imb, pred_imb)
    rec_imb = recall_score(y_test_imb, pred_imb)
    f1_imb = f1_score(y_test_imb, pred_imb)

    print("\nMetrics on imbalanced data:")
    print(f"  Accuracy:  {acc_imb:.4f}")
    print(f"  Precision: {prec_imb:.4f}")
    print(f"  Recall:    {rec_imb:.4f}")
    print(f"  F1-Score:  {f1_imb:.4f}")

    print("\nNote: High accuracy can be misleading!")
    print(f"  A 'dumb' classifier predicting always class 0: {90:.1f}% accuracy")
    print("  But 0% recall for class 1 (misses all minority class)")

    # 7. ROC curve
    print("\n7. ROC Curve and AUC")
    print("-" * 40)

    print("ROC (Receiver Operating Characteristic) Curve:")
    print("  - Plots True Positive Rate vs False Positive Rate")
    print("  - TPR = Recall = TP / (TP + FN)")
    print("  - FPR = FP / (FP + TN)")
    print("  - Shows trade-off at different thresholds")

    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)

    print(f"\nAUC (Area Under Curve): {auc:.4f}")
    print("  AUC interpretation:")
    print("    1.0 = Perfect classifier")
    print("    0.5 = Random classifier")
    print("    <0.5 = Worse than random")

    print(f"\nSample ROC points (threshold, FPR, TPR):")
    for i in [0, len(thresholds)//4, len(thresholds)//2, -1]:
        if i < len(thresholds):
            print(f"  {thresholds[i]:.3f}    {fpr[i]:.3f}    {tpr[i]:.3f}")

    # 8. Precision-recall curve
    print("\n8. Precision-Recall Curve")
    print("-" * 40)

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_test, probabilities
    )

    print("Precision-Recall Curve:")
    print("  - Plots Precision vs Recall")
    print("  - Useful for imbalanced datasets")
    print("  - Focuses on positive class performance")

    print(f"\nSample PR points (threshold, Precision, Recall):")
    for i in [0, len(pr_thresholds)//4, len(pr_thresholds)//2, -1]:
        if i < len(pr_thresholds):
            print(f"  {pr_thresholds[i]:.3f}    {precision_curve[i]:.3f}    {recall_curve[i]:.3f}")

    # 9. Multiclass evaluation
    print("\n9. Multiclass Classification Metrics")
    print("-" * 40)

    # Use full iris dataset (3 classes)
    X, y = iris.data, iris.target

    X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model_mc = LogisticRegression(random_state=42, max_iter=200)
    model_mc.fit(X_train_mc, y_train_mc)
    pred_mc = model_mc.predict(X_test_mc)

    print("Multiclass confusion matrix:")
    cm_mc = confusion_matrix(y_test_mc, pred_mc)
    print("              Predicted")
    print("           0    1    2")
    for i, row in enumerate(cm_mc):
        print(f"Actual {i}  {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")

    print("\nMulticlass metrics (macro average):")
    acc_mc = accuracy_score(y_test_mc, pred_mc)
    prec_mc = precision_score(y_test_mc, pred_mc, average='macro')
    rec_mc = recall_score(y_test_mc, pred_mc, average='macro')
    f1_mc = f1_score(y_test_mc, pred_mc, average='macro')

    print(f"  Accuracy:  {acc_mc:.4f}")
    print(f"  Precision: {prec_mc:.4f} (macro avg)")
    print(f"  Recall:    {rec_mc:.4f} (macro avg)")
    print(f"  F1-Score:  {f1_mc:.4f} (macro avg)")

    print("\nAveraging strategies:")
    print("  - macro: Average of per-class metrics (equal weight)")
    print("  - weighted: Weighted by class support")
    print("  - micro: Global average (count all TP, FP, FN)")

    # 10. Regression metrics
    print("\n10. Regression Metrics")
    print("-" * 40)

    # Load diabetes dataset
    diabetes = datasets.load_diabetes()
    X_reg = diabetes.data
    y_reg = diabetes.target

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )

    from sklearn.linear_model import LinearRegression
    model_reg = LinearRegression()
    model_reg.fit(X_train_reg, y_train_reg)
    pred_reg = model_reg.predict(X_test_reg)

    mse = mean_squared_error(y_test_reg, pred_reg)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reg, pred_reg)
    r2 = r2_score(y_test_reg, pred_reg)

    print("Regression metrics:")
    print(f"\nMSE (Mean Squared Error): {mse:.2f}")
    print(f"  Formula: avg((actual - predicted)²)")
    print(f"  Penalizes large errors heavily")

    print(f"\nRMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"  Formula: sqrt(MSE)")
    print(f"  Same units as target variable")
    print(f"  Easier to interpret than MSE")

    print(f"\nMAE (Mean Absolute Error): {mae:.2f}")
    print(f"  Formula: avg(|actual - predicted|)")
    print(f"  More robust to outliers than MSE")

    print(f"\nR² (R-squared): {r2:.4f}")
    print(f"  Formula: 1 - (SS_res / SS_tot)")
    print(f"  Range: (-∞, 1], best = 1.0")
    print(f"  Explains {r2*100:.1f}% of variance")

    # 11. Comparing models
    print("\n11. Comparing Multiple Models")
    print("-" * 40)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=200),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
    }

    print(f"{'Model':<20s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print("-" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        print(f"{name:<20s} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

    print("\n" + "=" * 60)
    print("Summary - Choosing Metrics:")
    print("\nClassification:")
    print("  Balanced data:")
    print("    - Accuracy: Overall correctness")
    print("    - F1-Score: Balance of precision and recall")
    print("  Imbalanced data:")
    print("    - Precision: When FP is costly")
    print("    - Recall: When FN is costly")
    print("    - F1-Score: Balance both")
    print("    - AUC-ROC: Threshold-independent performance")
    print("\nRegression:")
    print("  - R²: How well model explains variance")
    print("  - RMSE: Average prediction error (same units)")
    print("  - MAE: Robust to outliers")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Plot ROC curve and find optimal threshold")
    print("2. Calculate metrics manually from confusion matrix")
    print("3. Compare weighted vs macro average for imbalanced data")
    print("4. Find model that maximizes recall while keeping precision > 0.8")
    print("=" * 60)

if __name__ == "__main__":
    main()
