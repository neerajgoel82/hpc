"""
Project: Customer Churn Prediction
==================================
Classification problem with data cleaning, feature engineering, and multiple models.

Dataset: Customer data with churn labels
Goals:
- Data preprocessing and comprehensive EDA
- Handle class imbalance
- Feature engineering for better predictions
- Build and compare multiple classification models
- Evaluate with appropriate metrics (precision, recall, F1)
- Feature importance analysis
- Generate actionable insights

Skills: Pandas, scikit-learn, visualization, classification metrics
Run: python project_churn_prediction.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, roc_curve)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def generate_churn_data(n_samples=2000):
    """Generate synthetic customer churn data."""
    print("Generating customer churn dataset...")

    np.random.seed(42)

    # Generate features that influence churn
    # Customers with high churn probability
    n_churned = int(n_samples * 0.25)  # 25% churn rate
    n_retained = n_samples - n_churned

    # Churned customers characteristics
    churned_data = {
        'tenure_months': np.random.exponential(8, n_churned).clip(1, 72),
        'monthly_charges': np.random.normal(85, 20, n_churned).clip(20, 150),
        'total_charges': np.random.normal(2500, 1500, n_churned).clip(100, 10000),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                         n_churned, p=[0.7, 0.2, 0.1]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check',
                                           'Bank transfer', 'Credit card'],
                                          n_churned, p=[0.5, 0.2, 0.15, 0.15]),
        'tech_support': np.random.choice([0, 1], n_churned, p=[0.7, 0.3]),
        'online_security': np.random.choice([0, 1], n_churned, p=[0.7, 0.3]),
        'customer_service_calls': np.random.poisson(4, n_churned).clip(0, 10),
        'age': np.random.normal(35, 15, n_churned).clip(18, 80),
        'num_products': np.random.randint(1, 3, n_churned),
        'churn': 1
    }

    # Retained customers characteristics
    retained_data = {
        'tenure_months': np.random.exponential(30, n_retained).clip(1, 72),
        'monthly_charges': np.random.normal(70, 25, n_retained).clip(20, 150),
        'total_charges': np.random.normal(4500, 2500, n_retained).clip(100, 10000),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                         n_retained, p=[0.3, 0.3, 0.4]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check',
                                           'Bank transfer', 'Credit card'],
                                          n_retained, p=[0.2, 0.2, 0.3, 0.3]),
        'tech_support': np.random.choice([0, 1], n_retained, p=[0.3, 0.7]),
        'online_security': np.random.choice([0, 1], n_retained, p=[0.3, 0.7]),
        'customer_service_calls': np.random.poisson(1.5, n_retained).clip(0, 10),
        'age': np.random.normal(45, 18, n_retained).clip(18, 80),
        'num_products': np.random.randint(2, 5, n_retained),
        'churn': 0
    }

    # Combine data
    churned_df = pd.DataFrame(churned_data)
    retained_df = pd.DataFrame(retained_data)
    df = pd.concat([churned_df, retained_df], ignore_index=True)

    # Add customer IDs
    df['customer_id'] = [f'CUST{str(i).zfill(5)}' for i in range(len(df))]

    # Add internet service
    df['internet_service'] = np.random.choice(['DSL', 'Fiber optic', 'No'],
                                              len(df), p=[0.4, 0.4, 0.2])

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'total_charges'] = np.nan

    missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_indices, 'tenure_months'] = np.nan

    # Ensure data types
    df['tenure_months'] = df['tenure_months'].astype(float)
    df['age'] = df['age'].astype(int)
    df['customer_service_calls'] = df['customer_service_calls'].astype(int)

    print(f"Generated {len(df)} customer records")
    print(f"Churn rate: {df['churn'].mean()*100:.1f}%")

    return df


def exploratory_analysis(df):
    """Comprehensive exploratory data analysis."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\nDataset Overview:")
    print(f"  Shape: {df.shape}")
    print(f"  Churn rate: {df['churn'].mean()*100:.2f}%")
    print(f"  Churned customers: {df['churn'].sum()}")
    print(f"  Retained customers: {(df['churn']==0).sum()}")

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
    print(missing_df[missing_df['Count'] > 0])

    print("\nDescriptive Statistics:")
    print(df.describe().round(2))

    # Analyze churn by categorical features
    print("\nChurn Rate by Contract Type:")
    churn_by_contract = df.groupby('contract_type')['churn'].agg(['mean', 'count'])
    churn_by_contract['mean'] = (churn_by_contract['mean'] * 100).round(2)
    print(churn_by_contract)

    print("\nChurn Rate by Payment Method:")
    churn_by_payment = df.groupby('payment_method')['churn'].agg(['mean', 'count'])
    churn_by_payment['mean'] = (churn_by_payment['mean'] * 100).round(2)
    print(churn_by_payment)

    print("\nChurn Rate by Internet Service:")
    churn_by_internet = df.groupby('internet_service')['churn'].agg(['mean', 'count'])
    churn_by_internet['mean'] = (churn_by_internet['mean'] * 100).round(2)
    print(churn_by_internet)

    # Numeric features by churn
    print("\nNumeric Features by Churn Status:")
    numeric_cols = ['tenure_months', 'monthly_charges', 'age', 'customer_service_calls']
    comparison = df.groupby('churn')[numeric_cols].mean()
    print(comparison.round(2))


def clean_and_engineer_features(df):
    """Clean data and create new features."""
    print("\n" + "=" * 60)
    print("DATA CLEANING & FEATURE ENGINEERING")
    print("=" * 60)

    df = df.copy()

    # Handle missing values
    print(f"Missing values before: {df.isnull().sum().sum()}")

    # Fill total_charges with median
    df['total_charges'].fillna(df['total_charges'].median(), inplace=True)

    # Fill tenure_months with median
    df['tenure_months'].fillna(df['tenure_months'].median(), inplace=True)

    print(f"Missing values after: {df.isnull().sum().sum()}")

    # Feature engineering
    print("\nEngineering new features...")

    # Average monthly spend
    df['avg_monthly_spend'] = df['total_charges'] / df['tenure_months'].replace(0, 1)

    # Tenure categories
    df['tenure_category'] = pd.cut(df['tenure_months'],
                                    bins=[0, 12, 24, 48, 100],
                                    labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])

    # Age groups
    df['age_group'] = pd.cut(df['age'],
                             bins=[0, 30, 45, 60, 100],
                             labels=['Young', 'Middle', 'Senior', 'Elder'])

    # Service quality score
    df['service_quality'] = df['tech_support'] + df['online_security']

    # High value customer (top 25% in total charges)
    df['high_value'] = (df['total_charges'] > df['total_charges'].quantile(0.75)).astype(int)

    # Charge per product
    df['charge_per_product'] = df['monthly_charges'] / df['num_products']

    # Risk flag (high service calls + low tenure)
    df['risk_flag'] = ((df['customer_service_calls'] > 3) &
                       (df['tenure_months'] < 12)).astype(int)

    # Encode categorical variables
    print("\nEncoding categorical variables...")

    # Label encoding for ordinal features
    le_contract = LabelEncoder()
    df['contract_encoded'] = le_contract.fit_transform(df['contract_type'])

    # One-hot encoding for nominal features
    payment_dummies = pd.get_dummies(df['payment_method'], prefix='payment')
    internet_dummies = pd.get_dummies(df['internet_service'], prefix='internet')

    df = pd.concat([df, payment_dummies, internet_dummies], axis=1)

    print("New features created:")
    print("  - avg_monthly_spend")
    print("  - tenure_category")
    print("  - age_group")
    print("  - service_quality")
    print("  - high_value")
    print("  - charge_per_product")
    print("  - risk_flag")
    print("  - contract_encoded")
    print("  - payment_* (one-hot)")
    print("  - internet_* (one-hot)")

    return df


def prepare_model_data(df):
    """Prepare features and target for modeling."""
    print("\n" + "=" * 60)
    print("PREPARING DATA FOR MODELING")
    print("=" * 60)

    # Select features
    feature_cols = [
        'tenure_months', 'monthly_charges', 'total_charges', 'age',
        'num_products', 'tech_support', 'online_security',
        'customer_service_calls', 'avg_monthly_spend', 'service_quality',
        'high_value', 'charge_per_product', 'risk_flag', 'contract_encoded',
        'payment_Bank transfer', 'payment_Credit card',
        'payment_Electronic check', 'payment_Mailed check',
        'internet_DSL', 'internet_Fiber optic', 'internet_No'
    ]

    X = df[feature_cols].copy()
    y = df['churn'].copy()

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Check for class imbalance
    imbalance_ratio = y.value_counts()[0] / y.value_counts()[1]
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 2:
        print("WARNING: Significant class imbalance detected!")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"  Churn: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"Test set:  {len(X_test)} samples")
    print(f"  Churn: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Features scaled (StandardScaler)")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_cols


def train_and_evaluate_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train multiple classification models."""
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    models = {
        'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000),
                               X_train_scaled, X_test_scaled),
        'Decision Tree': (DecisionTreeClassifier(random_state=42, max_depth=10),
                         X_train, X_test),
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42),
                         X_train, X_test),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, random_state=42),
                             X_train, X_test)
    }

    results = {}

    for name, (model, X_tr, X_te) in models.items():
        print(f"\n{'-' * 60}")
        print(f"Training {name}...")

        # Train
        model.fit(X_tr, y_train)

        # Predictions
        y_train_pred = model.predict(X_tr)
        y_test_pred = model.predict(X_te)

        # Probabilities for ROC AUC
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_te)[:, 1]
        else:
            y_test_proba = y_test_pred

        # Evaluate
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_proba)

        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Precision:      {precision:.4f}")
        print(f"  Recall:         {recall:.4f}")
        print(f"  F1 Score:       {f1:.4f}")
        print(f"  ROC AUC:        {roc_auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
        print(f"    FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_test_pred,
            'y_proba': y_test_proba,
            'confusion_matrix': cm
        }

    return results


def analyze_feature_importance(model, feature_cols):
    """Analyze feature importance."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15).to_string(index=False))

        return feature_importance
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nTop 15 Most Important Features (by coefficient magnitude):")
        print(feature_importance.head(15).to_string(index=False))

        return feature_importance
    else:
        print("Model does not support feature importance analysis")
        return None


def create_visualizations(df, results, y_test):
    """Create comprehensive visualizations."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 12))

    # 1. Churn distribution
    ax1 = plt.subplot(3, 3, 1)
    churn_counts = df['churn'].value_counts()
    ax1.bar(['Retained', 'Churned'], churn_counts.values, color=['green', 'red'], alpha=0.7)
    ax1.set_ylabel('Count')
    ax1.set_title('Churn Distribution', fontweight='bold')
    for i, v in enumerate(churn_counts.values):
        ax1.text(i, v, str(v), ha='center', va='bottom')

    # 2. Tenure by churn
    ax2 = plt.subplot(3, 3, 2)
    df.boxplot(column='tenure_months', by='churn', ax=ax2)
    ax2.set_xlabel('Churn (0=No, 1=Yes)')
    ax2.set_ylabel('Tenure (months)')
    ax2.set_title('Tenure by Churn Status', fontweight='bold')
    plt.suptitle('')

    # 3. Monthly charges by churn
    ax3 = plt.subplot(3, 3, 3)
    df.boxplot(column='monthly_charges', by='churn', ax=ax3)
    ax3.set_xlabel('Churn (0=No, 1=Yes)')
    ax3.set_ylabel('Monthly Charges ($)')
    ax3.set_title('Monthly Charges by Churn', fontweight='bold')
    plt.suptitle('')

    # 4. Churn by contract type
    ax4 = plt.subplot(3, 3, 4)
    contract_churn = df.groupby('contract_type')['churn'].mean() * 100
    contract_churn.plot(kind='bar', ax=ax4, color='coral')
    ax4.set_xlabel('Contract Type')
    ax4.set_ylabel('Churn Rate (%)')
    ax4.set_title('Churn Rate by Contract Type', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)

    # 5. Model comparison - Accuracy
    ax5 = plt.subplot(3, 3, 5)
    model_names = list(results.keys())
    test_accs = [results[m]['test_acc'] for m in model_names]
    ax5.barh(model_names, test_accs, color='steelblue')
    ax5.set_xlabel('Test Accuracy')
    ax5.set_title('Model Comparison: Accuracy', fontweight='bold')

    # 6. Model comparison - F1 Score
    ax6 = plt.subplot(3, 3, 6)
    f1_scores = [results[m]['f1'] for m in model_names]
    ax6.barh(model_names, f1_scores, color='lightgreen')
    ax6.set_xlabel('F1 Score')
    ax6.set_title('Model Comparison: F1 Score', fontweight='bold')

    # 7. ROC Curves
    ax7 = plt.subplot(3, 3, 7)
    for name in model_names:
        if hasattr(results[name]['model'], 'predict_proba'):
            y_proba = results[name]['y_proba']
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = results[name]['roc_auc']
            ax7.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    ax7.plot([0, 1], [0, 1], 'k--', label='Random')
    ax7.set_xlabel('False Positive Rate')
    ax7.set_ylabel('True Positive Rate')
    ax7.set_title('ROC Curves', fontweight='bold')
    ax7.legend(fontsize=8)

    # 8. Best model confusion matrix
    ax8 = plt.subplot(3, 3, 8)
    best_model_name = max(results.keys(), key=lambda m: results[m]['f1'])
    cm = results[best_model_name]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax8,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax8.set_ylabel('Actual')
    ax8.set_xlabel('Predicted')
    ax8.set_title(f'{best_model_name}: Confusion Matrix', fontweight='bold')

    # 9. Service calls vs churn
    ax9 = plt.subplot(3, 3, 9)
    calls_churn = df.groupby('customer_service_calls')['churn'].mean() * 100
    ax9.plot(calls_churn.index, calls_churn.values, marker='o', linewidth=2)
    ax9.set_xlabel('Customer Service Calls')
    ax9.set_ylabel('Churn Rate (%)')
    ax9.set_title('Churn Rate vs Service Calls', fontweight='bold')
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    print("Visualizations created successfully!")
    print("Close the plot window to continue...")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION PROJECT")
    print("=" * 60)

    # 1. Generate data
    df = generate_churn_data(n_samples=2000)

    # 2. Exploratory analysis
    exploratory_analysis(df)

    # 3. Clean and engineer features
    df = clean_and_engineer_features(df)

    # 4. Prepare data for modeling
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_cols = prepare_model_data(df)

    # 5. Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test,
                                       X_train_scaled, X_test_scaled)

    # 6. Find best model
    best_model_name = max(results.keys(), key=lambda m: results[m]['f1'])
    best_model = results[best_model_name]['model']
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"  Test Accuracy: {results[best_model_name]['test_acc']:.4f}")
    print(f"  F1 Score:      {results[best_model_name]['f1']:.4f}")
    print(f"  ROC AUC:       {results[best_model_name]['roc_auc']:.4f}")
    print('=' * 60)

    # 7. Feature importance
    feature_importance = analyze_feature_importance(best_model, feature_cols)

    # 8. Create visualizations
    create_visualizations(df, results, y_test)

    # 9. Business insights
    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)

    print("\nKey Churn Indicators:")
    if feature_importance is not None:
        top_features = feature_importance.head(5)['feature'].tolist()
        for i, feature in enumerate(top_features, 1):
            print(f"{i}. {feature}")

    print("\nActionable Recommendations:")
    print("1. Focus retention efforts on month-to-month contract customers")
    print("2. Proactively reach out to customers with >3 service calls")
    print("3. Offer incentives for long-term contracts")
    print("4. Improve tech support and online security adoption")
    print("5. Target high-risk customers (low tenure + high service calls)")
    print("6. Consider pricing adjustments for high-churn segments")

    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
