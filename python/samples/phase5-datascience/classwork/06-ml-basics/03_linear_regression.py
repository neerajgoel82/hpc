"""
Linear Regression
=================
Simple and multiple linear regression with scikit-learn.

Topics:
- Simple linear regression (one feature)
- Multiple linear regression (many features)
- Model coefficients and intercept
- R-squared score
- Predictions and residuals
- Built-in datasets (diabetes, boston)

Run: python 03_linear_regression.py
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets

def main():
    print("=" * 60)
    print("Linear Regression")
    print("=" * 60)

    # 1. Simple linear regression
    print("\n1. Simple Linear Regression")
    print("-" * 40)

    # Create simple dataset
    np.random.seed(42)
    X_simple = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y_simple = 2 * X_simple.flatten() + 1 + np.random.randn(10) * 0.5

    print(f"Training data: {len(X_simple)} samples")
    print("First 5 samples:")
    for i in range(5):
        print(f"  X={X_simple[i][0]:.1f}, y={y_simple[i]:.2f}")

    # Fit model
    model = LinearRegression()
    model.fit(X_simple, y_simple)

    print("\nModel learned:")
    print(f"  Coefficient (slope): {model.coef_[0]:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    print(f"  Equation: y = {model.coef_[0]:.4f}*x + {model.intercept_:.4f}")

    # Make predictions
    predictions = model.predict(X_simple)
    print("\nPredictions vs Actual (first 5):")
    for i in range(5):
        print(f"  X={X_simple[i][0]:.1f}: Predicted={predictions[i]:.2f}, "
              f"Actual={y_simple[i]:.2f}")

    # Calculate metrics
    r2 = model.score(X_simple, y_simple)
    mse = mean_squared_error(y_simple, predictions)
    rmse = np.sqrt(mse)

    print(f"\nModel performance:")
    print(f"  R² score: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    # 2. Understanding R-squared
    print("\n2. Understanding R-Squared")
    print("-" * 40)

    print("R² measures how well the model explains variance in y")
    print(f"  R² = {r2:.4f}")
    print(f"  Interpretation: Model explains {r2*100:.1f}% of variance")
    print("\nR² ranges from 0 to 1:")
    print("  1.0 = Perfect predictions")
    print("  0.0 = Model no better than predicting mean")
    print("  <0.0 = Model worse than predicting mean")

    # 3. Residuals
    print("\n3. Residuals (Errors)")
    print("-" * 40)

    residuals = y_simple - predictions
    print("Residuals (actual - predicted):")
    for i in range(5):
        print(f"  Sample {i+1}: {residuals[i]:.4f}")

    print(f"\nResidual statistics:")
    print(f"  Mean: {residuals.mean():.4f} (should be ~0)")
    print(f"  Std: {residuals.std():.4f}")
    print(f"  Min: {residuals.min():.4f}")
    print(f"  Max: {residuals.max():.4f}")

    # 4. Multiple linear regression
    print("\n4. Multiple Linear Regression")
    print("-" * 40)

    # Load diabetes dataset (10 features)
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    print(f"Diabetes dataset:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target: Disease progression (continuous)")

    print("\nFeature names:")
    for i, name in enumerate(diabetes.feature_names):
        print(f"  {i+1}. {name}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")

    # 5. Training the model
    print("\n5. Training Multiple Regression Model")
    print("-" * 40)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Model trained successfully")
    print(f"  Intercept: {model.intercept_:.2f}")
    print("\nCoefficients (feature importance):")
    for name, coef in zip(diabetes.feature_names, model.coef_):
        print(f"  {name:8s}: {coef:7.2f}")

    # 6. Making predictions
    print("\n6. Making Predictions")
    print("-" * 40)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print("Sample predictions on test set:")
    for i in range(5):
        print(f"  Predicted: {test_pred[i]:6.1f}, Actual: {y_test[i]:6.1f}, "
              f"Error: {abs(test_pred[i] - y_test[i]):5.1f}")

    # 7. Evaluating the model
    print("\n7. Model Evaluation")
    print("-" * 40)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    print("Training set performance:")
    print(f"  R² score: {train_r2:.4f}")
    print(f"  MSE: {train_mse:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")

    print("\nTest set performance:")
    print(f"  R² score: {test_r2:.4f}")
    print(f"  MSE: {test_mse:.2f}")
    print(f"  RMSE: {test_rmse:.2f}")

    print("\nInterpretation:")
    print(f"  Model explains {test_r2*100:.1f}% of variance in test data")
    print(f"  Average prediction error: {test_rmse:.1f} units")

    # 8. Feature importance
    print("\n8. Feature Importance Analysis")
    print("-" * 40)

    # Get absolute coefficients for importance
    coef_importance = np.abs(model.coef_)
    feature_importance = sorted(
        zip(diabetes.feature_names, coef_importance),
        key=lambda x: x[1],
        reverse=True
    )

    print("Features ranked by absolute coefficient:")
    for rank, (name, importance) in enumerate(feature_importance, 1):
        print(f"  {rank}. {name:8s}: {importance:.2f}")

    # 9. Single feature regression for comparison
    print("\n9. Single Feature vs Multiple Features")
    print("-" * 40)

    # Use only BMI feature (most important)
    X_bmi = X[:, 2:3]  # Keep 2D shape
    X_train_bmi, X_test_bmi, y_train_bmi, y_test_bmi = train_test_split(
        X_bmi, y, test_size=0.2, random_state=42
    )

    model_bmi = LinearRegression()
    model_bmi.fit(X_train_bmi, y_train_bmi)

    bmi_r2 = model_bmi.score(X_test_bmi, y_test_bmi)

    print("Single feature (BMI only):")
    print(f"  R² score: {bmi_r2:.4f}")

    print("\nAll features:")
    print(f"  R² score: {test_r2:.4f}")

    print(f"\nImprovement: {(test_r2 - bmi_r2)*100:.1f}% better with all features")

    # 10. Prediction on new data
    print("\n10. Predicting on New Data")
    print("-" * 40)

    # Create new sample (mean values of features)
    new_sample = X.mean(axis=0).reshape(1, -1)

    print("New patient with average feature values:")
    for name, value in zip(diabetes.feature_names, new_sample[0]):
        print(f"  {name}: {value:.3f}")

    prediction = model.predict(new_sample)
    print(f"\nPredicted disease progression: {prediction[0]:.1f}")
    print(f"Average actual progression: {y.mean():.1f}")

    print("\n" + "=" * 60)
    print("Summary - Linear Regression:")
    print("  - Simple regression: y = mx + b (one feature)")
    print("  - Multiple regression: y = b0 + b1*x1 + b2*x2 + ...")
    print("  - Coefficients show feature importance")
    print("  - R² measures goodness of fit (0 to 1)")
    print("  - RMSE gives average prediction error")
    print("  - More features generally improve predictions")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Try polynomial features (x, x², x³) for non-linear data")
    print("2. Compare performance with different train/test splits")
    print("3. Identify and remove least important features")
    print("4. Predict on new data with different feature values")
    print("=" * 60)

if __name__ == "__main__":
    main()
