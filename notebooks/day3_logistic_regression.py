"""
=========================================
AIML Daily Practice
Day 03 - Feature Scaling & Model Evaluation
=========================================

Author: Harsh
Date: 26-01-2026

Description:
This script demonstrates the importance of
feature scaling and proper model evaluation
in machine learning.

Concepts Covered:
- Feature Scaling
- Normalization
- Train-Test Split
- R2 Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Objective:
Understand why scaling improves convergence
and how to properly evaluate a model.

Status:
Implemented successfully.
=========================================
"""

# ============================================
# Day 3 - Model Evaluation & Visualization
# AIML Daily Practice
# ============================================

import matplotlib.pyplot as plt

# Sample Data
X = [1, 2, 3, 4]
Y = [2, 4, 6, 8]

# Optimized parameters from Day 2 (manually set for now)
w = 2
b = 0


# Prediction Function
def predict(x, w, b):
    return w * x + b


# Mean Absolute Error
def compute_mae(X, Y, w, b):
    total_error = 0
    m = len(X)

    for i in range(m):
        y_pred = predict(X[i], w, b)
        total_error += abs(y_pred - Y[i])

    return total_error / m


# Mean Squared Error
def compute_mse(X, Y, w, b):
    total_error = 0
    m = len(X)

    for i in range(m):
        y_pred = predict(X[i], w, b)
        total_error += (y_pred - Y[i]) ** 2

    return total_error / m


# Print Predictions
print("Predictions:")
for i in range(len(X)):
    print(f"X: {X[i]}, Actual: {Y[i]}, Predicted: {predict(X[i], w, b)}")

print("\nMAE:", compute_mae(X, Y, w, b))
print("MSE:", compute_mse(X, Y, w, b))


# Visualization
plt.scatter(X, Y)
predictions = [predict(x, w, b) for x in X]
plt.plot(X, predictions)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Fit - Day 3")
plt.show()
