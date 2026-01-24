"""
=========================================
AIML Daily Practice
Day 02 - Gradient Descent (From Scratch)
=========================================

Author: Harsh
Date: 25-01-2026

Description:
This script implements Gradient Descent to optimize
the cost function of Linear Regression.

Concepts Covered:
- Cost function minimization
- Learning rate
- Iterative parameter updates
- Convergence

Objective:
Understand how model parameters are updated
to minimize prediction error.

Status:
Implemented successfully.
=========================================
"""

# ============================================
# Day 2 - Gradient Descent Implementation
# AIML Daily Practice
# ============================================

# Sample Data
X = [1, 2, 3, 4]
Y = [2, 4, 6, 8]

# Initial parameters
w = 0
b = 0

alpha = 0.01   # learning rate
iterations = 1000


# Prediction
def predict(x, w, b):
    return w * x + b


# Cost function
def compute_cost(X, Y, w, b):
    total_error = 0
    m = len(X)

    for i in range(m):
        y_pred = predict(X[i], w, b)
        total_error += (y_pred - Y[i]) ** 2

    return total_error / (2 * m)


# Gradient Descent
def gradient_descent(X, Y, w, b, alpha, iterations):
    m = len(X)

    for _ in range(iterations):
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            y_pred = predict(X[i], w, b)
            error = y_pred - Y[i]

            dj_dw += error * X[i]
            dj_db += error

        dj_dw = dj_dw / m
        dj_db = dj_db / m

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b


# Run Gradient Descent
w, b = gradient_descent(X, Y, w, b, alpha, iterations)

print("Optimized w:", w)
print("Optimized b:", b)
print("Final Cost:", compute_cost(X, Y, w, b))