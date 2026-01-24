"""
=========================================
AIML Daily Practice
Day 01 - Linear Regression (From Scratch)
=========================================

Author: Harsh
Date: 24-01-2026

Description:
This script implements Linear Regression manually
using basic Python and NumPy.

Concepts:
- Hypothesis function
- Mean Squared Error
- Cost calculation

Future Improvements:
- Add gradient descent
- Add visualization

=========================================
"""

"""
Day 01 - Linear Regression (From Scratch)
"""

# ============================================
# Day 1 - Linear Regression (Basic Implementation)
# AIML Daily Practice
# ============================================

# Sample Data
X = [1, 2, 3, 4]
Y = [2, 4, 6, 8]

# Initial parameters
w = 0
b = 0


# Prediction function
def predict(x, w, b):
    return w * x + b


# Cost function (Mean Squared Error)
def compute_cost(X, Y, w, b):
    total_error = 0
    m = len(X)

    for i in range(m):
        y_pred = predict(X[i], w, b)
        total_error += (y_pred - Y[i]) ** 2

    return total_error / (2 * m)


# Print initial cost
print("Initial Cost:", compute_cost(X, Y, w, b))


# Try better parameters
w = 2
b = 0

print("Cost after updating w=2, b=0:", compute_cost(X, Y, w, b))

