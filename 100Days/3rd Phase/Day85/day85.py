import numpy as np
import matplotlib.pyplot as plt

# Generate sample data for regression
np.random.seed(42)
X_reg = np.linspace(0, 10, 20)
y_reg = 2 * X_reg + 3 + np.random.randn(20) * 2  # y = 2x + 3 + noise

# Fit a perceptron-like linear model (for regression)
w1, b = np.polyfit(X_reg, y_reg, 1)  # Linear fit
