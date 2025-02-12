import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron

# Generate Gaussian-distributed data for two classes
np.random.seed(42)
n_samples = 200
mean1 = [2, 2, 2]
mean2 = [-2, -2, -2]
cov = np.eye(3)  # Identity matrix (equal covariance)

class1 = np.random.multivariate_normal(mean1, cov, n_samples)
class2 = np.random.multivariate_normal(mean2, cov, n_samples)

X = np.vstack((class1, class2))
y = np.hstack((np.ones(n_samples), -1 * np.ones(n_samples)))

# Train Perceptron
perceptron = Perceptron(max_iter=1000)
perceptron.fit(X, y)
w = perceptron.coef_[0]
b = perceptron.intercept_[0]

# Function to calculate decision boundary
def decision_boundary(x, y, w, b):
    return (-w[0] * x - w[1] * y - b) / w[2]

# Create 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot data points
ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c='r', label="Class 1")
ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c='b', label="Class 2")

# Generate decision boundary plane
xx, yy = np.meshgrid(np.linspace(-4, 4, 20), np.linspace(-4, 4, 20))
zz = decision_boundary(xx, yy, w, b)
ax.plot_surface(xx, yy, zz, color='g', alpha=0.5, label="Perceptron Decision Boundary")

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title("Perceptron vs. Bayes Decision Boundary in Gaussian Environment")
plt.legend()
plt.show()
