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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

# Set random seed for reproducibility
np.random.seed(42)

# Define means and shared covariance matrix
mu1 = np.array([2, 2])   # Mean for class 1
mu2 = np.array([-2, -2]) # Mean for class 2
cov = np.array([[2, 1], [1, 2]])  # Shared covariance matrix

# Generate Gaussian-distributed data points
num_samples = 200
X1 = np.random.multivariate_normal(mu1, cov, num_samples)
X2 = np.random.multivariate_normal(mu2, cov, num_samples)

# Compute decision boundary (linear)
W = np.linalg.inv(cov) @ (mu1 - mu2)  # Compute weight vector
b = -0.5 * (mu1.T @ np.linalg.inv(cov) @ mu1 - mu2.T @ np.linalg.inv(cov) @ mu2) # Compute bias

# Generate grid for contour plot
x_vals = np.linspace(-6, 6, 100)
y_vals = np.linspace(-6, 6, 100)
xx, yy = np.meshgrid(x_vals, y_vals)

# Compute decision boundary (where pX_C1 * P(C1) = pX_C2 * P(C2))
decision_boundary = (W[0] * xx + W[1] * yy + b).reshape(xx.shape)

# Plot the data points
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X1[:, 0], y=X1[:, 1], color="blue", label="Class 1", alpha=0.6)
sns.scatterplot(x=X2[:, 0], y=X2[:, 1], color="red", label="Class 2", alpha=0.6)

# Plot the decision boundary
plt.contour(xx, yy, decision_boundary, levels=[0], colors="black", linewidths=2, label="Bayes Decision Boundary")

# Labels and legend
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Bayes Classifier: Linear Decision Boundary under Gaussian Distribution")
plt.legend()
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import multivariate_normal

# Define parameters
np.random.seed(42)
m1_initial = np.array([0, 0])  # Mean of class 1
m2_initial = np.array([3, 3])  # Mean of class 2
C = np.array([[1, 0.5], [0.5, 1]])  # Shared covariance matrix

# Generate grid for decision boundary
x, y = np.meshgrid(np.linspace(-2, 6, 100), np.linspace(-2, 6, 100))
xy = np.c_[x.ravel(), y.ravel()]

# Function to compute decision boundary
def compute_boundary(m1, m2, C):
    inv_C = np.linalg.inv(C)
    w = np.dot(inv_C, (m1 - m2))
    b = 0.5 * (np.dot(m2.T, np.dot(inv_C, m2)) - np.dot(m1.T, np.dot(inv_C, m1)))
    return w, b

# Initialize figure
fig, ax = plt.subplots()
levels = [0]  # Contour at decision boundary
contour = None

def update(frame):
    global contour
    ax.clear()
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)
    ax.set_title("Bayes Classifier Decision Boundary Animation")
    
    # Update means (simulating movement over time)
    m1 = m1_initial + np.array([frame * 0.1, 0])
    m2 = m2_initial + np.array([-frame * 0.1, 0])
    
    # Compute new decision boundary
    w, b = compute_boundary(m1, m2, C)
    
    # Compute decision scores
    scores = xy @ w + b
    scores = scores.reshape(x.shape)
    
    # Plot decision boundary
    contour = ax.contour(x, y, scores, levels=levels, colors='r')
    
    # Plot class means
    ax.scatter(*m1, color='blue', label='Class 1 Mean')
    ax.scatter(*m2, color='green', label='Class 2 Mean')
    ax.legend()

# Create animation
ani = animation.FuncAnimation(fig, update, frames=20, interval=200)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the two Gaussian distributions
mu1, sigma1 = -1.5, 1  # Mean and standard deviation for class C2
mu2, sigma2 = 1.5, 1   # Mean and standard deviation for class C1

# Define the x-axis range
x = np.linspace(-5, 5, 1000)

# Compute the probability density functions
p_x_C1 = norm.pdf(x, mu1, sigma1)  # P(x | C2)
p_x_C2 = norm.pdf(x, mu2, sigma2)  # P(x | C1)

# Decision boundary (Bayes classifier)
decision_boundary = 0

# Create the plot
plt.figure(figsize=(8, 4))
plt.plot(x, p_x_C1, label=r'$p_X(x | \mathcal{C}_1)$', color='black')
plt.plot(x, p_x_C2, label=r'$p_X(x | \mathcal{C}_2)$', color='black', linestyle='dashed')

# Fill misclassification areas
plt.fill_between(x, p_x_C1, where=(x > decision_boundary), color='lightblue', alpha=0.6)
plt.fill_between(x, p_x_C2, where=(x < decision_boundary), color='gray', alpha=0.6)

# Decision boundary line
plt.axvline(decision_boundary, color='black', linestyle='solid', label="Decision boundary")

# Annotate means
plt.axvline(mu1, color='black', linestyle='dashed')
plt.axvline(mu2, color='black', linestyle='dashed')

plt.text(mu1, -0.02, r'$\mu_1$', ha='center', fontsize=12)
plt.text(mu2, -0.02, r'$\mu_2$', ha='center', fontsize=12)
plt.text(decision_boundary, -0.02, '0', ha='center', fontsize=12)

# Labels and legend
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.title("Two Overlapping Gaussian Distributions")

# Show the plot
plt.show()
