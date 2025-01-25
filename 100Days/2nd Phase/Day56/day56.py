import matplotlib.pyplot as plt

# Random variable values and probabilities
X = [0, 1, 2]
P_X = [0.25, 0.5, 0.25]

# Bar chart
plt.bar(X, P_X, color='skyblue')
plt.xlabel('Number of Heads (X)')
plt.ylabel('Probability P(X)')
plt.title('Probability Distribution of Random Variable X')
plt.show()


from scipy.stats import binom
import numpy as np

n, p = 5, 0.5  # 5 trials, 50% success probability
x = np.arange(0, n+1)  # Possible outcomes
P_X = binom.pmf(x, n, p)  # Probabilities

# Bar chart
plt.bar(x, P_X, color='salmon')
plt.xlabel('Number of Successes (X)')
plt.ylabel('Probability P(X)')
plt.title('Binomial Distribution (n=5, p=0.5)')
plt.show()


from matplotlib_venn import venn2

# Venn Diagram
venn = venn2(subsets=(30, 20, 10), set_labels=('Cloudy', 'Rain'))
plt.title('Conditional Probability: P(Rain | Cloudy)')
plt.show()

# Plotting Bayes' Theorem components visually

# Define probabilities
P_A = 0.3  # Prior probability of A
P_B_given_A = 0.8  # Likelihood (Probability of B given A)
P_B = 0.5  # Total probability of B

# Compute posterior using Bayes' theorem
P_A_given_B = (P_B_given_A * P_A) / P_B

# Create a bar plot to represent the probabilities
labels = ['P(A)', 'P(B|A)', 'P(B)', 'P(A|B)']
probabilities = [P_A, P_B_given_A, P_B, P_A_given_B]
colors = ['#FF9999', '#FFCC99', '#99CCFF', '#66B3FF']

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, probabilities, color=colors, alpha=0.7, edgecolor='black')
plt.title("Bayes' Theorem Visualization", fontsize=14)
plt.ylim(0, 1)
plt.ylabel("Probability")
plt.xlabel("Components of Bayes' Theorem")

# Add probability values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.2f}', ha='center', fontsize=10)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Only using the first two features for 2D
y = iris.target

# Fit the Naive Bayes model
model = GaussianNB()
model.fit(X, y)

# Visualize the Gaussian distribution for each class
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict the likelihood for each feature given the class
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape + (3,))

# Plot the likelihood for each class
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z[:, :, 0], alpha=0.3, cmap='Reds')
plt.contourf(xx, yy, Z[:, :, 1], alpha=0.3, cmap='Blues')
plt.contourf(xx, yy, Z[:, :, 2], alpha=0.3, cmap='Greens')

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap='viridis')
plt.title("Naive Bayes Theorem: Feature Likelihoods (2D Gaussian)")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()




import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.stats import binom
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# Create the plot window
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# -------------------- Binomial Distribution (3D Bar Chart Animation) --------------------
n, p = 5, 0.5  # 5 trials, 50% success probability
x = np.arange(0, n+1)  # Possible outcomes
P_X_binom = binom.pmf(x, n, p)  # Probabilities

def update_binomial_3d(frame):
    ax.cla()  # Clear axis
    ax.bar3d(x, np.zeros_like(x), np.zeros_like(x), 1, 1, P_X_binom * (frame / 100), color='salmon', alpha=0.7)
    ax.set_xlabel('Number of Successes (X)')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability P(X)')
    ax.set_title('Binomial Distribution (n=5, p=0.5)')

ani_binomial = FuncAnimation(fig, update_binomial_3d, frames=100, repeat=True)

# -------------------- Naive Bayes Gaussian Distribution (3D Contour Plot Animation) --------------------
iris = datasets.load_iris()
X_iris = iris.data[:, :3]  # Using the first three features for 3D
y_iris = iris.target
model = GaussianNB()
model.fit(X_iris, y_iris)

x_min, x_max = X_iris[:, 0].min() - 1, X_iris[:, 0].max() + 1
y_min, y_max = X_iris[:, 1].min() - 1, X_iris[:, 1].max() + 1
z_min, z_max = X_iris[:, 2].min() - 1, X_iris[:, 2].max() + 1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2), np.arange(z_min, z_max, 0.2))

def update_naive_bayes_3d(frame):
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    Z = Z.reshape(xx.shape + (3,))
    
    ax.cla()  # Clear axis
    ax.contourf(xx, yy, zz, Z[:, :, frame], cmap='viridis', alpha=0.7)
    ax.scatter(X_iris[:, 0], X_iris[:, 1], X_iris[:, 2], c=y_iris, s=30, edgecolor='k', cmap='viridis')
    ax.set_title("Naive Bayes Theorem: Feature Likelihoods (3D Gaussian)")
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_zlabel('Petal Length')

ani_naive_bayes_3d = FuncAnimation(fig, update_naive_bayes_3d, frames=3, repeat=True)

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.stats import binom
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# Create the plot window
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# -------------------- Binomial Distribution (3D Bar Chart Animation) --------------------
n, p = 5, 0.5  # 5 trials, 50% success probability
x = np.arange(0, n+1)  # Possible outcomes
P_X_binom = binom.pmf(x, n, p)  # Probabilities

def update_binomial_3d(frame):
    ax.cla()  # Clear axis
    ax.bar3d(x, np.zeros_like(x), np.zeros_like(x), 1, 1, P_X_binom * (frame / 100), color='lightcoral', alpha=0.7)
    ax.set_xlabel('Number of Successes (X)')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability P(X)')
    ax.set_title('Binomial Distribution (n=5, p=0.5)')
    ax.set_zlim(0, 1)

ani_binomial = FuncAnimation(fig, update_binomial_3d, frames=100, repeat=True)

# -------------------- Naive Bayes Gaussian Distribution (3D Contour Plot Animation) --------------------
iris = datasets.load_iris()
X_iris = iris.data[:, :3]  # Using the first three features for 3D
y_iris = iris.target
model = GaussianNB()
model.fit(X_iris, y_iris)

x_min, x_max = X_iris[:, 0].min() - 1, X_iris[:, 0].max() + 1
y_min, y_max = X_iris[:, 1].min() - 1, X_iris[:, 1].max() + 1
z_min, z_max = X_iris[:, 2].min() - 1, X_iris[:, 2].max() + 1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2), np.arange(z_min, z_max, 0.2))

def update_naive_bayes_3d(frame):
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    Z = Z.reshape(xx.shape + (3,))  # Reshape to match the shape of the grid
    
    # Select the appropriate slice of Z for the current frame
    Z_slice = Z[:, :, frame]

    ax.cla()  # Clear axis
    ax.contourf(xx, yy, Z_slice, cmap='viridis', alpha=0.6)
    ax.scatter(X_iris[:, 0], X_iris[:, 1], X_iris[:, 2], c=y_iris, s=30, edgecolor='k', cmap='coolwarm')
    ax.set_title("Naive Bayes Theorem: Feature Likelihoods (3D Gaussian)")
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_zlabel('Petal Length')
    ax.set_zlim(z_min, z_max)

ani_naive_bayes_3d = FuncAnimation(fig, update_naive_bayes_3d, frames=3, repeat=True)

plt.tight_layout()
plt.show()
