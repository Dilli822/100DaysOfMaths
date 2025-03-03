import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import networkx as nx

# 1. Covariance & Variance (Matrix)
data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.8], [0.8, 1]], size=100)
cov_matrix = np.cov(data, rowvar=False)
plt.figure(figsize=(6, 5))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm')
plt.title("Covariance Matrix")
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], c='b', marker='o')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Density')
plt.title("3D Scatter Plot - Variance Visualization")
plt.show()

# 2. PCA Detail
pca = PCA(n_components=2)
pca.fit(data)
eigenvectors = pca.components_
mean = np.mean(data, axis=0)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
for vec, eig in zip(eigenvectors, pca.explained_variance_):
    plt.arrow(mean[0], mean[1], vec[0] * eig, vec[1] * eig, color='r', width=0.02)
plt.title("PCA: Eigenvectors and Eigenvalues")
plt.show()

# 3. Markov Model
graph = nx.DiGraph()
states = ['A', 'B', 'C']
edges = [('A', 'B', 0.3), ('A', 'C', 0.7), ('B', 'A', 0.5), ('B', 'C', 0.5), ('C', 'A', 0.6), ('C', 'B', 0.4)]
graph.add_weighted_edges_from(edges)
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
edge_labels = {(u, v): f'{w:.2f}' for u, v, w in edges}
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
plt.title("Markov Model - State Transition Graph")
plt.show()

# 4. Unsupervised ML: Anomaly Detection
x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
pos = np.dstack((x, y))
rv = multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]])
z = rv.pdf(pos)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
plt.title("3D Density Estimation for Anomaly Detection")
plt.show()


import networkx as nx
import matplotlib.pyplot as plt

# Define Markov transition probabilities
transition_matrix = {
    "Raining": {"Raining": 0.7, "Not Raining": 0.3},
    "Not Raining": {"Raining": 0.4, "Not Raining": 0.6},
}

# Create directed graph
G = nx.DiGraph()

# Add edges with transition probabilities
for state, transitions in transition_matrix.items():
    for next_state, prob in transitions.items():
        G.add_edge(state, next_state, weight=prob)

# Position nodes
pos = nx.spring_layout(G, seed=42)

# Draw graph
plt.figure(figsize=(7, 5))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', edge_color='black', font_size=12)

# Add edge labels (transition probabilities)
edge_labels = {(u, v): f"{p:.1f}" for u, v, p in G.edges(data="weight")}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

plt.title("Markov Model: Rain & No Rain Transition")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate synthetic 2D data with correlation
np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 2]]  # Covariance matrix
X = np.random.multivariate_normal(mean, cov, 200)

# Apply PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# Get principal components
pc1, pc2 = pca.components_

# Plot original data
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original Data")

# Plot principal component vectors
origin = np.mean(X, axis=0)  # Mean of data
ax.quiver(*origin, *pc1, color='r', scale=3, label="PC1 (Most Variance)")
ax.quiver(*origin, *pc2, color='b', scale=3, label="PC2 (Least Variance)")

# Project data onto the first principal component
X_projected = X @ pc1.reshape(-1, 1) @ pc1.reshape(1, -1)
ax.scatter(X_projected[:, 0], X_projected[:, 1], color="orange", alpha=0.6, label="Projected Data (PC1)")

# Labels & legend
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("PCA: Projection onto Principal Components")
ax.legend()
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Generate synthetic 3D data
np.random.seed(42)
mean = [0, 0, 0]
cov = [[3, 2, 1], [2, 2, 1], [1, 1, 1]]  # Covariance matrix
X = np.random.multivariate_normal(mean, cov, 200)

# Apply PCA to reduce from 3D to 2D
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X)

# Set up the figure for animation
fig = plt.figure(figsize=(12, 6))

# Subplot 1: 3D Scatter Plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Original 3D Data")
ax1.set_xlabel("X-axis")
ax1.set_ylabel("Y-axis")
ax1.set_zlabel("Z-axis")
scatter3D = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.6, color='blue')

# Subplot 2: 2D Scatter Plot (After PCA)
ax2 = fig.add_subplot(122)
ax2.set_title("Projected 2D Data")
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
scatter2D = ax2.scatter([], [], alpha=0.6, color="red")

# Animation update function
def update(frame):
    ax2.clear()
    ax2.set_title("Projected 2D Data")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    ax2.scatter(X_2D[:frame, 0], X_2D[:frame, 1], alpha=0.6, color="red")

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(X), interval=50, repeat=True)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.mixture import GaussianMixture

# Generate normal data
np.random.seed(42)
X_normal = np.random.randn(200, 2) * 0.75 + np.array([2, 2])

# Generate anomalies (outliers)
X_anomalies = np.random.uniform(low=-3, high=6, size=(15, 2))

# Combine data
X = np.vstack([X_normal, X_anomalies])

# Fit Gaussian Mixture Model (GMM) for anomaly detection
gmm = GaussianMixture(n_components=1, covariance_type='full')
gmm.fit(X_normal)  # Train only on normal data
probabilities = gmm.score_samples(X)  # Get probability densities

# Set a threshold for anomalies
threshold = np.percentile(probabilities, 5)  # Lowest 5% are anomalies
anomalies = probabilities < threshold

# Setup figure for animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-3, 6)
ax.set_ylim(-3, 6)
ax.set_title("Anomaly Detection with Clustering")
ax.set_xlabel("Feature X1")
ax.set_ylabel("Feature X2")

# Initialize scatter plots
scatter_normal = ax.scatter([], [], color='blue', label='Normal Data', alpha=0.6)
scatter_anomaly = ax.scatter([], [], color='red', label='Anomalies', marker='x', s=100)

# Animation update function
def update(frame):
    if frame < len(X_normal):
        scatter_normal.set_offsets(X_normal[:frame])  # Normal data points gradually appear
    else:
        scatter_anomaly.set_offsets(X_anomalies[:frame - len(X_normal)])  # Anomalies appear later
    return scatter_normal, scatter_anomaly

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(X), interval=50, repeat=True)

plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define a 2x2 transformation matrix
A = np.array([[2, 1], [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Create initial unit vectors (standard basis)
grid = np.array([[1, 0], [0, 1]])

# Apply transformation matrix
transformed_grid = A @ grid

# Setup figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_title("Effect of Eigenvalues & Eigenvectors on Transformation")
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.grid(True)

# Plot original basis vectors
original_vectors = [ax.quiver(0, 0, grid[0, i], grid[1, i], angles='xy', scale_units='xy', scale=1, color='blue') for i in range(2)]

# Plot eigenvectors
eigenvector_plots = [ax.quiver(0, 0, eigenvectors[0, i], eigenvectors[1, i], angles='xy', scale_units='xy', scale=1, color='green', alpha=0.5) for i in range(2)]

# Plot transformed vectors
transformed_vectors = [ax.quiver(0, 0, transformed_grid[0, i], transformed_grid[1, i], angles='xy', scale_units='xy', scale=1, color='red') for i in range(2)]

# Animation function
def update(frame):
    # Scale eigenvectors with eigenvalues over time
    scaled_eigenvectors = eigenvectors * (eigenvalues ** (frame / 20))
    
    for i in range(2):
        transformed_vectors[i].set_UVC(scaled_eigenvectors[0, i], scaled_eigenvectors[1, i])

    return transformed_vectors

# Create animation
ani = animation.FuncAnimation(fig, update, frames=20, interval=100, repeat=True)

plt.show()
