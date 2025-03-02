import numpy as np
import matplotlib.pyplot as plt

# Define a 2x2 matrix
A = np.array([[4, -2], 
              [1,  1]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Plot eigenvectors
origin = np.array([[0, 0], [0, 0]])  # Origin point

plt.figure(figsize=(6, 6))
plt.axhline(0, color='gray', lw=1)
plt.axvline(0, color='gray', lw=1)

# Plot eigenvectors scaled by eigenvalues
for i in range(2):
    vec = eigenvectors[:, i] * eigenvalues[i]
    plt.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1, color=['r', 'b'], label=f'λ={eigenvalues[i]:.2f}')

plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.legend()
plt.grid()
plt.title("Eigenvectors & Eigenvalues (2D)")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a 3x3 matrix
A = np.array([[3, 1, 1], 
              [1, 3, 1], 
              [1, 1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# 3D Plot
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot eigenvectors
origin = np.zeros((3, 1))
for i in range(3):
    vec = eigenvectors[:, i] * eigenvalues[i]
    ax.quiver(*origin.flatten(), *vec, color=['r', 'b', 'g'][i], label=f'λ={eigenvalues[i]:.2f}')

# Set plot limits and labels
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
ax.set_title("Eigenvectors & Eigenvalues (3D)")

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate synthetic 2D data
np.random.seed(42)
X = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
eigenvectors = pca.components_
eigenvalues = pca.explained_variance_

# Plot original data
plt.figure(figsize=(7, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original Data")

# Plot principal components
origin = np.mean(X, axis=0)  # Mean (center of data)
for i in range(2):
    plt.quiver(*origin, *eigenvectors[i] * eigenvalues[i], angles='xy', scale_units='xy', scale=1.5, color=['r', 'b'], label=f'PC{i+1}')

plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.legend()
plt.title("PCA - Principal Components in 2D")
plt.grid()
plt.show()


from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic 3D data
X, _ = make_blobs(n_samples=200, centers=3, n_features=3, random_state=42)

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3D Plot of original data
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.6)
ax.set_title("Original 3D Data")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")

# 2D Projection plot
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, color='r')
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.title("PCA Projected Data (3D → 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic 3D data
X, _ = make_blobs(n_samples=200, centers=3, n_features=3, random_state=42)

# Apply PCA to reduce from 3D to 2D
pca_3d_to_2d = PCA(n_components=2)
X_2D = pca_3d_to_2d.fit_transform(X)

# Apply PCA to reduce from 2D to 1D (Y projection)
pca_2d_to_1d = PCA(n_components=1)
X_1D = pca_2d_to_1d.fit_transform(X_2D)

# 3D Scatter Plot (Original Data)
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.6)
ax1.set_title("Original 3D Data")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.set_zlabel("Feature 3")

# 2D Scatter Plot (Projection on X-Axis)
ax2 = fig.add_subplot(132)
ax2.scatter(X_2D[:, 0], X_2D[:, 1], alpha=0.6, color='r')
ax2.axhline(0, color='gray', lw=0.5)
ax2.axvline(0, color='gray', lw=0.5)
ax2.set_title("PCA Projection (3D → 2D)")
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.grid()

# 1D Scatter Plot (Projection on Y-Axis)
ax3 = fig.add_subplot(133)
ax3.scatter(X_1D, np.zeros_like(X_1D), alpha=0.6, color='b')
ax3.axhline(0, color='gray', lw=0.5)
ax3.axvline(0, color='gray', lw=0.5)
ax3.set_title("Final Projection (2D → 1D)")
ax3.set_xlabel("Principal Component 1")
ax3.set_yticks([])
ax3.grid()

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define matrix A
A = np.array([[3, 1],
              [0, 2]])

# Compute eigenvalues & eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Generate animation data
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.axhline(0, color='gray', lw=1)
ax.axvline(0, color='gray', lw=1)

# Plot original eigenvectors
eig_vecs = [ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r') for v in eigenvectors.T]

# Animation function
def update(frame):
    global eig_vecs
    for eig_vec in eig_vecs:
        eig_vec.remove()
    
    scaled_vectors = [eigenvectors[:, i] * (1 + frame / 10 * eigenvalues[i]) for i in range(2)]
    
    eig_vecs = [ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r') for v in scaled_vectors]

# Create animation
ani = animation.FuncAnimation(fig, update, frames=20, interval=200)
plt.title("Eigenvector Scaling by Eigenvalues")
plt.show()
