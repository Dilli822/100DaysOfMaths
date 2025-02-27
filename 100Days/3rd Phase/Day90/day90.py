import numpy as np
import matplotlib.pyplot as plt

def plot_matrix_with_numbers(ax, matrix, title):
    """Plots a matrix with color and overlays numerical values."""
    cax = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_title(title)

    # Display numbers
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha='center', va='center', color='black')

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))

# Generate random matrices
np.random.seed(0)
A_non_singular = np.random.randint(1, 10, (3, 3))  # Full rank
A_singular = A_non_singular.copy()
A_singular[2] = A_singular[1]  # Make two rows identical (rank deficiency)

# Compute ranks
rank_non_singular = np.linalg.matrix_rank(A_non_singular)
rank_singular = np.linalg.matrix_rank(A_singular)

# Plot matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

plot_matrix_with_numbers(axes[0], A_non_singular, f"Non-Singular Matrix\nRank = {rank_non_singular}")
plot_matrix_with_numbers(axes[1], A_singular, f"Singular Matrix\nRank = {rank_singular}")

plt.tight_layout()
plt.show()


# Define augmented matrices
A1 = np.array([[1, 2, -1], [2, 4, -2], [3, 6, -3]])  # Singular (dependent rows)
b1 = np.array([[3], [6], [9]])  # Matches dependency → Infinite solutions

A2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Singular (rank < variables)
b2 = np.array([[3], [2], [1]])  # Doesn't match → No solution

aug1 = np.hstack([A1, b1])
aug2 = np.hstack([A2, b2])

# Compute ranks
rank_A1, rank_aug1 = np.linalg.matrix_rank(A1), np.linalg.matrix_rank(aug1)
rank_A2, rank_aug2 = np.linalg.matrix_rank(A2), np.linalg.matrix_rank(aug2)

# Plot augmented matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

plot_matrix_with_numbers(axes[0], aug1, f"Consistent (Infinite Solutions)\nRank A={rank_A1}, Rank Aug={rank_aug1}")
plot_matrix_with_numbers(axes[1], aug2, f"Inconsistent (No Solution)\nRank A={rank_A2}, Rank Aug={rank_aug2}")

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu

# Example matrix
A = np.array([[2, 1, -1, 8],
              [-3, -1, 2, -11],
              [-2, 1, 2, -3]])

# Compute REF (Upper triangular part from LU decomposition)
P, L, U = lu(A)

# Compute RREF
from sympy import Matrix
rref_A, _ = Matrix(A).rref()

# Convert to numpy array
rref_A = np.array(rref_A).astype(float)

# Plot matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, mat, title in zip(axes, 
                          [A, U, rref_A], 
                          ["Original Matrix", "Row Echelon Form (REF)", "Reduced Row Echelon Form (RREF)"]):
    ax.imshow(mat, cmap="coolwarm", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(A.shape[1]))
    ax.set_yticks(range(A.shape[0]))

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu
from sympy import Matrix

def plot_matrix_with_numbers(ax, matrix, title):
    """Plots a matrix with color and overlays numerical values."""
    cax = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_title(title)

    # Display numbers
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha='center', va='center', color='black')

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))

# Example matrix
A = np.array([[2, 1, -1, 8],
              [-3, -1, 2, -11],
              [-2, 1, 2, -3]])

# Compute REF (Upper triangular part from LU decomposition)
P, L, U = lu(A)

# Compute RREF
rref_A, _ = Matrix(A).rref()
rref_A = np.array(rref_A).astype(float)

# Plot matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

plot_matrix_with_numbers(axes[0], A, "Original Matrix")
plot_matrix_with_numbers(axes[1], U, "Row Echelon Form (REF)")
plot_matrix_with_numbers(axes[2], rref_A, "Reduced Row Echelon Form (RREF)")

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import Matrix, symbols, Eq, solve

# 1. Visualizing a System of Linear Equations in 3D
def plot_3d_system():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
    
    # Define the three planes
    z1 = (10 - x - 2*y) / 3  # x + 2y + 3z = 10
    z2 = (4 - 2*x - 6*y) / 12  # 2x + 6y + 12z = 4
    z3 = (8 - 4*x + 8*y) / 4  # 4x - 8y + 4z = 8
    
    ax.plot_surface(x, y, z1, alpha=0.5, color='blue', edgecolor='k')
    ax.plot_surface(x, y, z2, alpha=0.5, color='red', edgecolor='k')
    ax.plot_surface(x, y, z3, alpha=0.5, color='green', edgecolor='k')
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('System of Linear Equations in 3D')
    
    plt.show()

# 2. Visualizing Matrix Rank in 3D
def plot_matrix_rank():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define vectors as columns of a matrix
    M = np.array([[1, 2, 3], [2, 6, 12], [4, -8, 4]])
    rank = np.linalg.matrix_rank(M)
    
    origin = np.zeros((3, 3))  # Origin for vectors
    ax.quiver(origin[0], origin[1], origin[2], M[:, 0], M[:, 1], M[:, 2], color=['r', 'g', 'b'], linewidth=2)
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'Matrix Rank Visualization (Rank = {rank})')
    
    plt.show()

# Run visualizations
plot_3d_system()
plot_matrix_rank()
