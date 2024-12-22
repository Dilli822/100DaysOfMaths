import numpy as np
import matplotlib.pyplot as plt

def draw_matrix(ax, matrix, title):
    """
    Draws a matrix as a grid with element values annotated.
    """
    rows, cols = matrix.shape
    
    # Draw the grid
    for i in range(rows + 1):
        ax.plot([0, cols], [-i, -i], color='black', linewidth=0.5)  # Horizontal lines
    for j in range(cols + 1):
        ax.plot([j, j], [0, -rows], color='black', linewidth=0.5)  # Vertical lines
    
    # Annotate the matrix elements
    for i in range(rows):
        for j in range(cols):
            ax.text(j + 0.5, -i - 0.5, f"{matrix[i, j]:.1f}", ha='center', va='center', fontsize=10)
    
    # Set title and limits
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, cols)
    ax.set_ylim(-rows, 0)
    ax.axis('off')  # Hide axes

# Define matrices
A = np.array([[2, 4], [1, 3]])
B = np.array([[1, 2], [3, 4]])
scalar = 2

# Matrix operations
sum_matrix = A + B
scalar_matrix = scalar * A
transpose_A = A.T
product_AB = A @ B

# Create a figure for visualization
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Original matrices
draw_matrix(axs[0, 0], A, "Matrix A")
draw_matrix(axs[0, 1], B, "Matrix B")

# Matrix addition
draw_matrix(axs[0, 2], sum_matrix, "A + B")

# Scalar multiplication
draw_matrix(axs[1, 0], scalar_matrix, f"{scalar} * A")

# Transpose
draw_matrix(axs[1, 1], transpose_A, "Aᵀ")

# Matrix multiplication
draw_matrix(axs[1, 2], product_AB, "A @ B")

# Adjust layout and display
plt.tight_layout()
plt.show()

def draw_matrix(ax, matrix, title):
    """
    Draws a matrix as a grid with element values annotated.
    """
    rows, cols = matrix.shape
    
    # Draw the grid
    for i in range(rows + 1):
        ax.plot([0, cols], [-i, -i], color='black', linewidth=0.5)  # Horizontal lines
    for j in range(cols + 1):
        ax.plot([j, j], [0, -rows], color='black', linewidth=0.5)  # Vertical lines
    
    # Annotate the matrix elements
    for i in range(rows):
        for j in range(cols):
            ax.text(j + 0.5, -i - 0.5, f"{matrix[i, j]:.1f}", ha='center', va='center', fontsize=10)
    
    # Set title and limits
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, cols)
    ax.set_ylim(-rows, 0)
    ax.axis('off')  # Hide axes

# Define matrices
A = np.array([[2, 4], [1, 3]])
B = np.array([[1, 2], [3, 4]])

# Transpose theorems
double_transpose = A.T.T  # (Aᵀ)ᵀ = A
sum_transpose = (A + B).T  # (A + B)ᵀ
scalar_transpose = (2 * A).T  # (cA)ᵀ
product_transpose = (A @ B).T  # (A @ B)ᵀ
reverse_product_transpose = B.T @ A.T  # Bᵀ @ Aᵀ

# Create subplots for visualization
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Original matrix and double transpose
draw_matrix(axs[0, 0], A, "Matrix A")
draw_matrix(axs[0, 1], A.T, "Transpose of A (Aᵀ)")
draw_matrix(axs[0, 2], double_transpose, "Double Transpose (Aᵀ)ᵀ = A")

# Transpose of sum and scalar multiplication
draw_matrix(axs[1, 0], sum_transpose, "Transpose of Sum (A + B)ᵀ")
draw_matrix(axs[1, 1], scalar_transpose, "Transpose of Scalar Mult. (cA)ᵀ")

# Transpose of product and reverse product theorem
draw_matrix(axs[1, 2], reverse_product_transpose, "Bᵀ @ Aᵀ = (A @ B)ᵀ")

# Adjust layout and display
plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

def plot_matrix_3d(ax, matrix, title, color='blue', transpose=False):
    """
    Plots a matrix in 3D space with coordinates (row, column, value).
    """
    rows, cols = matrix.shape
    x, y, z = [], [], []
    
    # Prepare coordinates
    for i in range(rows):
        for j in range(cols):
            xi, yi = (j, i) if not transpose else (i, j)  # Transpose swaps rows and columns
            x.append(xi)
            y.append(yi)
            z.append(matrix[i, j])
            ax.text(xi, yi, matrix[i, j], f"{matrix[i, j]:.1f}", color='black', fontsize=10)
    
    # Plot points
    ax.scatter(x, y, z, c=color, s=50)
    ax.plot_trisurf(x, y, z, alpha=0.3, color=color)
    
    # Set axes labels and title
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.set_zlabel("Values")
    ax.grid(True)

# Define matrices
A = np.array([[2, 4], [1, 3]])
B = np.array([[1, 2], [3, 4]])

# Transpose operations
A_T = A.T
B_T = B.T

# Create a figure and 3D axes
fig = plt.figure(figsize=(12, 10))

# Plot original matrices in 3D
ax1 = fig.add_subplot(221, projection='3d')
plot_matrix_3d(ax1, A, "Matrix A (3D Space)", color='blue')

ax2 = fig.add_subplot(222, projection='3d')
plot_matrix_3d(ax2, B, "Matrix B (3D Space)", color='green')

# Plot transposed matrices in 3D
ax3 = fig.add_subplot(223, projection='3d')
plot_matrix_3d(ax3, A_T, "Transpose of A (Aᵀ)", color='red', transpose=True)

ax4 = fig.add_subplot(224, projection='3d')
plot_matrix_3d(ax4, B_T, "Transpose of B (Bᵀ)", color='purple', transpose=True)

# Adjust layout and display
plt.tight_layout()
plt.show()
