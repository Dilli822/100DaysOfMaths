import numpy as np
import matplotlib.pyplot as plt

# Define two vectors in R^2
v1 = np.array([2, 1])
v2 = np.array([-1, 2])

# Generate linear combinations
t = np.linspace(-2, 2, 10)
combinations = [a * v1 + b * v2 for a in t for b in t]

# Extract x and y coordinates for visualization
x_coords = [v[0] for v in combinations]
y_coords = [v[1] for v in combinations]

# Plot the vectors and combinations
plt.figure(figsize=(8, 8))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
plt.scatter(x_coords, y_coords, alpha=0.5, label='Linear Combinations', color='green')

# Set axes and title
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid()
plt.legend()
plt.title("Linear Combinations of Vectors in R^2")
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Define two vectors in R^3
v1 = np.array([1, 2, 3])
v2 = np.array([-2, 1, 1])

# Generate a grid of linear combinations
t = np.linspace(-1, 1, 10)
s = np.linspace(-1, 1, 10)
T, S = np.meshgrid(t, s)
plane = np.array([T * v1[0] + S * v2[0], T * v1[1] + S * v2[1], T * v1[2] + S * v2[2]])

# Plot the vectors and the spanned plane
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the plane
ax.plot_surface(plane[0], plane[1], plane[2], alpha=0.5, color='cyan')

# Plot the original vectors
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='v1')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='v2')

# Set axes and labels
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
ax.set_title("Spanning Plane in R^3")
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def visualize_ax_equals_b():
    # Define a matrix A and a target vector b
    A = np.array([[2, 1], [1, 3]])
    b = np.array([4, 7])

    # Define a grid of x values
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute the linear transformation Ax
    Ax = np.einsum('ij,jkl->ikl', A, np.array([X1, X2]))

    # Plot
    plt.figure(figsize=(8, 8))
    plt.quiver(0, 0, A[0, 0], A[1, 0], color='r', angles='xy', scale_units='xy', scale=1, label="Column 1 of A")
    plt.quiver(0, 0, A[0, 1], A[1, 1], color='b', angles='xy', scale_units='xy', scale=1, label="Column 2 of A")
    plt.quiver(0, 0, b[0], b[1], color='g', angles='xy', scale_units='xy', scale=1, label="Vector b")
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend()
    plt.title("Transformation of x under A in R^2")
    plt.grid()
    plt.show()

visualize_ax_equals_b()


import numpy as np
import matplotlib.pyplot as plt

def visualize_pivot_positions(matrix):
    def row_reduce_to_ref(matrix):
        """Perform Gaussian elimination to get the matrix in REF."""
        matrix = matrix.astype(float)
        rows, cols = matrix.shape
        pivot_positions = []
        
        for r in range(rows):
            # Find the pivot in the current row
            for c in range(cols):
                if matrix[r, c] != 0:
                    pivot_positions.append((r, c))
                    # Normalize the row
                    matrix[r] /= matrix[r, c]
                    # Eliminate below the pivot
                    for i in range(r + 1, rows):
                        matrix[i] -= matrix[i, c] * matrix[r]
                    break
        return matrix, pivot_positions

    # Perform row reduction and get pivot positions
    ref_matrix, pivots = row_reduce_to_ref(matrix)

    # Plot the original matrix with pivot positions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(matrix, cmap='viridis', alpha=0.8)
    
    for (i, j) in pivots:
        ax.text(j, i, 'P', va='center', ha='center', color='red', fontsize=14, fontweight='bold')

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels([f"Col {i}" for i in range(1, matrix.shape[1] + 1)])
    ax.set_yticklabels([f"Row {i}" for i in range(1, matrix.shape[0] + 1)])
    plt.title("Pivot Positions in Matrix")
    plt.show()

# Example matrix
matrix = np.array([
    [2, 4, 1],
    [1, 3, 0],
    [0, 0, 5]
])

visualize_pivot_positions(matrix)


import numpy as np
import matplotlib.pyplot as plt

def visualize_r2_properties():
    # Define two vectors in R^2
    u = np.array([2, 1])
    v = np.array([1, 3])
    
    # Scalar multiples
    scalars = [-1, -0.5, 0, 0.5, 1, 1.5]
    scalar_vectors = [s * u for s in scalars]
    
    # Plot the vectors
    plt.figure(figsize=(8, 8))
    plt.quiver(0, 0, u[0], u[1], color='r', angles='xy', scale_units='xy', scale=1, label='Vector u')
    plt.quiver(0, 0, v[0], v[1], color='b', angles='xy', scale_units='xy', scale=1, label='Vector v')
    
    # Visualize scalar multiples of u
    for sv in scalar_vectors:
        plt.quiver(0, 0, sv[0], sv[1], color='orange', angles='xy', scale_units='xy', scale=1, alpha=0.5)
    
    # Vector addition
    sum_vector = u + v
    plt.quiver(0, 0, sum_vector[0], sum_vector[1], color='g', angles='xy', scale_units='xy', scale=1, label='u + v')
    
    # Plot settings
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid()
    plt.legend()
    plt.title("Algebraic Properties of R^n (Visualized in R^2)")
    plt.show()

visualize_r2_properties()
