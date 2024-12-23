import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to check if a matrix is invertible
def is_invertible(matrix):
    determinant = np.linalg.det(matrix)
    return determinant != 0, determinant

# Create some example matrices
matrices = {
    "Invertible Matrix": np.array([[2, 3], [1, 4]]),
    "Singular Matrix": np.array([[1, 2], [2, 4]]),
    "Partitioned Matrix": np.block([
        [np.eye(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), np.eye(2)]
    ])
}

# Display properties of matrices
for name, matrix in matrices.items():
    invertible, determinant = is_invertible(matrix)
    print(f"{name}:\n{matrix}")
    print(f"Determinant: {determinant}")
    print(f"Invertible: {invertible}")
    if invertible:
        print(f"Inverse:\n{np.linalg.inv(matrix)}")
    print("\n" + "-"*40 + "\n")

# Generate 3D plot for determinants
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create a range of matrices and calculate determinants
x_vals = np.linspace(-5, 5, 30)
y_vals = np.linspace(-5, 5, 30)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.array([[np.linalg.det(np.array([[x, 1], [y, 1]])) for x in x_vals] for y in y_vals])

# Plot the determinant surface
ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.8)
ax.set_title("Determinant of 2x2 Matrices")
ax.set_xlabel("Matrix Element [0, 0]")
ax.set_ylabel("Matrix Element [1, 0]")
ax.set_zlabel("Determinant Value")
plt.show()
