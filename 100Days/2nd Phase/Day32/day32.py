import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix

# Function to visualize the matrix at each step of row reduction
def plot_matrix(matrix, title):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap="Blues")
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='red')
    plt.title(title)
    plt.colorbar(cax)
    plt.show()

# Function to perform row reduction and plot steps
def row_reduction_visualization(A):
    print("Initial Matrix:")
    plot_matrix(A, "Initial Matrix")
    
    matrix = Matrix(A)
    echelon_form = matrix.rref()[0]
    
    print("Row Echelon Form:")
    plot_matrix(np.array(echelon_form).astype(np.float64), "Row Echelon Form (REF)")

    reduced_echelon_form = echelon_form.rref()[0]
    print("Reduced Row Echelon Form (RREF):")
    plot_matrix(np.array(reduced_echelon_form).astype(np.float64), "Reduced Row Echelon Form (RREF)")

# Function to visualize vector addition
def vector_addition(v1, v2):
    result = v1 + v2
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v1')
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='green', label='v2')
    plt.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy', scale=1, color='red', label='v1 + v2')
    plt.xlim(-2, 6)
    plt.ylim(-2, 6)
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Function for scalar multiplication visualization
def scalar_multiply(v, scalar):
    result = scalar * v
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v')
    plt.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy', scale=1, color='red', label=f'{scalar} * v')
    plt.xlim(-2, 6)
    plt.ylim(-2, 6)
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Function to check if two vectors are equal
def check_vector_equality(v1, v2):
    if np.array_equal(v1, v2):
        print(f"The vectors {v1} and {v2} are equal.")
    else:
        print(f"The vectors {v1} and {v2} are not equal.")

# Function to show general solution of a system of equations
def general_solution(A, b):
    matrix = Matrix(A)
    augmented_matrix = matrix.row_join(Matrix(b))
    print("Augmented Matrix [A|b]:")
    plot_matrix(np.array(augmented_matrix).astype(np.float64), "Augmented Matrix [A|b]")
    rref_matrix, pivot_columns = augmented_matrix.rref()
    print("Reduced Row Echelon Form of Augmented Matrix [A|b]:")
    plot_matrix(np.array(rref_matrix).astype(np.float64), "RREF of Augmented Matrix [A|b]")
    return rref_matrix

# Example usage

# Define a matrix for row reduction visualization
A = np.array([[1, 2, 3],
              [2, 4, 6],
              [3, 6, 9]])

# Row reduction and echelon form visualization
row_reduction_visualization(A)

# Define vectors for vector operations visualization
v1 = np.array([1, 2])
v2 = np.array([2, 1])

# Vector addition visualization
vector_addition(v1, v2)

# Scalar multiplication visualization
scalar_multiply(v1, 2)

# Check if two vectors are equal
check_vector_equality(v1, v2)

# General solution example for a system of equations
A_system = np.array([[1, 2, 3],
                     [2, 4, 6],
                     [1, 1, 1]])

b_system = np.array([6, 12, 4])
general_solution(A_system, b_system)
