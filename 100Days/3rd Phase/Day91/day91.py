import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_2d_vectors(vectors, labels, colors, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, v in enumerate(vectors):
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=colors[i], label=labels[i])
    
    max_val = max(np.max(np.abs(vectors)), 1) + 1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xticks(range(-int(max_val), int(max_val) + 1))
    ax.set_yticks(range(-int(max_val), int(max_val) + 1))
    ax.set_title(title)
    ax.legend()
    plt.grid()
    plt.show()

def plot_3d_vectors(vectors, labels, colors, title):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i, v in enumerate(vectors):
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=colors[i], label=labels[i])
    
    max_val = max(np.max(np.abs(vectors)), 1) + 1
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(title)
    ax.legend()
    plt.show()

# Example: Basic Vectors
v1 = np.array([3, 2])
v2 = np.array([-2, 3])
plot_2d_vectors([v1, v2], ["Vector 1", "Vector 2"], ['r', 'b'], "Basic 2D Vectors")

# Example: 3D Vectors
v3 = np.array([2, 1, 3])
v4 = np.array([-1, 2, 2])
plot_3d_vectors([v3, v4], ["Vector 1", "Vector 2"], ['g', 'm'], "Basic 3D Vectors")


import numpy as np
import matplotlib.pyplot as plt

# 1. Vectors and Their Properties
def plot_vectors_and_properties():
    # Define vectors
    v1 = np.array([2, 3])
    v2 = np.array([-1, 2])

    # Plot vectors
    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector v1')
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector v2')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.title("Vectors and Their Properties")
    plt.legend()
    plt.show()

# 2. Vector Operations (Multiplication by a Matrix)
def plot_vector_multiplication():
    # Define vector and matrix
    v = np.array([1, 2])
    A = np.array([[2, 0], [0, 2]])  # Scaling matrix

    # Multiply vector by matrix
    v_transformed = A @ v

    # Plot original and transformed vector
    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='Original Vector')
    plt.quiver(0, 0, v_transformed[0], v_transformed[1], angles='xy', scale_units='xy', scale=1, color='b', label='Transformed Vector')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid()
    plt.title("Vector Multiplication by a Matrix")
    plt.legend()
    plt.show()

# 3. Dot Product & Geometric Interpretation
def plot_dot_product():
    # Define vectors
    v1 = np.array([2, 3])
    v2 = np.array([-1, 2])

    # Compute dot product
    dot_product = np.dot(v1, v2)

    # Plot vectors
    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label=f'v1: {v1}')
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label=f'v2: {v2}')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.title(f"Dot Product: {dot_product}")
    plt.legend()
    plt.show()

# 4. Matrices as Linear Transformations
def plot_linear_transformation():
    # Define matrix and vectors
    A = np.array([[1, -1], [1, 1]])  # Rotation matrix
    v = np.array([1, 0])

    # Apply transformation
    v_transformed = A @ v

    # Plot original and transformed vector
    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='Original Vector')
    plt.quiver(0, 0, v_transformed[0], v_transformed[1], angles='xy', scale_units='xy', scale=1, color='b', label='Transformed Vector')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()
    plt.title("Matrices as Linear Transformations")
    plt.legend()
    plt.show()

# 5. Identity Matrix & Matrix Inverse
def plot_identity_and_inverse():
    # Define vector and matrices
    v = np.array([1, 2])
    I = np.eye(2)  # Identity matrix
    A = np.array([[2, 0], [0, 2]])
    A_inv = np.linalg.inv(A)  # Inverse of A

    # Apply transformations
    v_identity = I @ v
    v_inverse = A_inv @ (A @ v)

    # Plot results
    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='Original Vector')
    plt.quiver(0, 0, v_identity[0], v_identity[1], angles='xy', scale_units='xy', scale=1, color='g', label='Identity Transformation')
    plt.quiver(0, 0, v_inverse[0], v_inverse[1], angles='xy', scale_units='xy', scale=1, color='b', label='Inverse Transformation')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.title("Identity Matrix & Matrix Inverse")
    plt.legend()
    plt.show()

# 6. Neural Networks and Matrices
def plot_neural_networks():
    # Simulate a simple neural network transformation
    input_vector = np.array([1, 2])
    weights = np.array([[0.5, -0.5], [0.5, 0.5]])
    output_vector = weights @ input_vector

    # Plot input and output vectors
    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, input_vector[0], input_vector[1], angles='xy', scale_units='xy', scale=1, color='r', label='Input Vector')
    plt.quiver(0, 0, output_vector[0], output_vector[1], angles='xy', scale_units='xy', scale=1, color='b', label='Output Vector')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.title("Neural Networks and Matrices")
    plt.legend()
    plt.show()

# Run all plots
plot_vectors_and_properties()
plot_vector_multiplication()
plot_dot_product()
plot_linear_transformation()
plot_identity_and_inverse()
plot_neural_networks()

import numpy as np
import matplotlib.pyplot as plt

# 1. Vectors and Their Properties
def plot_vectors_and_properties():
    """
    Plot two vectors with illustrative examples (e.g., wind velocity and displacement).
    """
    # Define vectors (e.g., wind velocity and displacement)
    wind_velocity = np.array([3, 1])  # Wind blowing northeast
    displacement = np.array([1, 2])  # Object displacement

    # Plot vectors
    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, wind_velocity[0], wind_velocity[1], angles='xy', scale_units='xy', scale=1, color='r', label='Wind Velocity = [3, 1]')
    plt.quiver(0, 0, displacement[0], displacement[1], angles='xy', scale_units='xy', scale=1, color='b', label='Displacement = [1, 2]')
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.title("Vectors and Their Properties\n(Wind Velocity and Displacement)")
    plt.legend()
    plt.show()

# 2. Vector Operations (Multiplication by a Matrix)
def plot_vector_multiplication():
    """
    Plot vector transformation using matrix multiplication (e.g., scaling a shape).
    """
    # Define a triangle (3 vertices)
    triangle = np.array([[0, 0], [1, 0], [0.5, 1], [0, 0]])  # Closing the triangle

    # Define a scaling matrix
    A = np.array([[2, 0], [0, 2]])  # Scaling by 2x

    # Apply transformation
    triangle_transformed = (A @ triangle.T).T

    # Plot original and transformed triangle
    plt.figure(figsize=(6, 6))
    plt.plot(triangle[:, 0], triangle[:, 1], 'r-o', label='Original Triangle')
    plt.plot(triangle_transformed[:, 0], triangle_transformed[:, 1], 'b-o', label='Scaled Triangle (2x)')
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.title("Vector Multiplication by a Matrix\n(Scaling a Triangle)")
    plt.legend()
    plt.show()

# 3. Dot Product & Geometric Interpretation
def plot_dot_product():
    """
    Plot vectors and compute their dot product (e.g., work done by a force).
    """
    # Define vectors (e.g., force and displacement)
    force = np.array([2, 3])  # Force vector
    displacement = np.array([1, 0])  # Displacement vector

    # Compute dot product (work done)
    work_done = np.dot(force, displacement)

    # Plot vectors
    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, force[0], force[1], angles='xy', scale_units='xy', scale=1, color='r', label=f'Force = [2, 3]')
    plt.quiver(0, 0, displacement[0], displacement[1], angles='xy', scale_units='xy', scale=1, color='b', label=f'Displacement = [1, 0]')
    plt.xlim(-1, 3)
    plt.ylim(-1, 4)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.title(f"Dot Product: Work Done = {work_done} Joules")
    plt.legend()
    plt.show()

# 4. Matrices as Linear Transformations
def plot_linear_transformation():
    """
    Plot vector transformation using a rotation matrix (e.g., rotating a shape).
    """
    # Define a square (4 vertices)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])  # Closing the square

    # Define a rotation matrix (45 degrees)
    theta = np.radians(45)
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Apply transformation
    square_transformed = (A @ square.T).T

    # Plot original and transformed square
    plt.figure(figsize=(6, 6))
    plt.plot(square[:, 0], square[:, 1], 'r-o', label='Original Square')
    plt.plot(square_transformed[:, 0], square_transformed[:, 1], 'b-o', label='Rotated Square (45°)')
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.title("Matrices as Linear Transformations\n(Rotating a Square)")
    plt.legend()
    plt.show()

# 5. Identity Matrix & Matrix Inverse
def plot_identity_and_inverse():
    """
    Plot vector transformation using identity and inverse matrices (e.g., scaling and unscaling a shape).
    """
    # Define a rectangle (4 vertices)
    rectangle = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]])  # Closing the rectangle

    # Define scaling matrix and its inverse
    A = np.array([[2, 0], [0, 2]])  # Scaling by 2x
    A_inv = np.linalg.inv(A)  # Inverse of A (scaling by 0.5x)

    # Apply transformations
    rectangle_scaled = (A @ rectangle.T).T
    rectangle_unscaled = (A_inv @ rectangle_scaled.T).T

    # Plot results
    plt.figure(figsize=(6, 6))
    plt.plot(rectangle[:, 0], rectangle[:, 1], 'r-o', label='Original Rectangle')
    plt.plot(rectangle_scaled[:, 0], rectangle_scaled[:, 1], 'b-o', label='Scaled Rectangle (2x)')
    plt.plot(rectangle_unscaled[:, 0], rectangle_unscaled[:, 1], 'g-o', label='Unscaled Rectangle (0.5x)')
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.title("Identity Matrix & Matrix Inverse\n(Scaling and Unscaling a Rectangle)")
    plt.legend()
    plt.show()

# 6. Neural Networks and Matrices
def plot_neural_networks():
    """
    Plot input and output vectors of a simple neural network (e.g., transforming coordinates).
    """
    # Simulate a simple neural network transformation
    input_vector = np.array([1, 2])  # Input coordinates
    weights = np.array([[0.5, -0.5], [0.5, 0.5]])  # Weight matrix
    output_vector = weights @ input_vector  # Transformed coordinates

    # Plot input and output vectors
    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, input_vector[0], input_vector[1], angles='xy', scale_units='xy', scale=1, color='r', label='Input Vector = [1, 2]')
    plt.quiver(0, 0, output_vector[0], output_vector[1], angles='xy', scale_units='xy', scale=1, color='b', label='Output Vector = [0, 1.5]')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.title("Neural Networks and Matrices\n(Transforming Coordinates)")
    plt.legend()
    plt.show()

# Run all plots
plot_vectors_and_properties()
plot_vector_multiplication()
plot_dot_product()
plot_linear_transformation()
plot_identity_and_inverse()
plot_neural_networks()