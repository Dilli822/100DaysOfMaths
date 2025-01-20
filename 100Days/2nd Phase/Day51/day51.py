

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Function to plot vectors in 2D
# def plot_vectors_2d(vectors, colors, labels, ax=None):
#     if ax is None:
#         ax = plt.gca()
#     for vector, color, label in zip(vectors, colors, labels):
#         ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=color)
#         ax.text(vector[0], vector[1], label, fontsize=12, color=color)
#     ax.set_xlim(-5, 5)
#     ax.set_ylim(-5, 5)
#     ax.set_aspect('equal', 'box')
#     ax.grid(True)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.legend(loc="upper right")

# # Function to plot vectors in 3D
# def plot_vectors_3d(vectors, colors, labels, ax=None):
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#     for vector, color, label in zip(vectors, colors, labels):
#         ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=color, length=1, normalize=True)
#         ax.text(vector[0], vector[1], vector[2], label, fontsize=12, color=color)
#     ax.set_xlim([-5, 5])
#     ax.set_ylim([-5, 5])
#     ax.set_zlim([-5, 5])
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.grid(True)
#     ax.legend(loc="upper left")

# # Basis Vectors and Spanning Example (2D)
# basis_vectors_2d = np.array([[1, 0], [0, 1]])  # Standard basis in 2D
# labels_2d = ['e1', 'e2']
# colors_2d = ['r', 'b']

# # Spanning Example (2D)
# vector_2d = np.array([3, 4])  # Vector in 2D
# labels_2d_spanning = ['v', 'e1', 'e2']
# colors_2d_spanning = ['g', 'r', 'b']

# # Plot 2D
# fig, ax = plt.subplots()
# plot_vectors_2d([basis_vectors_2d[0], basis_vectors_2d[1], vector_2d], colors_2d_spanning, labels_2d_spanning, ax)
# plt.title("Basis Vectors and Spanning in 2D")
# plt.legend(["Basis Vector e1 (Red)", "Basis Vector e2 (Blue)", "Spanning Vector v (Green)"])
# plt.show()

# # Basis Vectors and Spanning Example (3D)
# basis_vectors_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Standard basis in 3D
# vector_3d = np.array([2, 3, 5])  # Vector in 3D
# labels_3d = ['e1', 'e2', 'e3', 'v']
# colors_3d = ['r', 'b', 'g', 'y']

# # Plot 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plot_vectors_3d([basis_vectors_3d[0], basis_vectors_3d[1], basis_vectors_3d[2], vector_3d], colors_3d, labels_3d, ax)
# plt.title("Basis Vectors and Spanning in 3D")
# ax.legend(["Basis Vector e1 (Red)", "Basis Vector e2 (Blue)", "Basis Vector e3 (Green)", "Spanning Vector v (Yellow)"])
# plt.show()

# # Eigenvalues and Eigenvectors Example
# A = np.array([[4, -2], [1, 1]])
# eigenvalues, eigenvectors = np.linalg.eig(A)
# print(f"Eigenvalues: {eigenvalues}")
# print(f"Eigenvectors: \n{eigenvectors}")

# # Plot Eigenvectors in 2D
# fig, ax = plt.subplots()
# plot_vectors_2d([eigenvectors[:, 0], eigenvectors[:, 1]], ['r', 'b'], ['Eigenvector 1', 'Eigenvector 2'], ax)
# plt.title("Eigenvectors of Matrix A")
# ax.legend(["Eigenvector 1 (Red)", "Eigenvector 2 (Blue)"])
# plt.show()

# # Linear Independence Example
# vectors = np.array([[1, 2], [2, 4]])  # These are linearly dependent
# rank = np.linalg.matrix_rank(vectors)
# print(f"Rank of the matrix: {rank}")

# # Change of Basis Example
# # Transforming vector [1, 1] from basis e1, e2 to a new basis e1' = [1, 1] and e2' = [1, -1]
# new_basis = np.array([[1, 1], [1, -1]])  # Change of basis matrix
# vector_to_transform = np.array([1, 1])

# # In the new basis
# transformed_vector = np.linalg.inv(new_basis).dot(vector_to_transform)
# print(f"Transformed vector: {transformed_vector}")

# # Null Space and Column Space Example
# A = np.array([[1, 2], [3, 4], [5, 6]])
# null_space = np.linalg.svd(A)[2].T[:, 2:]
# print(f"Null space of A: \n{null_space}")

# # Kernel and Range of Linear Transformation
# # Kernel is the null space, and range is the column space of matrix A
# column_space = np.linalg.matrix_rank(A)
# print(f"Column space rank of matrix A: {column_space}")

# # Plot Null Space and Column Space (2D)
# fig, ax = plt.subplots()

# # Check if the null space is empty
# if null_space.size > 0:
#     plot_vectors_2d([null_space[0]], ['r'], ['Null Space'], ax)
# else:
#     print("The null space is trivial (only the zero vector) and cannot be plotted.")

# plot_vectors_2d([A[:, 0], A[:, 1]], ['g', 'b'], ['Column Space 1', 'Column Space 2'], ax)
# plt.title("Null Space and Column Space (2D)")
# ax.legend(["Null Space (Red)", "Column Space 1 (Green)", "Column Space 2 (Blue)"])
# plt.show()

# # Linearly Independent and Dependent Vectors (3D)
# independent_vectors_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Independent vectors in 3D
# dependent_vectors_3d = np.array([[1, 2, 3], [2, 4, 6]])  # Dependent vectors in 3D (multiple of each other)
# labels_independent_3d = ['e1', 'e2', 'e3']
# labels_dependent_3d = ['v1', 'v2']
# colors_independent_3d = ['r', 'b', 'g']
# colors_dependent_3d = ['y', 'c']

# # Plotting Linearly Independent and Dependent Vectors (3D)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plot_vectors_3d([independent_vectors_3d[0], independent_vectors_3d[1], independent_vectors_3d[2]], 
#                 colors_independent_3d, labels_independent_3d, ax)
# plot_vectors_3d([dependent_vectors_3d[0], dependent_vectors_3d[1]], colors_dependent_3d, labels_dependent_3d, ax)
# plt.title("Linearly Independent and Dependent Vectors (3D)")
# ax.legend(["Independent Vectors (Red, Blue, Green)", "Dependent Vectors (Yellow, Cyan)"])
# plt.show()

# # Null Space and Column Space (3D)
# A_3d = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 7]])
# null_space_3d = np.linalg.svd(A_3d)[2].T[:, 2:]
# print(f"Null space of A_3d: \n{null_space_3d}")

# # Kernel and Range of Linear Transformation (3D)
# kernel_3d = null_space_3d  # Kernel is the same as null space in this case
# print(f"Kernel of A_3d: \n{kernel_3d}")

# # Column Space (Range) of Matrix (3D)
# column_space_3d = np.linalg.matrix_rank(A_3d)
# print(f"Column space (rank) of A_3d: {column_space_3d}")

# # Plot Null Space and Column Space (3D)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Check if the null space is empty and if it has 3 components
# if null_space_3d.size > 0:
#     if null_space_3d.shape[1] >= 3:  # Ensure there are at least 3 components in the vector
#         plot_vectors_3d([null_space_3d[0]], ['r'], ['Null Space'], ax)
#     else:
#         print("The null space vector does not have 3 components.")
# else:
#     print("The null space is trivial (only the zero vector) and cannot be plotted.")

# # Plot Column Space in 3D
# plot_vectors_3d([A_3d[:, 0], A_3d[:, 1], A_3d[:, 2]], ['g', 'b', 'y'], ['Column Space 1', 'Column Space 2', 'Column Space 3'], ax)
# plt.title("Null Space and Column Space (3D)")
# ax.legend(["Null Space (Red)", "Column Space 1 (Green)", "Column Space 2 (Blue)", "Column Space 3 (Yellow)"])
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to plot vectors in 2D
def plot_vectors_2d(vectors, colors, labels, ax=None):
    if ax is None:
        ax = plt.gca()
    for vector, color, label in zip(vectors, colors, labels):
        ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=color)
        ax.text(vector[0], vector[1], f'{label}: ({vector[0]}, {vector[1]})', fontsize=12, color=color)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc="upper right")

# Function to plot vectors in 3D
def plot_vectors_3d(vectors, colors, labels, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for vector, color, label in zip(vectors, colors, labels):
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=color, length=1, normalize=True)
        ax.text(vector[0], vector[1], vector[2], f'{label}: ({vector[0]}, {vector[1]}, {vector[2]})', fontsize=12, color=color)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True)
    ax.legend(loc="upper left")

# Basis Vectors and Spanning Example (2D)
basis_vectors_2d = np.array([[1, 0], [0, 1]])  # Standard basis in 2D
labels_2d = ['e1', 'e2']
colors_2d = ['r', 'b']

# Spanning Example (2D)
vector_2d = np.array([3, 4])  # Vector in 2D
labels_2d_spanning = ['v', 'e1', 'e2']
colors_2d_spanning = ['g', 'r', 'b']

# Plot 2D
fig, ax = plt.subplots()
plot_vectors_2d([basis_vectors_2d[0], basis_vectors_2d[1], vector_2d], colors_2d_spanning, labels_2d_spanning, ax)
plt.title("Basis Vectors and Spanning in 2D")
plt.legend(["Basis Vector e1 (Red)", "Basis Vector e2 (Blue)", "Spanning Vector v (Green)"])
plt.show()

# Basis Vectors and Spanning Example (3D)
basis_vectors_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Standard basis in 3D
vector_3d = np.array([2, 3, 5])  # Vector in 3D
labels_3d = ['e1', 'e2', 'e3', 'v']
colors_3d = ['r', 'b', 'g', 'y']

# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_vectors_3d([basis_vectors_3d[0], basis_vectors_3d[1], basis_vectors_3d[2], vector_3d], colors_3d, labels_3d, ax)
plt.title("Basis Vectors and Spanning in 3D")
ax.legend(["Basis Vector e1 (Red)", "Basis Vector e2 (Blue)", "Basis Vector e3 (Green)", "Spanning Vector v (Yellow)"])
plt.show()

# Eigenvalues and Eigenvectors Example
A = np.array([[4, -2], [1, 1]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors: \n{eigenvectors}")

# Plot Eigenvectors in 2D
fig, ax = plt.subplots()
plot_vectors_2d([eigenvectors[:, 0], eigenvectors[:, 1]], ['r', 'b'], ['Eigenvector 1', 'Eigenvector 2'], ax)
plt.title("Eigenvectors of Matrix A")
ax.legend(["Eigenvector 1 (Red)", "Eigenvector 2 (Blue)"])
plt.show()

# Linear Independence Example
vectors = np.array([[1, 2], [2, 4]])  # These are linearly dependent
rank = np.linalg.matrix_rank(vectors)
print(f"Rank of the matrix: {rank}")

# Change of Basis Example
# Transforming vector [1, 1] from basis e1, e2 to a new basis e1' = [1, 1] and e2' = [1, -1]
new_basis = np.array([[1, 1], [1, -1]])  # Change of basis matrix
vector_to_transform = np.array([1, 1])

# In the new basis
transformed_vector = np.linalg.inv(new_basis).dot(vector_to_transform)
print(f"Transformed vector: {transformed_vector}")

# Null Space and Column Space Example
A = np.array([[1, 2], [3, 4], [5, 6]])
null_space = np.linalg.svd(A)[2].T[:, 2:]
print(f"Null space of A: \n{null_space}")

# Kernel and Range of Linear Transformation
# Kernel is the null space, and range is the column space of matrix A
column_space = np.linalg.matrix_rank(A)
print(f"Column space rank of matrix A: {column_space}")

# Plot Null Space and Column Space (2D)
fig, ax = plt.subplots()

# Check if the null space is empty
if null_space.size > 0:
    plot_vectors_2d([null_space[0]], ['r'], ['Null Space'], ax)
else:
    print("The null space is trivial (only the zero vector) and cannot be plotted.")

plot_vectors_2d([A[:, 0], A[:, 1]], ['g', 'b'], ['Column Space 1', 'Column Space 2'], ax)
plt.title("Null Space and Column Space (2D)")
ax.legend(["Null Space (Red)", "Column Space 1 (Green)", "Column Space 2 (Blue)"])
plt.show()

# Linearly Independent and Dependent Vectors (3D)
independent_vectors_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Independent vectors in 3D
dependent_vectors_3d = np.array([[1, 2, 3], [2, 4, 6]])  # Dependent vectors in 3D (multiple of each other)
labels_independent_3d = ['e1', 'e2', 'e3']
labels_dependent_3d = ['v1', 'v2']
colors_independent_3d = ['r', 'b', 'g']
colors_dependent_3d = ['y', 'c']

# Plotting Linearly Independent and Dependent Vectors (3D)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_vectors_3d([independent_vectors_3d[0], independent_vectors_3d[1], independent_vectors_3d[2]], 
                colors_independent_3d, labels_independent_3d, ax)
plot_vectors_3d([dependent_vectors_3d[0], dependent_vectors_3d[1]], colors_dependent_3d, labels_dependent_3d, ax)
plt.title("Linearly Independent and Dependent Vectors (3D)")
ax.legend(["Independent Vectors (Red, Blue, Green)", "Dependent Vectors (Yellow, Cyan)"])
plt.show()

# Null Space and Column Space (3D)
A_3d = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 7]])
null_space_3d = np.linalg.svd(A_3d)[2].T[:, 2:]
print(f"Null space of A_3d: \n{null_space_3d}")

# Kernel and Range of Linear Transformation (3D)
kernel_3d = null_space_3d  # Kernel is the same as null space in this case
print(f"Kernel of A_3d: \n{kernel_3d}")

# Column Space (Range) of Matrix (3D)
column_space_3d = np.linalg.matrix_rank(A_3d)
print(f"Column space (rank) of A_3d: {column_space_3d}")

# Plot Null Space and Column Space (3D)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Check if the null space is empty and if it has 3 components
if null_space_3d.size > 0:
    if null_space_3d.shape[1] >= 3:  # Ensure there are at least 3 components in the vector
        plot_vectors_3d([null_space_3d[0]], ['r'], ['Null Space'], ax)
    else:
        print("The null space vector does not have 3 components.")
else:
    print("The null space is trivial (only the zero vector) and cannot be plotted.")

# Plot Column Space in 3D
plot_vectors_3d([A_3d[:, 0], A_3d[:, 1], A_3d[:, 2]], ['g', 'b', 'y'], ['Column Space 1', 'Column Space 2', 'Column Space 3'], ax)
plt.title("Null Space and Column Space (3D)")
ax.legend(["Null Space (Red)", "Column Space 1 (Green)", "Column Space 2 (Blue)", "Column Space 3 (Yellow)"])
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define basis vectors and a sample vector (v_3d) for the 3D space
e1_3d = np.array([1, 0, 0])  # X-axis basis vector
e2_3d = np.array([0, 1, 0])  # Y-axis basis vector
e3_3d = np.array([0, 0, 1])  # Z-axis basis vector
v_3d = np.array([2, 3, 4])  # Example vector in 3D space

# Set up the 3D figure for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def init():
    """
    This function initializes the 3D plot with the basis vectors (e1, e2, e3)
    and the example vector (v_3d) at the origin.
    """
    # Draw the basis vectors (e1, e2, e3) in red, green, and blue respectively
    ax.quiver(0, 0, 0, e1_3d[0], e1_3d[1], e1_3d[2], color='r', label='e1 (X-axis)')
    ax.quiver(0, 0, 0, e2_3d[0], e2_3d[1], e2_3d[2], color='g', label='e2 (Y-axis)')
    ax.quiver(0, 0, 0, e3_3d[0], e3_3d[1], e3_3d[2], color='b', label='e3 (Z-axis)')
    
    # Draw the initial example vector v_3d in black
    ax.quiver(0, 0, 0, v_3d[0], v_3d[1], v_3d[2], color='k', label='v_3d (Example Vector)')

    # Set axis limits
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)

    # Labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a title
    ax.set_title("3D Vector Visualization with Animation")
    
    # Add a legend
    ax.legend()

    return []

def update(frame):
    """
    This function updates the position of the vector v_3d over time during animation.
    The vector v_3d moves along the X and Y components with a sine and cosine movement
    while keeping the Z-component constant.
    """
    ax.cla()  # Clear the axis
    
    # Redraw the static basis vectors (e1, e2, e3) and initial vector v_3d
    ax.quiver(0, 0, 0, e1_3d[0], e1_3d[1], e1_3d[2], color='r', label='e1 (X-axis)')
    ax.quiver(0, 0, 0, e2_3d[0], e2_3d[1], e2_3d[2], color='g', label='e2 (Y-axis)')
    ax.quiver(0, 0, 0, e3_3d[0], e3_3d[1], e3_3d[2], color='b', label='e3 (Z-axis)')

    # Update the position of the vector v_3d
    ax.quiver(0, 0, 0, v_3d[0] * np.sin(frame / 10), v_3d[1] * np.cos(frame / 10), v_3d[2], color='k', label='v_3d (Moving Vector)')

    # Set axis limits again
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)

    # Labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a title
    ax.set_title("3D Vector Visualization with Animation")

    # Add a legend
    ax.legend()

    return []

# Create the animation
ani = FuncAnimation(fig, update, frames=50, init_func=init, blit=True)

# Display the plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define a transformation matrix (example: scaling and rotation)
A = np.array([[2, 1], 
              [1, 2]])

# Eigenvalues and eigenvectors for the matrix A
eigenvalues, eigenvectors = np.linalg.eig(A)

# Vectors to animate (other than eigenvectors)
v = np.array([1, 1])  # Example non-eigenvector

# Set up 2D plot
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Initialize the vectors to be plotted
line1, = ax.plot([], [], 'r-', label='Eigenvector 1 (scaled)', lw=2)
line2, = ax.plot([], [], 'g-', label='Eigenvector 2 (scaled)', lw=2)
line3, = ax.plot([], [], 'b-', label='Other Vector', lw=2)

# Function to initialize the plot
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3

# Function to update the plot for each frame
def update(frame):
    # Eigenvector 1 transformation (scaled by eigenvalue)
    eigenvector_1 = eigenvectors[:, 0] * eigenvalues[0] * np.sin(frame / 10)
    line1.set_data([0, eigenvector_1[0]], [0, eigenvector_1[1]])

    # Eigenvector 2 transformation (scaled by eigenvalue)
    eigenvector_2 = eigenvectors[:, 1] * eigenvalues[1] * np.cos(frame / 10)
    line2.set_data([0, eigenvector_2[0]], [0, eigenvector_2[1]])

    # Other vector transformation (not an eigenvector)
    v_transformed = np.dot(A, v) * np.sin(frame / 10)
    line3.set_data([0, v_transformed[0]], [0, v_transformed[1]])

    return line1, line2, line3

# Create the animation
ani = FuncAnimation(fig, update, frames=50, init_func=init, blit=True)

# Display the plot with the legend
ax.legend(loc='upper left')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Linearly Independent Vectors (Example)
v1_independent = np.array([2, 1])
v2_independent = np.array([1, 3])

# Linearly Dependent Vectors (Example)
v1_dependent = np.array([2, 1])
v2_dependent = np.array([4, 2])  # Scalar multiple of v1

# Set up 2D plot
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Initialize the vectors to be plotted
line1, = ax.plot([], [], 'r-', label='Vector 1', lw=2)
line2, = ax.plot([], [], 'g-', label='Vector 2', lw=2)
line3, = ax.plot([], [], 'b-', label='Linearly Dependent Vector 1', lw=2)
line4, = ax.plot([], [], 'y-', label='Linearly Dependent Vector 2', lw=2)

# Function to initialize the plot
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1, line2, line3, line4

# Function to update the plot for each frame
def update(frame):
    # Linearly Independent Vectors Animation
    v1_ind = v1_independent * np.sin(frame / 10)
    v2_ind = v2_independent * np.cos(frame / 10)
    line1.set_data([0, v1_ind[0]], [0, v1_ind[1]])
    line2.set_data([0, v2_ind[0]], [0, v2_ind[1]])

    # Linearly Dependent Vectors Animation
    v1_dep = v1_dependent * np.sin(frame / 10)
    v2_dep = v2_dependent * np.sin(frame / 10)  # Same direction as v1_dep
    line3.set_data([0, v1_dep[0]], [0, v1_dep[1]])
    line4.set_data([0, v2_dep[0]], [0, v2_dep[1]])

    return line1, line2, line3, line4

# Create the animation
ani = FuncAnimation(fig, update, frames=50, init_func=init, blit=True)

# Display the plot with the legend
ax.legend(loc='upper left')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define vectors for animation
v1 = np.array([2, 1])  # Vector 1
v2 = np.array([1, 3])  # Vector 2
v_subspace = np.array([1, 2])  # Subspace vector (part of the spanning space)

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Initialize vectors and plot
line1, = ax.plot([], [], 'r-', label='Vector 1', lw=2)
line2, = ax.plot([], [], 'g-', label='Vector 2', lw=2)
line3, = ax.plot([], [], 'b-', label='Subspace Vector', lw=2)
text = ax.text(0, 4, '', fontsize=12, ha='left')

# Initialize plot function
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    text.set_text('')
    return line1, line2, line3, text

# Function to update the plot for each frame
def update(frame):
    # Spanning Animation (V1 and V2 combined)
    v1_scaled = v1 * np.sin(frame / 10)
    v2_scaled = v2 * np.cos(frame / 10)
    line1.set_data([0, v1_scaled[0]], [0, v1_scaled[1]])
    line2.set_data([0, v2_scaled[0]], [0, v2_scaled[1]])

    # Vector Subspace (A vector in the subspace)
    v_sub_scaled = v_subspace * np.cos(frame / 10)
    line3.set_data([0, v_sub_scaled[0]], [0, v_sub_scaled[1]])

    # Text showing what's happening
    if frame < 20:
        text.set_text('Spanning of Vectors: Two vectors can span a plane.')
    elif frame < 40:
        text.set_text('Subspace: A subspace is a part of a larger space.')
    else:
        text.set_text('Basis of Vectors: Vectors must be linearly independent to form a basis.')

    return line1, line2, line3, text

# Create the animation
ani = FuncAnimation(fig, update, frames=60, init_func=init, blit=True)

# Display the plot with the legend
ax.legend(loc='upper left')
plt.show()
