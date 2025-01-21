import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- Helper Functions ---
# Compute dot product (Inner Product)
def dot_product(v1, v2):
    return np.dot(v1, v2)

# Length of a vector (Norm)
def length(v):
    return np.linalg.norm(v)

# Normalize a vector
def normalize(v):
    return v / length(v)

# Compute the distance between two vectors
def distance(v1, v2):
    return np.linalg.norm(v1 - v2)

# Gram-Schmidt Process
def gram_schmidt(vectors):
    orthogonal_vectors = []
    for v in vectors:
        for u in orthogonal_vectors:
            v = v - (dot_product(v, u) / dot_product(u, u)) * u
        orthogonal_vectors.append(v)
    return orthogonal_vectors

# --- 2D Visualization ---
def plot_2d():
    plt.figure(figsize=(8, 8))

    # Vectors A and B
    A = np.array([4, 2])
    B = np.array([1, 4])

    # Plotting vectors
    plt.quiver(0, 0, A[0], A[1], angles='xy', scale_units='xy', scale=1, color='r', label="Vector A")
    plt.quiver(0, 0, B[0], B[1], angles='xy', scale_units='xy', scale=1, color='b', label="Vector B")
    
    # Orthogonal projection of B onto A
    proj_B_on_A = (dot_product(A, B) / dot_product(A, A)) * A
    plt.quiver(0, 0, proj_B_on_A[0], proj_B_on_A[1], angles='xy', scale_units='xy', scale=1, color='g', label="Projection of B onto A")

    # Labels and Grid
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("2D Vector Visualization: Inner Product and Orthogonal Projection")

    # Display Plot
    plt.show()

# --- 3D Visualization ---
def plot_3d():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Vectors A and B
    A = np.array([4, 2, 3])
    B = np.array([1, 4, 2])

    # Plotting vectors
    ax.quiver(0, 0, 0, A[0], A[1], A[2], color='r', label="Vector A")
    ax.quiver(0, 0, 0, B[0], B[1], B[2], color='b', label="Vector B")

    # Orthogonal projection of B onto A
    proj_B_on_A = (dot_product(A, B) / dot_product(A, A)) * A
    ax.quiver(0, 0, 0, proj_B_on_A[0], proj_B_on_A[1], proj_B_on_A[2], color='g', label="Projection of B onto A")

    # Labels and Grid
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("3D Vector Visualization: Inner Product and Orthogonal Projection")

    # Display Plot
    plt.show()

# --- Animation Visualization for Least Squares and Linear Models ---
def animate_least_squares():
    fig, ax = plt.subplots(figsize=(8, 8))

    # Generate some data points
    np.random.seed(0)
    x_data = np.random.rand(10)
    y_data = 2 * x_data + 1 + 0.5 * np.random.randn(10)  # Line with some noise

    # Fit a linear model using Least Squares
    A = np.vstack([x_data, np.ones(len(x_data))]).T
    m, c = np.linalg.lstsq(A, y_data, rcond=None)[0]  # Linear model coefficients

    # Plotting Data Points
    ax.scatter(x_data, y_data, color='b', label="Data Points")

    # Line from Least Squares
    line, = ax.plot([], [], 'r-', label="Least Squares Line")

    def update(frame):
        line.set_data([0, 1], [m * 0 + c, m * 1 + c])
        return line,

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_title("Least Squares Linear Fit")

    # Create Animation
    ani = FuncAnimation(fig, update, frames=10, blit=True)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Plot 2D Visualization
    plot_2d()

    # Plot 3D Visualization
    plot_3d()

    # Animate Least Squares Linear Model
    animate_least_squares()







import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Helper Functions ---
# Compute dot product (Inner Product)
def dot_product(v1, v2):
    return np.dot(v1, v2)

# Gram-Schmidt Process
def gram_schmidt(vectors):
    orthogonal_vectors = []
    for v in vectors:
        for u in orthogonal_vectors:
            v = v - (dot_product(v, u) / dot_product(u, u)) * u
        orthogonal_vectors.append(v)
    return orthogonal_vectors

# --- 2D Gram-Schmidt Visualization ---
def plot_2d_gram_schmidt():
    plt.figure(figsize=(8, 8))

    # Define two vectors A and B
    A = np.array([4, 2])
    B = np.array([1, 4])

    # Apply Gram-Schmidt to make B orthogonal to A
    orthogonal_vectors = gram_schmidt([A, B])

    # Plot the original vectors A and B
    plt.quiver(0, 0, A[0], A[1], angles='xy', scale_units='xy', scale=1, color='r', label="Original Vector A")
    plt.quiver(0, 0, B[0], B[1], angles='xy', scale_units='xy', scale=1, color='b', label="Original Vector B")

    # Plot the orthogonalized vectors
    plt.quiver(0, 0, orthogonal_vectors[0][0], orthogonal_vectors[0][1], angles='xy', scale_units='xy', scale=1, color='g', label="Orthogonalized Vector A'")
    plt.quiver(0, 0, orthogonal_vectors[1][0], orthogonal_vectors[1][1], angles='xy', scale_units='xy', scale=1, color='purple', label="Orthogonalized Vector B'")

    # Labels and Grid
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("2D Gram-Schmidt Process")

    # Display Plot
    plt.show()

# --- 3D Gram-Schmidt Visualization ---
def plot_3d_gram_schmidt():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define two vectors A and B
    A = np.array([4, 2, 3])
    B = np.array([1, 4, 2])

    # Apply Gram-Schmidt to make B orthogonal to A
    orthogonal_vectors = gram_schmidt([A, B])

    # Plot the original vectors A and B
    ax.quiver(0, 0, 0, A[0], A[1], A[2], color='r', label="Original Vector A")
    ax.quiver(0, 0, 0, B[0], B[1], B[2], color='b', label="Original Vector B")

    # Plot the orthogonalized vectors
    ax.quiver(0, 0, 0, orthogonal_vectors[0][0], orthogonal_vectors[0][1], orthogonal_vectors[0][2], color='g', label="Orthogonalized Vector A'")
    ax.quiver(0, 0, 0, orthogonal_vectors[1][0], orthogonal_vectors[1][1], orthogonal_vectors[1][2], color='purple', label="Orthogonalized Vector B'")

    # Labels and Grid
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("3D Gram-Schmidt Process")

    # Display Plot
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Plot 2D Gram-Schmidt Process
    plot_2d_gram_schmidt()
    
    # Plot 3D Gram-Schmidt Process
    plot_3d_gram_schmidt()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Helper Functions ---
# Function to calculate the length of a vector (Euclidean norm)
def vector_length(v):
    return np.linalg.norm(v)

# --- 2D Length Visualization ---
def plot_2d_vector_length():
    plt.figure(figsize=(8, 8))

    # Define a 2D vector
    v = np.array([3, 4])

    # Calculate its length
    length = vector_length(v)

    # Plot the vector
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label=f"Vector v = {v}")
    plt.text(v[0] + 0.1, v[1], f'|v| = {length:.2f}', fontsize=12, color='black')

    # Plot the length as a dashed line from the origin to the vector's tip
    plt.plot([0, v[0]], [0, v[1]], 'k--', label="Length of Vector v")

    # Labels and Grid
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("2D Vector and Its Length")

    # Display Plot
    plt.show()

# --- 3D Length Visualization ---
def plot_3d_vector_length():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define a 3D vector
    v = np.array([3, 4, 5])

    # Calculate its length
    length = vector_length(v)

    # Plot the vector
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color='r', label=f"Vector v = {v}")
    
    # Plot the length as a dashed line from the origin to the vector's tip
    ax.plot([0, v[0]], [0, v[1]], [0, v[2]], 'k--', label="Length of Vector v")

    # Labels and Grid
    ax.set_xlim([0, 6])
    ax.set_ylim([0, 6])
    ax.set_zlim([0, 6])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("3D Vector and Its Length")

    # Display Plot
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Plot 2D Vector Length
    plot_2d_vector_length()

    # Plot 3D Vector Length
    plot_3d_vector_length()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Helper Functions ---
# Function to calculate the inner product (dot product) of two vectors
def inner_product(v1, v2):
    return np.dot(v1, v2)

# --- 2D Orthogonal Set Visualization ---
def plot_2d_orthogonal_set():
    plt.figure(figsize=(8, 8))

    # Define two orthogonal vectors in 2D
    v1 = np.array([3, 0])  # Vector along the x-axis
    v2 = np.array([0, 4])  # Vector along the y-axis

    # Check if they are orthogonal
    dot_product = inner_product(v1, v2)

    # Plot the orthogonal vectors
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label=f"Vector v1 = {v1}")
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label=f"Vector v2 = {v2}")

    # Plot the inner product text
    plt.text(v1[0] + 0.2, v1[1], f"v1 · v2 = {dot_product}", fontsize=12, color='black')

    # Labels and Grid
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Orthogonal Set in 2D")

    # Display Plot
    plt.show()

# --- 3D Orthogonal Set Visualization ---
def plot_3d_orthogonal_set():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define three orthogonal vectors in 3D
    v1 = np.array([3, 0, 0])  # Vector along the x-axis
    v2 = np.array([0, 4, 0])  # Vector along the y-axis
    v3 = np.array([0, 0, 5])  # Vector along the z-axis

    # Check if the vectors are orthogonal pairwise
    dot_product_12 = inner_product(v1, v2)
    dot_product_13 = inner_product(v1, v3)
    dot_product_23 = inner_product(v2, v3)

    # Plot the orthogonal vectors
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label=f"Vector v1 = {v1}")
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label=f"Vector v2 = {v2}")
    ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='g', label=f"Vector v3 = {v3}")

    # Plot the inner product text
    ax.text(v1[0] + 0.5, v1[1], v1[2], f"v1 · v2 = {dot_product_12}", fontsize=12, color='black')
    ax.text(v1[0] + 0.5, v1[1], v1[2], f"v1 · v3 = {dot_product_13}", fontsize=12, color='black')
    ax.text(v2[0] + 0.5, v2[1], v2[2], f"v2 · v3 = {dot_product_23}", fontsize=12, color='black')

    # Labels and Grid
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("Orthogonal Set in 3D")

    # Display Plot
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Plot 2D Orthogonal Set
    plot_2d_orthogonal_set()

    # Plot 3D Orthogonal Set
    plot_3d_orthogonal_set()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Helper Functions ---
# Function to calculate the inner product (dot product) of two vectors
def inner_product(v1, v2):
    return np.dot(v1, v2)

# --- 2D Inner Product Space Visualization ---
def plot_2d_inner_product_space():
    plt.figure(figsize=(8, 8))

    # Define two vectors in 2D
    v1 = np.array([2, 3])  # Vector v1
    v2 = np.array([4, -1]) # Vector v2

    # Calculate the inner product
    dot_product = inner_product(v1, v2)
    
    # Plot the vectors
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label=f"Vector v1 = {v1}")
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label=f"Vector v2 = {v2}")

    # Display the inner product on the plot
    plt.text(v1[0] + 0.5, v1[1], f"v1 · v2 = {dot_product}", fontsize=12, color='black')

    # Labels and Grid
    plt.xlim(-1, 5)
    plt.ylim(-5, 5)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Inner Product Space in 2D")

    # Display Plot
    plt.show()

# --- 3D Inner Product Space Visualization ---
def plot_3d_inner_product_space():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define three vectors in 3D
    v1 = np.array([3, 0, 0])  # Vector v1
    v2 = np.array([0, 4, 0])  # Vector v2
    v3 = np.array([0, 0, 5])  # Vector v3

    # Calculate inner products between the vectors
    dot_product_12 = inner_product(v1, v2)
    dot_product_13 = inner_product(v1, v3)
    dot_product_23 = inner_product(v2, v3)

    # Plot the vectors
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label=f"Vector v1 = {v1}")
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label=f"Vector v2 = {v2}")
    ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='g', label=f"Vector v3 = {v3}")

    # Display the inner products on the plot
    ax.text(v1[0] + 0.5, v1[1], v1[2], f"v1 · v2 = {dot_product_12}", fontsize=12, color='black')
    ax.text(v1[0] + 0.5, v1[1], v1[2], f"v1 · v3 = {dot_product_13}", fontsize=12, color='black')
    ax.text(v2[0] + 0.5, v2[1], v2[2], f"v2 · v3 = {dot_product_23}", fontsize=12, color='black')

    # Labels and Grid
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("Inner Product Space in 3D")

    # Display Plot
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Plot 2D Inner Product Space
    plot_2d_inner_product_space()

    # Plot 3D Inner Product Space
    plot_3d_inner_product_space()



import numpy as np
import matplotlib.pyplot as plt

# --- Helper Functions ---
# Function to calculate the norm (length) of a vector
def norm(v):
    return np.linalg.norm(v)

# Function to normalize a vector (make it a unit vector)
def normalize(v):
    return v / norm(v)

# --- Plotting Unit Vector, Basis Vectors, and Inner Product ---
def plot_unit_vector_and_inner_product():
    plt.figure(figsize=(8, 8))

    # Define a random vector in 2D
    v = np.array([4, 3])

    # Calculate the unit vector of v
    unit_v = normalize(v)

    # Define the standard basis vectors in 2D
    i = np.array([1, 0])  # x-axis unit vector
    j = np.array([0, 1])  # y-axis unit vector

    # Plot the vector v and its unit vector
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label=f"Vector v = {v}")
    plt.quiver(0, 0, unit_v[0], unit_v[1], angles='xy', scale_units='xy', scale=1, color='b', label=f"Unit Vector of v = {unit_v}")

    # Plot the basis vectors i and j
    plt.quiver(0, 0, i[0], i[1], angles='xy', scale_units='xy', scale=1, color='g', label="Basis Vector i = (1, 0)")
    plt.quiver(0, 0, j[0], j[1], angles='xy', scale_units='xy', scale=1, color='y', label="Basis Vector j = (0, 1)")

    # Calculate and display the inner product of v and unit_v
    dot_product = np.dot(v, unit_v)
    plt.text(v[0] + 0.5, v[1], f"v · unit_v = {dot_product:.2f}", fontsize=12, color='black')

    # Labels and Grid
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Unit Vector, Basis Vectors, and Inner Product")

    # Display Plot
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    plot_unit_vector_and_inner_product()
