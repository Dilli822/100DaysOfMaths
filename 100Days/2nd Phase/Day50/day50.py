import numpy as np
import matplotlib.pyplot as plt

# Create 2D vectors
v1 = np.array([2, 3])
v2 = np.array([4, 1])

# Plotting the vectors
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1 (2, 3)')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2 (4, 1)')
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# Adding labels and grid
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title("2D Vector Space")
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Create 3D vectors
v1 = np.array([2, 3, 1])
v2 = np.array([1, -1, 2])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the vectors
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='v1 (2, 3, 1)')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='v2 (1, -1, 2)')

# Set limits and labels
ax.set_xlim([0, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([0, 3])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.legend()
plt.title("3D Vector Space")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create the base vectors
v1 = np.array([2, 3])
v2 = np.array([4, 1])

fig, ax = plt.subplots(figsize=(6, 6))

# Set limits
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Function to update the plot
def update(frame):
    ax.clear()
    ax.quiver(0, 0, v1[0]*frame/10, v1[1]*frame/10, angles='xy', scale_units='xy', scale=1, color='r', label='v1')
    ax.quiver(0, 0, v2[0]*frame/10, v2[1]*frame/10, angles='xy', scale_units='xy', scale=1, color='b', label='v2')
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_title(f"Animation Frame: {frame}")

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(1, 11), repeat=True, interval=500)

plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Vector space example: V = R^2
v1 = np.array([2, 3])
v2 = np.array([4, 1])

plt.figure(figsize=(6, 6))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1 (2, 3)')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2 (4, 1)')
plt.quiver(0, 0, v1[0] + v2[0], v1[1] + v2[1], angles='xy', scale_units='xy', scale=1, color='g', label='v1 + v2')

# Labels and grid
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title("Vector Space Example: R^2")
plt.show()


# Not a vector space example: V = {(x, y) in R^2 | x + y = 0}
v1 = np.array([1, -1])  # (1, -1)
v2 = np.array([2, -2])  # (2, -2)

# Plot the vectors and their scaled versions
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1 (1, -1)')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2 (2, -2)')
plt.quiver(0, 0, 3*v1[0], 3*v1[1], angles='xy', scale_units='xy', scale=1, color='g', label='3*v1')

# Labels and grid
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title("Not a Vector Space: x + y = 0")
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Vectors for 3D
v1 = np.array([2, 3, 1])
v2 = np.array([4, 1, 3])
v3 = np.array([1, 5, 2])

# Determinant calculation
det = np.linalg.det(np.array([v1, v2, v3]))

# Plotting the 3D vectors
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the vectors
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='v1 (2, 3, 1)')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='v2 (4, 1, 3)')
ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='g', label='v3 (1, 5, 2)')

# Setting the limits for the plot
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_zlim([0, 5])

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f"3D Vectors and Parallelepiped Volume (Det = {det:.2f})")

# Show legend and grid
ax.legend()
ax.grid(True)

# Show plot
plt.show()

import matplotlib.animation as animation

# Function to update the plot during animation
def update_vectors(frame):
    ax.clear()  # Clear the previous plot
    scale = 1 + 0.1 * frame  # Change in scale factor

    # Scale the vectors
    v1_scaled = v1 * scale
    v2_scaled = v2 * scale
    v3_scaled = v3 * scale

    # Recalculate determinant
    det = np.linalg.det(np.array([v1_scaled, v2_scaled, v3_scaled]))

    # Plot the scaled vectors
    ax.quiver(0, 0, 0, v1_scaled[0], v1_scaled[1], v1_scaled[2], color='r', label=f'v1 scaled ({v1_scaled[0]:.2f}, {v1_scaled[1]:.2f}, {v1_scaled[2]:.2f})')
    ax.quiver(0, 0, 0, v2_scaled[0], v2_scaled[1], v2_scaled[2], color='b', label=f'v2 scaled ({v2_scaled[0]:.2f}, {v2_scaled[1]:.2f}, {v2_scaled[2]:.2f})')
    ax.quiver(0, 0, 0, v3_scaled[0], v3_scaled[1], v3_scaled[2], color='g', label=f'v3 scaled ({v3_scaled[0]:.2f}, {v3_scaled[1]:.2f}, {v3_scaled[2]:.2f})')

    # Setting the limits for the plot
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"3D Vectors and Parallelepiped Volume (Det = {det:.2f})")
    ax.legend()
    ax.grid(True)

# Plotting and animation
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Animation
ani = animation.FuncAnimation(fig, update_vectors, frames=30, interval=200)

plt.show()




import numpy as np
import matplotlib.pyplot as plt

# Vector in R^2
v = np.array([3, 2])

# Scalar values
scalars = np.linspace(-5, 5, 50)

# Generate the line by multiplying the vector by scalars
line = np.array([scalars * v[0], scalars * v[1]])

# Plotting
plt.figure(figsize=(6, 6))
plt.plot(line[0], line[1], label="Subspace: Scalar multiples of v = (3, 2)")

# Plot vector v
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label="v = (3, 2)")

# Labels and title
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Subspace in \( R^2 \) : Line through the origin')
plt.legend()

# Show grid
plt.grid(True)
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Vectors for R^3 (span of two vectors forming a plane)
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# Define the grid for the plane
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Plotting the subspace plane
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the plane
ax.plot_surface(X, Y, Z, color='gray', alpha=0.5, rstride=100, cstride=100)

# Plot the vectors v1 and v2
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='v1 = (1, 0, 0)', linewidth=2)
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='v2 = (0, 1, 0)', linewidth=2)

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Subspace in \( R^3 \) : Plane through the origin')

# Show grid and legend
ax.grid(True)
ax.legend()

# Show plot
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the vector in R^2
v = np.array([3, 2])

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.axhline(0, color='black',linewidth=0.5)
ax.axvline(0, color='black',linewidth=0.5)
ax.grid(True)

# Plot the line that will be the subspace
line, = ax.plot([], [], label="Subspace: Scalar multiples of v", color="blue")
vector = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='red', label="v = (3, 2)")

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(r'Subspace in $\mathbb{R}^2$ : Line through the origin')  # Use raw string for LaTeX

# Create the update function
def update(frame):
    scalar = frame
    # Calculate the new coordinates of the vector
    x = scalar * v[0]
    y = scalar * v[1]
    
    # Update the vector and the line
    vector.set_UVC(x, y)
    line.set_data(np.linspace(0, x, 100), np.linspace(0, y, 100))
    
    return vector, line

# Create the animation
ani = FuncAnimation(fig, update, frames=np.linspace(-5, 5, 100), blit=True, interval=50)

# Show the legend
ax.legend()

plt.show()







from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Vectors for R^3 (span of two vectors forming a plane)
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# Define the grid for the plane
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Create a figure for 3D plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the axes limits
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])

# Plot the vectors v1 and v2
v1_quiver = ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='v1 = (1, 0, 0)', linewidth=2)
v2_quiver = ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='v2 = (0, 1, 0)', linewidth=2)

# Plane surface (the subspace)
plane = ax.plot_surface(X, Y, Z, color='gray', alpha=0.3)

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Animating Subspace in \( R^3 \) : Plane through the origin')

# Create the update function
def update(frame):
    # Scalar multiples of the vectors
    scalar_v1 = frame
    scalar_v2 = frame
    
    # New positions for the vectors
    ax.quiver(0, 0, 0, scalar_v1 * v1[0], scalar_v1 * v1[1], scalar_v1 * v1[2], color='r')
    ax.quiver(0, 0, 0, scalar_v2 * v2[0], scalar_v2 * v2[1], scalar_v2 * v2[2], color='b')
    
    # Update the plane
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.3)
    
    return v1_quiver, v2_quiver

# Create the animation
ani = FuncAnimation(fig, update, frames=np.linspace(0, 5, 50), blit=False, interval=200)

# Show the plot
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import lu

# 2D LU Factorization and Animation
def plot_2d_lu_animation(A):
    # Perform LU decomposition
    P, L, U = lu(A)
    
    # Create a plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.grid(True)
    
    # Initialize the point to transform
    point = np.array([1, 1])
    point_line, = ax.plot([], [], 'ro', label='Initial point')
    transformed_line, = ax.plot([], [], 'bo', label='Transformed point')
    
    # Update function for animation
    def update(frame):
        # Transform the point using L and U
        transformed_point = np.dot(U, np.dot(L, point))
        
        # Update the points on the plot
        point_line.set_data([point[0]], [point[1]])
        transformed_line.set_data([transformed_point[0]], [transformed_point[1]])
        
        return point_line, transformed_line

    # Create animation
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 1, 100), blit=False, interval=50)
    ax.set_title(r'Subspace in $R^2$ : Line through the origin')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    plt.show()

# 3D LU Factorization and Animation
def plot_3d_lu_animation(A):
    # Perform LU decomposition
    P, L, U = lu(A)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    
    # Initialize the point to transform
    point = np.array([1, 1, 1])
    point_line, = ax.plot([], [], [], 'ro', label='Initial point')
    transformed_line, = ax.plot([], [], [], 'bo', label='Transformed point')
    
    # Update function for animation
    def update(frame):
        # Transform the point using L and U
        transformed_point = np.dot(U, np.dot(L, point))
        
        # Update the points on the plot
        point_line.set_data([point[0]], [point[1]])
        point_line.set_3d_properties([point[2]])
        
        transformed_line.set_data([transformed_point[0]], [transformed_point[1]])
        transformed_line.set_3d_properties([transformed_point[2]])
        
        return point_line, transformed_line

    # Create animation
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 1, 100), blit=False, interval=50)
    ax.set_title(r'Subspace in $R^3$ : Plane through the origin')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

# 2D Matrix example for LU factorization
A_2d = np.array([[2, 3],
                 [4, 6]])

plot_2d_lu_animation(A_2d)

# 3D Matrix example for LU factorization
A_3d = np.array([[4, -2, 1],
                 [3,  6, -1],
                 [2,  1,  5]])

plot_3d_lu_animation(A_3d)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 2D Plot - Matrix Dimension and Rank Visualization
def plot_2d_rank_dimension():
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_xlabel('Column 1')
    ax.set_ylabel('Column 2')
    ax.set_title('2D Matrix: Rank and Dimension')

    # Matrix with rank 2 (two independent vectors)
    A = np.array([[1, 1], [2, 3]])

    # Vectors for plotting
    ax.quiver(0, 0, A[0, 0], A[1, 0], angles='xy', scale_units='xy', scale=1, color='r', label='Column 1')
    ax.quiver(0, 0, A[0, 1], A[1, 1], angles='xy', scale_units='xy', scale=1, color='b', label='Column 2')

    ax.legend()
    plt.grid(True)
    plt.show()

# 3D Plot - Matrix Dimension and Rank Visualization
def plot_3d_rank_dimension():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Matrix: Rank and Dimension')

    # Matrix with rank 3 (three independent vectors)
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Vectors for plotting (rows of the matrix)
    ax.quiver(0, 0, 0, A[0, 0], A[0, 1], A[0, 2], color='r', label='Row 1')
    ax.quiver(0, 0, 0, A[1, 0], A[1, 1], A[1, 2], color='g', label='Row 2')
    ax.quiver(0, 0, 0, A[2, 0], A[2, 1], A[2, 2], color='b', label='Row 3')

    ax.legend()
    plt.show()

# Animation - Demonstrating Rank Change
def plot_rank_animation():
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_xlabel('Column 1')
    ax.set_ylabel('Column 2')
    ax.set_title('Animating Rank Change in 2D')

    # Matrix changing over time (linear dependence added)
    def update(frame):
        ax.clear()
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.set_xlabel('Column 1')
        ax.set_ylabel('Column 2')
        ax.set_title('Animating Rank Change in 2D')

        A = np.array([[1, 1], [2, frame]])
        
        # Plot vectors based on matrix A
        ax.quiver(0, 0, A[0, 0], A[1, 0], angles='xy', scale_units='xy', scale=1, color='r', label='Column 1')
        ax.quiver(0, 0, A[0, 1], A[1, 1], angles='xy', scale_units='xy', scale=1, color='b', label='Column 2')

        # Rank change based on frame
        ax.legend()

    ani = FuncAnimation(fig, update, frames=np.linspace(1, 3, 10), blit=False, interval=500)
    plt.show()

# Calling the functions
plot_2d_rank_dimension()  # Plot 2D
plot_3d_rank_dimension()  # Plot 3D
plot_rank_animation()     # Animation of Rank Change


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 2D Plot (Line Intersection)
def plot_2d():
    x = np.linspace(-10, 10, 400)
    
    # Equation 1: 2x + 3y = 5 -> y = (5 - 2x) / 3
    y1 = (5 - 2 * x) / 3
    
    # Equation 2: 4x + y = 6 -> y = 6 - 4x
    y2 = 6 - 4 * x
    
    plt.figure(figsize=(6, 6))
    plt.plot(x, y1, label=r"$2x + 3y = 5$")
    plt.plot(x, y2, label=r"$4x + y = 6$")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("2D Plot of System of Equations.plot 2D, 3D, and animated visualizations for Cramer's Rule, we can visualize how the system of equations and its solutions evolve in a 2D and 3D space")
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.show()

# 3D Plot (Planes for a 3-variable system)
def plot_3d():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    
    # Equation 1: x + y + z = 6 => z = 6 - x - y
    Z1 = 6 - X - Y
    
    # Equation 2: 2x + y + 3z = 14 => z = (14 - 2x - y) / 3
    Z2 = (14 - 2 * X - Y) / 3
    
    # Equation 3: 3x + 2y + 2z = 13 => z = (13 - 3x - 2y) / 2
    Z3 = (13 - 3 * X - 2 * Y) / 2
    
    ax.plot_surface(X, Y, Z1, alpha=0.5, rstride=100, cstride=100, color='r')
    ax.plot_surface(X, Y, Z2, alpha=0.5, rstride=100, cstride=100, color='g')
    ax.plot_surface(X, Y, Z3, alpha=0.5, rstride=100, cstride=100, color='b')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of System of Equations')
    plt.show()

# Animation Plot (Visualizing changing solution or determinant)
def animate_solution():
    fig, ax = plt.subplots(figsize=(8, 8))
    x = np.linspace(-10, 10, 400)
    
    # Plot initial lines
    line1, = ax.plot([], [], label=r"$2x + 3y = 5$")
    line2, = ax.plot([], [], label=r"$4x + y = 6$")
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Animated Solution of System of Equations")
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.grid(True)
    ax.legend()
    
    def update(frame):
        y1 = (5 - 2 * x) / 3 + frame * 0.1
        y2 = 6 - 4 * x + frame * 0.1
        line1.set_data(x, y1)
        line2.set_data(x, y2)
        return line1, line2

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), blit=True, interval=50)
    plt.show()

# Call the plotting functions
plot_2d()           # To visualize the 2D case
plot_3d()           # To visualize the 3D case
animate_solution()  # To animate the changing solutions
