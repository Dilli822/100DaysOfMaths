import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 2D Example ---
def plot_2d_system():
    x = np.linspace(-10, 10, 100)

    # Non-Singular System: Unique intersection
    y1 = 2 * x + 3
    y2 = -x + 1

    # Singular System: Parallel lines (no solution)
    y3 = 2 * x + 3
    y4 = 2 * x - 2  # Parallel line

    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, label="2x + 3")
    plt.plot(x, y2, label="-x + 1")
    plt.plot(x, y3, '--', label="2x + 3 (Parallel)")
    plt.plot(x, y4, '--', label="2x - 2 (Parallel)")

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.legend()
    plt.title("2D: Singular vs Non-Singular Systems")
    plt.grid()
    plt.show()

# --- 3D Example ---
def plot_3d_system():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))

    # Non-Singular System: Unique intersection
    Z1 = (5 - X - Y) / 2
    Z2 = (8 - 2*X + Y) / 3
    Z3 = (3 - X + 2*Y) / 4

    # Singular System: Parallel planes
    Z4 = (5 - X - Y) / 2
    Z5 = (10 - X - Y) / 2  # Parallel

    ax.plot_surface(X, Y, Z1, alpha=0.5, color='r', edgecolor='k')
    ax.plot_surface(X, Y, Z2, alpha=0.5, color='g', edgecolor='k')
    ax.plot_surface(X, Y, Z3, alpha=0.5, color='b', edgecolor='k')

    ax.plot_surface(X, Y, Z4, alpha=0.3, color='y', edgecolor='k')
    ax.plot_surface(X, Y, Z5, alpha=0.3, color='m', edgecolor='k')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title("3D: Singular vs Non-Singular Systems")
    plt.show()

plot_2d_system()
plot_3d_system()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define grid
X, Y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))

# --- Non-Singular System ---
# Three planes that intersect at one point
Z1 = (5 - X - Y) / 2
Z2 = (8 - 2*X + Y) / 3
Z3 = (3 - X + 2*Y) / 4

# --- Singular System ---
# Two parallel planes (no solution)
Z4 = (5 - X - Y) / 2
Z5 = (10 - X - Y) / 2  # Parallel to Z4

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot non-singular planes
ax.plot_surface(X, Y, Z1, alpha=0.5, color='r', edgecolor='k', label="Plane 1")
ax.plot_surface(X, Y, Z2, alpha=0.5, color='g', edgecolor='k', label="Plane 2")
ax.plot_surface(X, Y, Z3, alpha=0.5, color='b', edgecolor='k', label="Plane 3")

# Plot singular (parallel) planes
ax.plot_surface(X, Y, Z4, alpha=0.3, color='y', edgecolor='k', label="Singular Plane 1")
ax.plot_surface(X, Y, Z5, alpha=0.3, color='m', edgecolor='k', label="Singular Plane 2")

# Labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title("3D: Singular vs Non-Singular Matrices")
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Define two sets of vectors (independent and dependent)
independent_vectors = np.array([[2, 3], [-3, 2]])  # Not multiples
dependent_vectors = np.array([[2, 3], [4, 6]])  # Second is 2x first

# Function to plot vectors
def plot_vectors(vectors, title):
    fig, ax = plt.subplots()
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid()
    
    colors = ['r', 'b']
    for i in range(len(vectors)):
        ax.quiver(0, 0, vectors[i, 0], vectors[i, 1], 
                  angles='xy', scale_units='xy', scale=1, color=colors[i])
    
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title(title)
    plt.show()

# Plot the independent and dependent vectors
plot_vectors(independent_vectors, "Linearly Independent Vectors (2D)")
plot_vectors(dependent_vectors, "Linearly Dependent Vectors (2D)")

from mpl_toolkits.mplot3d import Axes3D

# Define vectors
independent_vectors_3D = np.array([[1, 2, 3], [2, -1, 1], [-1, 3, -2]])  # Independent
dependent_vectors_3D = np.array([[1, 2, 3], [2, 4, 6], [-1, -2, -3]])  # Dependent

# Function to plot 3D vectors
def plot_3d_vectors(vectors, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['r', 'g', 'b']
    for i in range(len(vectors)):
        ax.quiver(0, 0, 0, vectors[i, 0], vectors[i, 1], vectors[i, 2], 
                  color=colors[i], length=1, normalize=True)
    
    ax.set_xlim([-1, 2])
    ax.set_ylim([-2, 4])
    ax.set_zlim([-3, 6])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

# Plot the independent and dependent vectors
plot_3d_vectors(independent_vectors_3D, "Linearly Independent Vectors (3D)")
plot_3d_vectors(dependent_vectors_3D, "Linearly Dependent Vectors (3D)")


import numpy as np
import matplotlib.pyplot as plt

# Define x values
x = np.linspace(-10, 10, 100)

# Define equations for a non-singular system (lines that meet at one point)
y1 = (5 - x) / 2
y2 = (8 - 2*x) / 3
y3 = (3 - x) / 4

# Create 2D plot
plt.figure(figsize=(7, 5))
plt.plot(x, y1, 'r', label="Line 1: (5 - x)/2")
plt.plot(x, y2, 'g', label="Line 2: (8 - 2x)/3")
plt.plot(x, y3, 'b', label="Line 3: (3 - x)/4")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("2D: Non-Singular System (Lines Intersect)")
plt.legend()
plt.grid()
plt.show()


# Define equations for a singular system (parallel lines)
y4 = (5 - x) / 2
y5 = (10 - x) / 2  # Parallel to y4

# Create 2D plot
plt.figure(figsize=(7, 5))
plt.plot(x, y4, 'y', label="Parallel Line 1: (5 - x)/2")
plt.plot(x, y5, 'm', label="Parallel Line 2: (10 - x)/2")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("2D: Singular System (Parallel Lines, No Solution)")
plt.legend()
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define the range of values for a, b, c, d
a_values = np.linspace(-5, 5, 50)
b_values = np.linspace(-5, 5, 50)

# Create a grid of (a, b) values, assuming c=1 and d=1 for simplicity
A, B = np.meshgrid(a_values, b_values)
C, D = 1, 1  # Fixed values
Determinant = (A * D) - (B * C)  # Compute determinant

# Plotting
plt.figure(figsize=(8, 6))
contour = plt.contourf(A, B, Determinant, cmap="coolwarm", levels=20)
plt.colorbar(label="Determinant Value")
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.contour(A, B, Determinant, levels=[0], colors="black", linewidths=2)  # Singular matrices

plt.title("2D Visualization of 2×2 Matrix Determinant")
plt.xlabel("a")
plt.ylabel("b")
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define grid for a, b values (keeping c, d constant for simplicity)
a_values = np.linspace(-5, 5, 50)
b_values = np.linspace(-5, 5, 50)
A, B = np.meshgrid(a_values, b_values)

C, D = 1, 1  # Fixed values
Determinant = (A * D) - (B * C)

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, B, Determinant, cmap="coolwarm", edgecolor='k', alpha=0.7)

# Labels and title
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.set_zlabel("Determinant")
ax.set_title("3D Surface Plot of 2×2 Matrix Determinant")

# Highlight determinant = 0 (singular case)
ax.contour(A, B, Determinant, levels=[0], colors='black', linewidths=2, linestyles="dashed")

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define grid for planes
X, Y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))

# --- Case 1: Unique Solution (Three planes intersect at a point) ---
Z1 = (5 - X - Y) / 2
Z2 = (8 - 2*X + Y) / 3
Z3 = (3 - X + 2*Y) / 4

# --- Case 2: No Solution (Parallel planes) ---
Z4 = (5 - X - Y) / 2
Z5 = (10 - X - Y) / 2  # Parallel to Z4

# --- Case 3: Infinitely Many Solutions (Two coinciding planes) ---
Z6 = (5 - X - Y) / 2
Z7 = (5 - X - Y) / 2  # Identical to Z6

# Function to plot planes
def plot_planes(ax, planes, colors, title):
    for i, (Z, color) in enumerate(zip(planes, colors)):
        ax.plot_surface(X, Y, Z, alpha=0.5, color=color, edgecolor='k')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(title)

# Create figure with 3 subplots
fig = plt.figure(figsize=(15, 5))

# Plot Unique Solution
ax1 = fig.add_subplot(131, projection='3d')
plot_planes(ax1, [Z1, Z2, Z3], ['r', 'g', 'b'], "Unique Solution (Intersection at 1 Point)")

# Plot No Solution
ax2 = fig.add_subplot(132, projection='3d')
plot_planes(ax2, [Z4, Z5], ['y', 'm'], "No Solution (Parallel Planes)")

# Plot Infinitely Many Solutions
ax3 = fig.add_subplot(133, projection='3d')
plot_planes(ax3, [Z6, Z7], ['c', 'c'], "Infinitely Many Solutions (Coinciding Planes)")

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define ranges for a, b, c (fixing d, e, f, g, h, i for simplicity)
a_vals = np.linspace(-5, 5, 50)
b_vals = np.linspace(-5, 5, 50)
A, B = np.meshgrid(a_vals, b_vals)
C = 1  # Fixed c value

# Fixed matrix values
d, e, f = 1, 2, 3
g, h, i = 4, 5, 6

# Compute determinant
Determinant = A * (e * i - f * h) - B * (d * i - f * g) + C * (d * h - e * g)

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, B, Determinant, cmap="coolwarm", edgecolor='k', alpha=0.7)

# Labels and title
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.set_zlabel("Determinant")
ax.set_title("3D Surface Plot of 3×3 Matrix Determinant")

# Highlight determinant = 0 (singular matrices)
ax.contour(A, B, Determinant, levels=[0], colors='black', linewidths=2, linestyles="dashed")

plt.show()
