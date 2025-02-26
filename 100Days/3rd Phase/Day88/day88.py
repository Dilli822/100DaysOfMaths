import numpy as np
import matplotlib.pyplot as plt

# Define two systems of equations
def plot_system(A, b, title):
    x_vals = np.linspace(-10, 10, 100)
    
    # Line equations: Ax = b → y = (-A[0,0]/A[0,1])x + b[0]/A[0,1]
    if A[0,1] != 0 and A[1,1] != 0:
        y1 = (-A[0,0] / A[0,1]) * x_vals + (b[0] / A[0,1])
        y2 = (-A[1,0] / A[1,1]) * x_vals + (b[1] / A[1,1])

        plt.plot(x_vals, y1, label="Equation 1", color='blue')
        plt.plot(x_vals, y2, label="Equation 2", color='red')

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

# Non-singular system (unique solution)
A1 = np.array([[2, 3], [1, 4]])
b1 = np.array([5, 6])
plot_system(A1, b1, "Non-Singular System (Unique Solution)")

# Singular system (infinitely many solutions)
A2 = np.array([[2, 4], [1, 2]])  # Linearly dependent rows
b2 = np.array([5, 6])
plot_system(A2, b2, "Singular System (Infinitely Many Solutions)")


import numpy as np
import matplotlib.pyplot as plt

def plot_lines(a1, b1, c1, a2, b2, c2, title):
    x_vals = np.linspace(-10, 10, 100)
    
    if b1 != 0:
        y1 = (-a1 / b1) * x_vals + (c1 / b1)
        plt.plot(x_vals, y1, label="Equation 1", color="blue")
    
    if b2 != 0:
        y2 = (-a2 / b2) * x_vals + (c2 / b2)
        plt.plot(x_vals, y2, label="Equation 2", color="red")
    
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

# Unique Solution (Intersecting Lines)
plot_lines(2, 1, 5, 1, -1, 1, "Unique Solution (Intersection at One Point)")

# No Solution (Parallel Lines)
plot_lines(2, 1, 5, 4, 2, 10, "No Solution (Parallel Lines)")

# Infinite Solutions (Overlapping Lines)
plot_lines(2, 1, 5, 4, 2, 10, "Infinite Solutions (Same Line)")


import numpy as np
import matplotlib.pyplot as plt

# Define the system of equations
def plot_system(A, b, title):
    x_vals = np.linspace(-10, 10, 100)

    # Compute line equations if not vertical
    if A[0,1] != 0 and A[1,1] != 0:
        y1 = (-A[0,0] / A[0,1]) * x_vals + (b[0] / A[0,1])
        y2 = (-A[1,0] / A[1,1]) * x_vals + (b[1] / A[1,1])

        plt.plot(x_vals, y1, label="Equation 1", color='blue')
        plt.plot(x_vals, y2, label="Equation 2", color='red')

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

# 1. No Solution (Parallel Lines)
A1 = np.array([[2, 1], [4, 2]])
b1 = np.array([3, 8])
plot_system(A1, b1, "No Solution (Parallel Lines)")

# 2. Unique Solution (Intersecting Lines)
A2 = np.array([[1, 1], [2, -1]])
b2 = np.array([5, 1])
plot_system(A2, b2, "One Unique Solution (Intersection)")

# 3. Infinite Solutions (Overlapping Lines)
A3 = np.array([[2, 3], [4, 6]])
b3 = np.array([6, 12])
plot_system(A3, b3, "Infinitely Many Solutions (Same Line)")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to plot 2D system (lines)
def plot_2d_system(A, b, title):
    x_vals = np.linspace(-10, 10, 100)

    # Equations y = (-A[0,0]/A[0,1])x + b[0]/A[0,1]
    y1 = (-A[0,0] / A[0,1]) * x_vals + (b[0] / A[0,1])
    y2 = (-A[1,0] / A[1,1]) * x_vals + (b[1] / A[1,1])

    plt.plot(x_vals, y1, label="Equation 1", color='blue')
    plt.plot(x_vals, y2, label="Equation 2", color='red')

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

# Function to plot 3D system (planes)
def plot_3d_system(A, b, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-10, 10, 10)
    y = np.linspace(-10, 10, 10)
    X, Y = np.meshgrid(x, y)

    # Plane equations Z = (-A[i,0]X - A[i,1]Y + b[i]) / A[i,2]
    Z1 = (-A[0,0] * X - A[0,1] * Y + b[0]) / A[0,2]
    Z2 = (-A[1,0] * X - A[1,1] * Y + b[1]) / A[1,2]
    Z3 = (-A[2,0] * X - A[2,1] * Y + b[2]) / A[2,2]

    ax.plot_surface(X, Y, Z1, alpha=0.5, color='blue', edgecolor='k')
    ax.plot_surface(X, Y, Z2, alpha=0.5, color='red', edgecolor='k')
    ax.plot_surface(X, Y, Z3, alpha=0.5, color='green', edgecolor='k')

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(title)

    plt.show()

# 1. Unique Solution (Intersecting Lines in 2D)
A1 = np.array([[2, -1], [1, 1]])
b1 = np.array([1, 5])
plot_2d_system(A1, b1, "2D System with Unique Solution")

# 2. No Solution (Parallel Lines in 2D)
A2 = np.array([[1, -2], [2, -4]])
b2 = np.array([3, 7])
plot_2d_system(A2, b2, "2D System with No Solution")

# 3. Infinitely Many Solutions (Same Line in 2D)
A3 = np.array([[1, 2], [2, 4]])
b3 = np.array([5, 10])
plot_2d_system(A3, b3, "2D System with Infinitely Many Solutions")

# 4. Unique Solution (Intersecting Planes in 3D)
A4 = np.array([[1, 1, 1], [1, -1, 2], [2, 1, -1]])
b4 = np.array([6, 3, 4])
plot_3d_system(A4, b4, "3D System with Unique Solution")

# 5. No Solution (Parallel Planes in 3D)
A5 = np.array([[1, 2, -1], [2, 4, -2], [1, -1, 3]])
b5 = np.array([3, 7, 5])
plot_3d_system(A5, b5, "3D System with No Solution")

# 6. Infinitely Many Solutions (Same Plane in 3D)
A6 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
b6 = np.array([6, 12, 18])
plot_3d_system(A6, b6, "3D System with Infinitely Many Solutions")


import numpy as np
import matplotlib.pyplot as plt

# Define the equations for different problems
x = np.linspace(0, 10, 100)

# Equation 1: John's age problem J = 2x + 3
J = 2 * x + 3

# Equation 2: Sarah's money problem S = 2d + 5
S = 2 * x + 5

# Equation 3: Car and truck speed problem C = t + 20
C = x + 20

# Plot the equations
plt.figure(figsize=(8, 5))
plt.plot(x, J, label="John's Age: J = 2x + 3", color='blue')
plt.plot(x, S, label="Sarah's Money: S = 2d + 5", color='red')
plt.plot(x, C, label="Car Speed: C = t + 20", color='green')

# Labels and legend
plt.xlabel("x (unknown variable)")
plt.ylabel("Result")
plt.title("Word Problems Translated into Equations")
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Define meshgrid for 3D planes
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Animation function to change plane positions dynamically
def update(frame):
    ax.clear()

    # Transition: Planes start with unique solution and shift to parallel
    Z1 = (6 - X - Y) + frame * 0.2  # Moves upward
    Z2 = (3 - X + 2*Y) - frame * 0.2  # Moves downward
    Z3 = (4 - 2*X - Y)  # Static plane

    # Plot updated planes
    ax.plot_surface(X, Y, Z1, alpha=0.6, color='blue', edgecolor='k')
    ax.plot_surface(X, Y, Z2, alpha=0.6, color='red', edgecolor='k')
    ax.plot_surface(X, Y, Z3, alpha=0.6, color='green', edgecolor='k')

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Animated System of Equations")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 10)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=50, interval=100)
plt.show()
