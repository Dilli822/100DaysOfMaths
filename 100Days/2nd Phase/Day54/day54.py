
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 1. Lagrange's Multipliers (Constrained Optimization) - 3D Plot
def lagrange_visualization():
    # Define the objective function: f(x, y) = x^2 + y^2
    def f(x, y):
        return x**2 + y**2

    # Define the constraint: g(x, y) = x + y - 1 = 0
    def g(x, y):
        return x + y - 1

    # Create grid points
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Plot the objective function and constraint
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Constraint curve (g(x, y) = 0) on the xy-plane
    x_constraint = np.linspace(-2, 2, 100)
    y_constraint = 1 - x_constraint  # From g(x, y) = x + y - 1 = 0
    z_constraint = f(x_constraint, y_constraint)
    ax.plot(x_constraint, y_constraint, z_constraint, color='red', label='Constraint (g(x, y) = 0)', linewidth=3)

    ax.set_title("Lagrange Multipliers - Constrained Optimization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("f(x, y)")
    ax.legend()
    plt.show()


# 3. Constrained Maxima and Minima - 2D Plot
def constrained_minima_visualization():
    # Define the function: f(x, y) = x^2 + y^2
    def f(x, y):
        return x**2 + y**2

    # Constraint: x + y = 1
    def constraint(x):
        return 1 - x

    # Generate data for plotting
    x = np.linspace(-1, 2, 500)
    y = constraint(x)
    z = f(x, y)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Constraint: x + y = 1", color='red')
    plt.scatter(x, y, c=z, cmap='viridis', label="Objective Function (f(x, y))")
    plt.colorbar(label="Objective Value f(x, y)")
    plt.title("Constrained Minima Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()

# Call Functions
lagrange_visualization()      # 3D visualization of Lagrange's method
  
constrained_minima_visualization()  # 2D constrained maxima/minima

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Constrained Optimization Visualization (2D Plot)
def constrained_visualization():
    # Objective Function: f(x, y) = x^2 + y^2
    def f(x, y):
        return x**2 + y**2

    # Constraint: x + y = 1 (A line)
    def constraint(x):
        return 1 - x

    # Generate Data
    x = np.linspace(-1, 2, 500)  # Range of x values
    y = constraint(x)            # Compute y based on the constraint
    z = f(x, y)                  # Compute the objective function values

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Constraint: x + y = 1", color='red', linewidth=2)
    plt.scatter(x, y, c=z, cmap='viridis', label="Objective Function (f(x, y))")
    plt.colorbar(label="Value of f(x, y)")
    plt.title("Constrained Optimization (Lagrange's Multipliers)")
    plt.xlabel("X (Variable 1)")
    plt.ylabel("Y (Variable 2)")
    plt.legend()
    plt.grid()
    plt.show()

# 2. Fubini's Theorem Animation (Integration Visualization)
def fubini_animation():
    # Function for the area under z = f(x, y)
    def f(x, y):
        return x * y

    x = np.linspace(0, 2, 100)  # X range
    y = np.linspace(0, 2, 100)  # Y range
    X, Y = np.meshgrid(x, y)    # Create a grid of x, y values
    Z = f(X, Y)                 # Compute Z values for f(x, y)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Fubini's Theorem (Integration Visualization)")
    ax.set_xlabel("X (Width)")
    ax.set_ylabel("Y (Height)")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)

    # Initialize plot
    area = ax.fill_between(x, 0, 0, color='skyblue', alpha=0.5, label="Accumulated Area")
    line, = ax.plot([], [], color='red', linewidth=2, label="Current Slice")

    # Update function for animation
    def update(frame):
        ax.clear()
        ax.set_title("Fubini's Theorem (Integration Visualization)")
        ax.set_xlabel("X (Width)")
        ax.set_ylabel("Y (Height)")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)

        # Plot updated area
        ax.fill_between(x[:frame], 0, f(x[:frame], 1), color='skyblue', alpha=0.5, label="Accumulated Area")
        ax.plot(x[:frame], f(x[:frame], 1), color='red', linewidth=2, label="Current Slice")
        ax.legend()

    # Animate
    ani = FuncAnimation(fig, update, frames=100, interval=100)
    plt.show()

# Call the Visualizations
constrained_visualization()  # Plot the constrained optimization problem
fubini_animation()           # Show the integration animation



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 1. Constrained Optimization in 3D
def constrained_optimization_3d():
    # Objective Function: f(x, y) = x^2 + y^2
    def f(x, y):
        return x**2 + y**2

    # Constraint: x + y = 1 (Line in 3D)
    def constraint(x):
        return 1 - x

    # Generate data
    x = np.linspace(-1, 2, 500)
    y = np.linspace(-1, 2, 500)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)  # Surface values

    # Constraint curve
    x_line = np.linspace(0, 1, 100)
    y_line = constraint(x_line)
    z_line = f(x_line, y_line)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none', label="Objective Surface")
    ax.plot(x_line, y_line, z_line, color='red', linewidth=3, label="Constraint Curve")
    ax.set_title("3D Constrained Optimization (Lagrange's Multipliers)")
    ax.set_xlabel("X (Variable 1)")
    ax.set_ylabel("Y (Variable 2)")
    ax.set_zlabel("f(X, Y)")
    ax.legend(loc='upper left')
    plt.show()

# 2. 3D Integration Animation
def integration_animation_3d():
    # Function for 3D surface: f(x, y) = x * y
    def f(x, y):
        return x * y

    # Generate data
    x = np.linspace(0, 2, 100)
    y = np.linspace(0, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(0, 4)
    ax.set_title("3D Integration Animation (Fubini's Theorem)")
    ax.set_xlabel("X (Width)")
    ax.set_ylabel("Y (Height)")
    ax.set_zlabel("f(X, Y)")

    # Surface plot
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    
    # Initialize a valid surface (optional, if needed to set the background initially)
    slice_surface = ax.plot_surface(X, Y, Z, color='blue', alpha=0.7)  # First surface plot for background
    
    def update(frame):
        # Remove the previous surface plot (if it exists)
        for collection in ax.collections:
            collection.remove()  # Remove each surface plot

        # Generate new slice data
        x_slice = np.linspace(0, frame / 50, 50)  # Growing slice
        y_slice = np.linspace(0, 2, 50)
        X_slice, Y_slice = np.meshgrid(x_slice, y_slice)
        Z_slice = f(X_slice, Y_slice)
        
        # Plot new surface for the growing slice
        ax.plot_surface(X_slice, Y_slice, Z_slice, cmap='cool', alpha=0.8)
        ax.text2D(0.1, 0.9, f"Frame {frame}", transform=ax.transAxes)  # Display current frame
    
    ani = FuncAnimation(fig, update, frames=100, interval=100)
    plt.show()

# Call the Visualizations
constrained_optimization_3d()  # 3D constrained optimization
integration_animation_3d()     # 3D integration animation


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Create a grid of x and y values (from 0 to 1)
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
X, Y = np.meshgrid(x, y)

# Calculate Z values for the surface
Z = f(X, Y)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Add labels
ax.set_xlabel('X-axis (x)')
ax.set_ylabel('Y-axis (y)')
ax.set_zlabel('Function value (f(x, y))')

# Add a title with description on the plot
ax.set_title('Double Integral: f(x, y) = x^2 + y^2\nArea Under Surface from x=0 to 1, y=0 to 1')

# Show the plot
plt.show()


# Define the function f(x, y, z) = x + y
def f3(x, y, z):
    return x + y

# Create a grid for x, y, z values
x = np.linspace(0, 1, 30)
y = np.linspace(0, 1, 30)
z = np.linspace(0, 1, 30)
X, Y, Z = np.meshgrid(x, y, z)

# Calculate function values
F = f3(X, Y, Z)

# Create a 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.scatter(X, Y, Z, c=F, cmap='plasma')

# Add labels
ax.set_xlabel('X-axis (x)')
ax.set_ylabel('Y-axis (y)')
ax.set_zlabel('Z-axis (z)')

# Add a title with description on the plot
ax.set_title('Triple Integral: f(x, y, z) = x + y\nVolume Under Surface in 3D space (0 <= x, y, z <= 1)')

# Show the plot
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Create a grid of x and y values (from 0 to 1)
x = np.linspace(0, 1, 50)  # x values from 0 to 1
y = np.linspace(0, 1, 50)  # y values from 0 to 1
X, Y = np.meshgrid(x, y)   # Create a 2D grid of x and y values

# Calculate Z values for the surface
Z = f(X, Y)

# Create a figure for plotting
plt.figure(figsize=(8, 6))

# Plot the surface as a contour plot
cp = plt.contourf(X, Y, Z, cmap='viridis')

# Add color bar to show the range of function values
plt.colorbar(cp)

# Add labels and title
plt.xlabel('X-axis (x)')
plt.ylabel('Y-axis (y)')
plt.title('Double Integral: f(x, y) = x^2 + y^2')

# Show the plot
plt.show()


# Define the function f(x, y) = sin(x) + cos(y)
def f_sin_cos(x, y):
    return np.sin(x) + np.cos(y)

# Calculate Z values for the new function
Z_sin_cos = f_sin_cos(X, Y)

# Create a figure for plotting
plt.figure(figsize=(8, 6))

# Plot the surface as a contour plot
cp = plt.contourf(X, Y, Z_sin_cos, cmap='inferno')

# Add color bar to show the range of function values
plt.colorbar(cp)

# Add labels and title
plt.xlabel('X-axis (x)')
plt.ylabel('Y-axis (y)')
plt.title('Double Integral: f(x, y) = sin(x) + cos(y)')

# Show the plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Create a grid of x and y values (from 0 to 1)
x = np.linspace(0, 1, 50)  # x values from 0 to 1
y = np.linspace(0, 1, 50)  # y values from 0 to 1
X, Y = np.meshgrid(x, y)   # Create a 2D grid of x and y values

# Calculate Z values for the surface
Z = f(X, Y)

# Create a figure for plotting
plt.figure(figsize=(10, 8))

# Plot the surface as a contour plot
cp = plt.contourf(X, Y, Z, cmap='viridis')

# Add color bar to show the range of function values
plt.colorbar(cp)

# Add labels and title
plt.xlabel('X-axis (x)', fontsize=12)
plt.ylabel('Y-axis (y)', fontsize=12)
plt.title('Double Integral: f(x, y) = x^2 + y^2', fontsize=14)

# Annotating key areas and points for better understanding

# Text annotation for a key feature (e.g., at the origin)
plt.text(0.05, 0.05, "Origin (0,0)", color="white", fontsize=12)

# Text annotation for a point (e.g., near the upper right corner)
plt.text(0.8, 0.8, "Point (0.8, 0.8)", color="white", fontsize=12)

# Add an explanation of the integral
plt.text(0.1, 0.7, "Double Integral: Sum of all values\nunder this surface", color="white", fontsize=12)

# Add explanation about the function itself
plt.text(0.1, 0.4, "f(x, y) = x^2 + y^2 increases as x or y increases", color="white", fontsize=12)

# Add more general annotation
plt.text(0.3, 0.3, "Each contour represents a constant value of f(x, y)", color="white", fontsize=12)

# Show the plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Create a grid of x and y values (from 0 to 1)
x = np.linspace(0, 1, 50)  # x values from 0 to 1
y = np.linspace(0, 1, 50)  # y values from 0 to 1
X, Y = np.meshgrid(x, y)   # Create a 2D grid of x and y values

# Calculate Z values for the surface
Z = f(X, Y)

# Create a figure for plotting
plt.figure(figsize=(12, 8))

# Plot the surface as a contour plot
cp = plt.contourf(X, Y, Z, cmap='viridis')

# Add color bar to show the range of function values
plt.colorbar(cp)

# Add labels and title
plt.xlabel('X-axis (x)', fontsize=12)
plt.ylabel('Y-axis (y)', fontsize=12)
plt.title('Double Integral: f(x, y) = x^2 + y^2 (Fubini’s Theorem)', fontsize=14)

# Annotating key areas and points for better understanding

# Text annotation for a key feature (e.g., at the origin)
plt.text(0.05, 0.05, "Origin (0,0)", color="white", fontsize=12)

# Text annotation for a point (e.g., near the upper right corner)
plt.text(0.8, 0.8, "Point (0.8, 0.8)", color="white", fontsize=12)

# Annotating Fubini's Theorem concept
plt.text(0.1, 0.7, "First Integral: Integrating over y\nthen x: ∫( ∫f(x,y) dy ) dx", color="white", fontsize=12)

# Annotating the second order of integration
plt.text(0.1, 0.4, "Second Integral: Integrating over x\nthen y: ∫( ∫f(x,y) dx ) dy", color="white", fontsize=12)

# Showing the integration approach with vertical and horizontal lines
plt.plot([0.2, 0.2], [0, 0.8], color="red", linestyle='--')  # Vertical line for first integration w.r.t y
plt.plot([0.2, 0.8], [0.8, 0.8], color="blue", linestyle='--')  # Horizontal line for first integration w.r.t x

# Show the plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Objective function f(x, y) = 4x^2 + 9y^2
def objective(x, y):
    return 4*x**2 + 9*y**2

# Constraint function g(x, y) = x + y - 2
def constraint(x, y):
    return x + y - 2

# Create a grid for plotting
x = np.linspace(-2, 3, 400)
y = np.linspace(-2, 3, 400)
X, Y = np.meshgrid(x, y)

# Calculate the objective function values
Z = objective(X, Y)

# Plot the surface of the objective function
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=np.linspace(0, 50, 11), cmap='viridis')

# Plot the constraint line (x + y = 2)
constraint_x = np.linspace(-2, 3, 400)
constraint_y = 2 - constraint_x
plt.plot(constraint_x, constraint_y, color='r', linewidth=2, label='Constraint: x + y = 2')

# Labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Lagrange\'s Multipliers: Maximize f(x, y) = 4x^2 + 9y^2\nsubject to x + y = 2')

# Add a color bar to the contour plot
plt.colorbar(contour)

# Mark the intersection of the constraint with the objective function
x_optimal = 1  # Solution to the optimization problem
y_optimal = 1  # Solution to the optimization problem
plt.plot(x_optimal, y_optimal, 'bo', label=f'Optimal Point ({x_optimal}, {y_optimal})')

# Show the plot
plt.legend()
plt.grid(True)
plt.show()
