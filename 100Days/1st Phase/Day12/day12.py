import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Piecewise function definition
def f(x):
    if 1 <= x <= 2:
        return 2 - x
    elif 2 < x <= 4:
        return 3
    else:
        return np.nan  # To handle undefined values outside the range

# Create a range of x values
x = np.linspace(0, 4, 100)
y = [f(xi) for xi in x]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the piecewise function
ax.plot(x, y, lw=2, label='f(x)')

# Add a moving point along the curve (e.g., at x = 2)
x_point = 2
y_point = f(x_point)
ax.plot(x_point, y_point, 'ro', label=f'Point at x = {x_point}')

# Adding labels, title, and grid
ax.set_title('Piecewise Function with a Moving Point')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.grid(True)
ax.legend()

# Display the plot
plt.show()

# Define a continuous function
def f(x):
    return x**3 - 4*x - 9  # This has a root between 2 and 3

# Create an array of x values for plotting
x = np.linspace(1, 4, 100)

# Evaluate the function at each point in x
y = f(x)

# Plot the function
plt.plot(x, y, label='$f(x) = x^3 - 4x - 9$', color='blue')

# Highlight the points where the function changes sign
x_sign_change = [2, 3]  # These points have opposite signs in f(x)
y_sign_change = [f(xi) for xi in x_sign_change]

# Mark these points on the plot
plt.plot(x_sign_change, y_sign_change, 'ro', label='Sign change points')

# Add the line y = 0 for reference (root line)
plt.axhline(0, color='black',linewidth=1)

# Add labels and title
plt.title('Intermediate Value Theorem (IVT) Demonstration')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# Define a continuous 3D function
def f(x, y):
    return x**3 - y**2 - 9  # This function has a root around x = 2 and y = 3

# Create a meshgrid for the x and y values
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)

# Evaluate the function on the grid
Z = f(X, Y)

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)

# Add labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('f(x, y)')
ax.set_title('3D Plot for Intermediate Value Theorem (IVT)')

# Mark the region where the function changes sign
ax.scatter(2, 3, f(2, 3), color='r', s=100, label='Root region (sign change)')

# Add color bar for reference
fig.colorbar(surf)

# Display the plot
plt.legend()
plt.show()