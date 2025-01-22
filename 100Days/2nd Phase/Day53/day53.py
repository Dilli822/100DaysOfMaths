import numpy as np
import matplotlib.pyplot as plt

# Define the curve function f(x, y)
def curve(x):
    return np.sin(x)  # example curve

# Define the gradient (partial derivatives)
def gradient(x):
    return np.cos(x)  # derivative of sin(x)

# Define the tangent and normal line equations
def tangent_line(x, x0, y0):
    A = gradient(x0)
    return A * (x - x0) + y0

def normal_line(x, x0, y0):
    A = gradient(x0)
    return -1 / A * (x - x0) + y0

# Choose a point on the curve
x0 = np.pi / 4  # example point
y0 = curve(x0)

# Generate the x values
x_vals = np.linspace(0, 2 * np.pi, 100)

# Plot the curve
plt.plot(x_vals, curve(x_vals), label='Curve: f(x)', color='blue')

# Plot the tangent and normal lines
plt.plot(x_vals, tangent_line(x_vals, x0, y0), label='Tangent Line', color='green')
plt.plot(x_vals, normal_line(x_vals, x0, y0), label='Normal Line', color='red')

# Highlight the point P0
plt.scatter([x0], [y0], color='black', label=f'P0({x0:.2f}, {y0:.2f})')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('2D Tangent and Normal Lines')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the surface and gradient as before
def surface(x, y):
    return np.sin(np.sqrt(x**2 + y**2))  # example surface

def gradient(x, y):
    df_dx = np.cos(np.sqrt(x**2 + y**2)) * (x / np.sqrt(x**2 + y**2))
    df_dy = np.cos(np.sqrt(x**2 + y**2)) * (y / np.sqrt(x**2 + y**2))
    df_dz = -np.sin(np.sqrt(x**2 + y**2))
    return np.array([df_dx, df_dy, df_dz])

# Initial point on the surface
x0, y0 = 1, 1
z0 = surface(x0, y0)

# Create the mesh grid for the surface with high resolution
x_vals = np.linspace(-5, 5, 500)  # Increased resolution for precision
y_vals = np.linspace(-5, 5, 500)  # Increased resolution for precision
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
z_grid = surface(x_grid, y_grid)

# Create the figure
fig = plt.figure(figsize=(12, 10))  # Larger figure for better clarity
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with enhanced precision
surface_plot = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', edgecolor='none', alpha=0.8)

# Add color bar for better visualization
fig.colorbar(surface_plot, shrink=0.5, aspect=5)

# Define the normal line
t_vals = np.linspace(-5, 5, 100)
normal_line_x = x0 + gradient(x0, y0)[0] * t_vals
normal_line_y = y0 + gradient(x0, y0)[1] * t_vals
normal_line_z = z0 + gradient(x0, y0)[2] * t_vals

# Plot the normal line
normal_line = ax.plot(normal_line_x, normal_line_y, normal_line_z, color='r', label='Normal Line')

# Set up the plot limits
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-2, 2])

# Labels and title for clarity
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface with Normal Line')

# Adjusting the view angle for better perspective
ax.view_init(elev=30, azim=45)

# Show legend
ax.legend()

# Show the plot
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the surface and gradient as in previous code
def surface(x, y):
    return np.sin(np.sqrt(x**2 + y**2))  # example surface

def gradient(x, y):
    df_dx = np.cos(np.sqrt(x**2 + y**2)) * (x / np.sqrt(x**2 + y**2))
    df_dy = np.cos(np.sqrt(x**2 + y**2)) * (y / np.sqrt(x**2 + y**2))
    df_dz = -np.sin(np.sqrt(x**2 + y**2))
    return np.array([df_dx, df_dy, df_dz])

# Initial surface and point
x0, y0 = 1, 1
z0 = surface(x0, y0)

# Create the mesh grid for the surface with higher resolution
x_vals = np.linspace(-5, 5, 300)  # Increased resolution
y_vals = np.linspace(-5, 5, 300)  # Increased resolution
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
z_grid = surface(x_grid, y_grid)

# Set up the figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', alpha=0.6, edgecolor='none')

# Define the normal line
t_vals = np.linspace(-5, 5, 100)
normal_line_x = x0 + gradient(x0, y0)[0] * t_vals
normal_line_y = y0 + gradient(x0, y0)[1] * t_vals
normal_line_z = z0 + gradient(x0, y0)[2] * t_vals

# Plot the initial normal line
line, = ax.plot([], [], [], color='r', label='Normal Line')

# Set up the plot limits with tighter precision
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-2, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Normal Line Animation')

# Adjusting the view angle for better perspective
ax.view_init(elev=30, azim=45)

# Update function for the animation with increased precision
def update(t):
    line.set_data(normal_line_x[:t], normal_line_y[:t])
    line.set_3d_properties(normal_line_z[:t])
    return line,

# Create the animation with increased frame rate
ani = FuncAnimation(fig, update, frames=len(t_vals), interval=40, blit=True)

# Show the plot
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import math

# Define the function and its Taylor series
def f(x):
    return np.exp(x)

# Taylor series approximation for e^x
def taylor_approximation(x, n_terms):
    approximation = sum([x**i / math.factorial(i) for i in range(n_terms)])  # Use math.factorial here
    return approximation

x = np.linspace(-2, 2, 400)
y_exact = f(x)

# Plot different Taylor series approximations
plt.plot(x, y_exact, label='e^x (Exact)', color='blue')

# Corrected the range to np.arange
for n_terms in np.arange(1, 25, 2):  # Plot only for odd terms for clarity
    y_approx = taylor_approximation(x, n_terms)
    plt.plot(x, y_approx, label=f'Taylor Approx. with {n_terms} terms')

plt.legend()
plt.title("Taylor Series Approximation to e^x")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Maclaurin series approximation for sin(x)
def sin_maclaurin(x, n_terms):
    approximation = sum([((-1)**(i//2)) * (x**(2*i+1)) / math.factorial(2*i+1) for i in range(n_terms)])
    return approximation

# Plotting Maclaurin Series Approximation to sin(x)
x = np.linspace(-2, 2, 400)
y_exact = np.sin(x)

plt.plot(x, y_exact, label='sin(x) (Exact)', color='green')

for n_terms in [1, 3, 5, 7]:
    y_approx = sin_maclaurin(x, n_terms)
    plt.plot(x, y_approx, label=f'Maclaurin Approx. with {n_terms} terms')

plt.legend()
plt.title("Maclaurin Series Approximation to sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# 2D Taylor Formula Approximation
# Function f(x, y)
def f_2d(x, y):
    return np.exp(x + y)

# Gradient of f(x, y)
def gradient_2d(x, y):
    df_dx = np.exp(x + y)
    df_dy = np.exp(x + y)
    return np.array([df_dx, df_dy])

x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-2, 2, 400)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
z_grid = f_2d(x_grid, y_grid)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of f(x, y)
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', alpha=0.6)

# Choose a point (x0, y0) for Taylor expansion
x0, y0 = 0, 0
z0 = f_2d(x0, y0)

# Plot tangent plane at (x0, y0)
def tangent_plane_2d(x, y, x0, y0, z0, gradient):
    grad = gradient(x0, y0)
    return grad[0] * (x - x0) + grad[1] * (y - y0) + z0

z_tangent = tangent_plane_2d(x_grid, y_grid, x0, y0, z0, gradient_2d)

ax.plot_surface(x_grid, y_grid, z_tangent, cmap='coolwarm', alpha=0.3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('2D Taylor Formula Approximation')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 2D function example: f(x) = x^3 - 3x^2 + 2
def func_2d(x):
    return x**3 - 3*x**2 + 2

# Values for the plot
x = np.linspace(-2, 3, 400)
y = func_2d(x)

# Create a 2D plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, label="f(x) = x^3 - 3x^2 + 2")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.axhline(0, color="black",linewidth=0.5)
ax.axvline(0, color="black",linewidth=0.5)
ax.grid(True)
ax.set_title("2D Function with Local Minima and Maxima")

# Plotting the local minima and maxima
min_x = 1  # Local minimum at x = 1
max_x = 0  # Local maximum at x = 0

ax.scatter(min_x, func_2d(min_x), color='red', zorder=5, label="Local Minima")
ax.scatter(max_x, func_2d(max_x), color='green', zorder=5, label="Local Maxima")

# Labeling the points
ax.text(min_x, func_2d(min_x), f"Min ({min_x}, {func_2d(min_x):.2f})", fontsize=12, color='red', ha='right')
ax.text(max_x, func_2d(max_x), f"Max ({max_x}, {func_2d(max_x):.2f})", fontsize=12, color='green', ha='right')

ax.legend()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# 3D function: f(x, y) = sin(x^2 + y^2)
def func_3d(x, y):
    return np.sin(x**2 + y**2)

# Values for the plot
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
Z = func_3d(X, Y)

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.7)

# Add labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('f(x, y) = sin(x^2 + y^2)')
ax.set_title('3D Surface with Local and Absolute Minima/Maxima')

# Mark local maxima (highest point) and minima (lowest point)
local_max_x, local_max_y = 0, 0  # Local maxima
local_min_x, local_min_y = 2, 2  # Local minima

# Plot local maxima
ax.scatter(local_max_x, local_max_y, func_3d(local_max_x, local_max_y), color='red', s=100, label='Local Maxima')

# Plot local minima
ax.scatter(local_min_x, local_min_y, func_3d(local_min_x, local_min_y), color='blue', s=100, label='Local Minima')

# Show legend
ax.legend()

plt.show()

from matplotlib.animation import FuncAnimation

# Animation of the 3D surface with highlighting the minima and maxima
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Define the update function for animation
def update(frame):
    ax.clear()
    ax.plot_surface(X, Y, Z * np.cos(frame / 10), cmap='viridis', edgecolor='k', alpha=0.7)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('f(x, y) = sin(x^2 + y^2)')
    ax.set_title(f'3D Surface Animation: Frame {frame}')
    
    # Highlight local maxima and minima for each frame
    ax.scatter(local_max_x, local_max_y, func_3d(local_max_x, local_max_y), color='red', s=100, label='Local Maxima')
    ax.scatter(local_min_x, local_min_y, func_3d(local_min_x, local_min_y), color='blue', s=100, label='Local Minima')
    ax.legend()

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=100, repeat=True)

plt.show()
