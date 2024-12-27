import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define functions for partial derivatives
def partial_w_r(r, s):
    return 1/s + 12*r

def partial_w_s(r, s):
    return -r/(s**2) + 2*np.cos(s)

# Generate grid for r and s
r = np.linspace(-2, 2, 100)
s = np.linspace(0.1, 2, 100)  # Avoid division by zero
R, S = np.meshgrid(r, s)

# Compute partial derivatives
W_r = partial_w_r(R, S)
W_s = partial_w_s(R, S)

# Create 3D visualization
fig = plt.figure(figsize=(14, 6))

# Partial derivative with respect to r
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(R, S, W_r, cmap='viridis', edgecolor='k', alpha=0.8)
ax1.set_title('Partial Derivative \u2202w/\u2202r')
ax1.set_xlabel('r')
ax1.set_ylabel('s')
ax1.set_zlabel('Value')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# Partial derivative with respect to s
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(R, S, W_s, cmap='plasma', edgecolor='k', alpha=0.8)
ax2.set_title('Partial Derivative \u2202w/\u2202s')
ax2.set_xlabel('r')
ax2.set_ylabel('s')
ax2.set_zlabel('Value')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define the function and its partial derivatives
def fx(x, y):
    return 2 * x  # Partial derivative of f(x, y) with respect to x

def fy(x, y):
    return 2 * y  # Partial derivative of f(x, y) with respect to y

# Define the equation dy/dx = -fx/fy
def dydx(x, y):
    return -fx(x, y) / fy(x, y)

# Create a grid of x, y values
x_vals = np.linspace(-5, 5, 20)
y_vals = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x_vals, y_vals)

# Calculate dy/dx at each point on the grid
Z = dydx(X, Y)

# Create a quiver plot to visualize the slopes
plt.figure(figsize=(6, 6))
plt.quiver(X, Y, np.ones_like(Z), Z, angles='xy', scale_units='xy', scale=1, color='blue')

# Add labels and title
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.title(r'Visualization of $\frac{dy}{dx} = -\frac{f_x}{f_y}$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function and its partial derivatives
def fx(x, y):
    return 2 * x  # Partial derivative of f(x, y) with respect to x

def fy(x, y):
    return 2 * y  # Partial derivative of f(x, y) with respect to y

# Define the equation dy/dx = -fx/fy
def dydx(x, y):
    return -fx(x, y) / fy(x, y)

# Create a grid of x, y values
x_vals = np.linspace(-5, 5, 10)
y_vals = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(x_vals, y_vals)

# Calculate dy/dx at each point on the grid
Z = dydx(X, Y)

# Create a 3D plot to visualize the slopes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot surface for the function f(x, y) = x^2 + y^2
Z_surface = X**2 + Y**2
ax.plot_surface(X, Y, Z_surface, cmap='viridis', alpha=0.7)

# Quiver plot to show the slopes as vectors
ax.quiver(X, Y, Z_surface, np.ones_like(Z), np.ones_like(Z), Z, length=0.2, normalize=True, color='blue')

# Add labels and title
ax.set_title(r'3D Visualization of $\frac{dy}{dx} = -\frac{f_x}{f_y}$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the scalar function f(x, y) = x^2 + 3xy + y^2
def f(x, y):
    return x**2 + 3*x*y + y**2

# Compute the gradient (first-order partial derivatives)
def gradient(x, y):
    df_dx = 2*x + 3*y
    df_dy = 3*x + 2*y
    return np.array([df_dx, df_dy])

# Compute the Hessian (second-order partial derivatives)
def hessian(x, y):
    d2f_dx2 = 2
    d2f_dy2 = 2
    d2f_dxdy = 3
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

# Set up the grid for plotting
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create a 2D plot of the function
fig, ax = plt.subplots(figsize=(8, 6))
CS = ax.contour(X, Y, Z, 20, cmap='viridis')
ax.set_title("2D Contour Plot of f(x, y) = x^2 + 3xy + y^2")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(CS)

# Create a 3D plot of the function
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax_3d.set_title("3D Surface Plot of f(x, y) = x^2 + 3xy + y^2")
ax_3d.set_xlabel("x")
ax_3d.set_ylabel("y")
ax_3d.set_zlabel("f(x, y)")

# Animation setup: Gradient descent steps showing the movement of the point on the function's surface
fig_anim, ax_anim = plt.subplots(figsize=(8, 6))
ax_anim.set_xlim(-5, 5)
ax_anim.set_ylim(-5, 5)
ax_anim.set_title("Gradient Descent Animation with Hessian Influence")
ax_anim.set_xlabel("x")
ax_anim.set_ylabel("y")

# Start position of the gradient descent
start_point = np.array([4, 4])
learning_rate = 0.1
steps = 50

# Create a function to update the animation
def update_frame(frame_num):
    global start_point
    ax_anim.clear()
    ax_anim.set_xlim(-5, 5)
    ax_anim.set_ylim(-5, 5)
    ax_anim.set_title("Gradient Descent Animation with Hessian Influence")
    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")

    # Perform one step of gradient descent
    grad = gradient(start_point[0], start_point[1])
    start_point = start_point - learning_rate * grad

    # Plot the function as a contour and the path of descent
    ax_anim.contour(X, Y, Z, 20, cmap='viridis')
    ax_anim.plot(start_point[0], start_point[1], 'ro')  # Red dot for current position

    return ax_anim,

# Create the animation
ani = FuncAnimation(fig_anim, update_frame, frames=steps, interval=200, blit=False)

# Show all the plots
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function
def f(x, y):
    return x**2 + y**2

# Gradient (first-order partial derivatives)
def grad_f(x, y):
    return np.array([2*x, 2*y])

# Hessian (second-order partial derivatives)
def hessian_f(x, y):
    return np.array([[2, 0], [0, 2]])

# 2D Plot of the function f(x, y) = x^2 + y^2
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Animation: Simulate gradient descent with the gradient and Hessian
x_init, y_init = 4, 4  # Starting point
learning_rate = 0.1
iterations = 30

# Prepare the plot for animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(X, Y, Z, 20, cmap='viridis')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])

# Create a scatter for the initial point
point, = ax.plot([], [], 'ro')

# Update function for animation
def update(frame):
    global x_init, y_init
    grad = grad_f(x_init, y_init)
    hessian = hessian_f(x_init, y_init)
    
    # Update using Newton's method
    step = np.linalg.inv(hessian).dot(grad)
    x_init -= learning_rate * step[0]
    y_init -= learning_rate * step[1]
    
    # Update point coordinates
    point.set_data([x_init], [y_init])
    return point,

# Create animation
ani = FuncAnimation(fig, update, frames=iterations, interval=500, blit=True)
plt.title('Animation of Gradient Descent with Hessian Matrix (Newton\'s Method)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Define the function
def f(x, y):
    return x**2 + y**2

# Gradient (first-order partial derivatives)
def grad_f(x, y):
    return np.array([2*x, 2*y])

# Hessian (second-order partial derivatives)
def hessian_f(x, y):
    return np.array([[2, 0], [0, 2]])

# 3D Plot of the function f(x, y) = x^2 + y^2
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Animation: Simulate optimization
x_init, y_init = 4, 4  # Starting point
learning_rate = 0.1
iterations = 30

# Prepare the plot for animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([0, 50])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

# Create a scatter for the initial point
point, = ax.plot([], [], [], 'ro', markersize=10)

# Update function for animation
def update(frame):
    global x_init, y_init
    grad = grad_f(x_init, y_init)
    hessian = hessian_f(x_init, y_init)
    
    # Update using Newton's method
    step = np.linalg.inv(hessian).dot(grad)
    x_init -= learning_rate * step[0]
    y_init -= learning_rate * step[1]
    z_init = f(x_init, y_init)
    
    # Update point coordinates
    point.set_data([x_init], [y_init])
    point.set_3d_properties([z_init])
    return point,

# Create animation
ani = FuncAnimation(fig, update, frames=iterations, interval=500, blit=True)
plt.title('3D Animation of Gradient Descent with Hessian Matrix (Newton\'s Method)')
plt.show()
