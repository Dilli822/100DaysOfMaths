
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Function definitions
def f(x):
    return np.sin(x)

def g(x):
    return x

def h(x):
    return f(x) / g(x)

# Derivatives for L'Hôpital's Rule
def f_prime(x):
    return np.cos(x)

def g_prime(x):
    return 1

def h_prime(x):
    return f_prime(x) / g_prime(x)

# Generate x values
x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
x_safe = x[np.abs(x) > 1e-6]  # Avoid division by zero for visualization

# 2D Plot
plt.figure(figsize=(10, 6))
plt.plot(x_safe, h(x_safe), label=r"$h(x)=\frac{\sin(x)}{x}$", color="blue")
plt.axhline(1, color="green", linestyle="--", label=r"Limit as $x \to 0$ (L'Hôpital's Rule)")
plt.title("2D Visualization of L'Hôpital's Rule")
plt.xlabel("x")
plt.ylabel("h(x)")
plt.legend()
plt.grid()
plt.show()

# 3D Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
X = np.linspace(-2 * np.pi, 2 * np.pi, 500)
Y = X.copy()
X, Y = np.meshgrid(X, Y)
Z = np.sin(X) / Y
Z[np.abs(Y) < 1e-6] = np.nan  # Avoid division by zero

ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
ax.set_title("3D Visualization of L'Hôpital's Rule")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("h(x) = sin(x)/x")
plt.show()

# Animation
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], color="blue", label="h(x) = sin(x)/x")
point, = ax.plot([], [], "ro")  # Point to highlight the current frame
ax.axhline(1, color="green", linestyle="--", label="Limit as x → 0")
ax.set_xlim(-2 * np.pi, 2 * np.pi)
ax.set_ylim(-0.5, 1.5)
ax.set_title("Animated Visualization of L'Hôpital's Rule")
ax.set_xlabel("x")
ax.set_ylabel("h(x)")
ax.legend()
ax.grid()

def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

def update(frame):
    x_data = x_safe[:frame]  # All x points up to the current frame
    y_data = h(x_safe[:frame])  # Corresponding y points
    line.set_data(x_data, y_data)
    
    # Set the current point as a sequence
    current_x = [x_safe[frame - 1]]
    current_y = [h(x_safe[frame - 1])]
    point.set_data(current_x, current_y)
    return line, point

ani = FuncAnimation(fig, update, frames=len(x_safe), init_func=init, blit=True, interval=30)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Parameters
eta = 0.1  # learning rate
theta_0 = 5  # initial parameter
t_max = 10  # maximum time
frames = 200  # number of animation frames
t = np.linspace(0, t_max, frames)  # time steps

# Solution to the ODE (Gradient Descent: theta(t) = theta_0 * exp(-eta * t))
theta_t = theta_0 * np.exp(-eta * t)

# 1. 2D Plot: Evolution of the parameter theta(t)
plt.figure(figsize=(8, 6))
plt.plot(t, theta_t, label=r'$\theta(t) = \theta_0 e^{-\eta t}$')
plt.title('Gradient Descent: 2D Plot')
plt.xlabel('Time (t)')
plt.ylabel('Parameter ($\theta$)')
plt.grid(True)
plt.legend()
plt.show()

# 2. 3D Plot: Loss function surface and parameter evolution in 3D
# Loss function L(theta) = 1/2 * theta^2
theta_vals = np.linspace(-theta_0, theta_0, 100)
L_vals = 0.5 * theta_vals**2

# Create the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the loss surface
ax.plot(theta_vals, L_vals, zs=0, zdir='z', label='Loss function L($\theta$)', color='b')

# Plot the parameter evolution in 3D (as a path)
ax.plot(theta_t, 0.5 * theta_t**2, zs=0, zdir='z', label=r'$\theta(t)$ path', color='r')

ax.set_title('Gradient Descent: 3D Plot')
ax.set_xlabel('Parameter ($\theta$)')
ax.set_ylabel('Loss function ($L(\theta)$)')
ax.set_zlabel('Time (t)')
plt.legend()
plt.show()

# 3. Animation: Evolution of theta(t) over time
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-theta_0-1, theta_0+1)
ax.set_ylim(0, 0.5*theta_0**2 + 1)

# Loss function plot (background)
theta_vals_anim = np.linspace(-theta_0, theta_0, 100)
L_vals_anim = 0.5 * theta_vals_anim**2
ax.plot(theta_vals_anim, L_vals_anim, label=r'$L(\theta) = \frac{1}{2} \theta^2$', color='b')

# Point to animate (evolving theta)
point, = ax.plot([], [], 'ro')

# Initialization function for animation
def init():
    point.set_data([], [])
    return point,

# Animation function: update the parameter (theta) at each time step
# Animation function: update the parameter (theta) at each time step
def animate(i):
    x = theta_t[i]
    y = 0.5 * x**2
    point.set_data([x], [y])  # x and y are now sequences (lists)
    return point,


# Create the animation
ani = FuncAnimation(fig, animate, frames=frames, init_func=init, blit=True, interval=100)

plt.title('Gradient Descent: Animation of Parameter Evolution')
plt.xlabel('Parameter ($\theta$)')
plt.ylabel('Loss function ($L(\theta)$)')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Parameters
k = 0.5  # Growth rate
C = 1    # Initial condition (C)
x = np.linspace(0, 10, 100)
y = C * np.exp(k * x)

# 2D plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$y = C e^{kx}$')
plt.title('2D Plot of the ODE Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Generate data for 3D plot
x = np.linspace(0, 10, 100)
z = np.sin(x)
y = C * np.exp(k * x)

# Create a meshgrid for 3D plotting
X, Z = np.meshgrid(x, z)
Y = C * np.exp(k * X)

# 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Z, Y, cmap='viridis')

ax.set_title('3D Plot of the ODE Solution')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
plt.show()

import matplotlib.animation as animation

# Function to animate the plot
def animate(i):
    x = np.linspace(0, 10, 100)
    y = C * np.exp(k * x)
    line.set_data(x[:i], y[:i])
    return line,

# Create the figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title('Animation of the ODE Solution')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Line object for animation
line, = ax.plot([], [], lw=2)

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=len(x), interval=50, blit=True)

# Show the animation
plt.show()
