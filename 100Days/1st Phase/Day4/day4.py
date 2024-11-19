import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the range for theta and the corresponding sine and cosine values
theta = np.linspace(-0.5, 0.5, 1000)  # Range of theta
sin_theta = np.sin(theta)
cos_theta = np.cos(theta)

# Create a figure and axis for plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Initialize the sine and cosine plots
sin_line, = ax.plot([], [], color='blue', label=r'$\sin(\theta)$', linewidth=2)
cos_line, = ax.plot([], [], color='green', label=r'$\cos(\theta)$', linewidth=2)

# Highlight the limits as dots
limit_sin, = ax.plot(0, 0, 'o', color='blue', label=r'$\sin(0) = 0$', markersize=8)
limit_cos, = ax.plot(0, 1, 'o', color='green', label=r'$\cos(0) = 1$', markersize=8)

# Configure the plot
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-1.5, 1.5)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_title(r'Behavior of $\sin(\theta)$ and $\cos(\theta)$ as $\theta \to 0$', fontsize=16)
ax.set_xlabel(r'$\theta$', fontsize=14)
ax.set_ylabel('Function Value', fontsize=14)
ax.legend(loc='upper right', fontsize=12)

# Initialization function
def init():
    sin_line.set_data([], [])
    cos_line.set_data([], [])
    return sin_line, cos_line

# Update function
def update(frame):
    sin_line.set_data(theta[:frame], sin_theta[:frame])
    cos_line.set_data(theta[:frame], cos_theta[:frame])
    return sin_line, cos_line

# Create the animation
ani = FuncAnimation(fig, update, frames=len(theta), init_func=init, blit=True, interval=20)

# Display the animation
plt.show()


# Define the functions
def f(x):
    return x**2  # Left-hand function

def g(x):
    return 3*x - 4  # Right-hand function

# Create a range of x values around x = 2
x_left = np.linspace(0, 2, 500)  # Values approaching from the left
x_right = np.linspace(2, 4, 500)  # Values approaching from the right

# Calculate the corresponding y values
y_left = f(x_left)
y_right = g(x_right)

# Plot the functions
plt.figure(figsize=(8, 6))
plt.plot(x_left, y_left, label=r"$f(x) = x^2$ (left-hand side)", color='blue')
plt.plot(x_right, y_right, label=r"$g(x) = 3x - 4$ (right-hand side)", color='red')

# Mark the x = 2 point
plt.axvline(x=2, color='black', linestyle='--', label="x = 2")

# Adding labels and title
plt.xlabel("x")
plt.ylabel("y")
plt.title("Left-hand and Right-hand Limits at x = 2")
plt.legend()

# Show grid
plt.grid(True)

# Display the plot
plt.show()


# Define the function
def f(x):
    return x**2  # Function where limit exists

# Create a range of x values around x = 2
x_left = np.linspace(0, 2, 500)  # Values approaching from the left
x_right = np.linspace(2, 4, 500)  # Values approaching from the right

# Calculate the corresponding y values
y_left = f(x_left)  # f(x) values for left-hand side
y_right = f(x_right)  # f(x) values for right-hand side

# Plot the function where the limit exists
plt.figure(figsize=(8, 6))
plt.plot(x_left, y_left, label=r"$f(x) = x^2$ (left-hand side)", color='blue')
plt.plot(x_right, y_right, label=r"$f(x) = x^2$ (right-hand side)", color='red', linestyle='--')

# Mark the x = 2 point
plt.axvline(x=2, color='black', linestyle='--', label="x = 2")

# Adding labels and title
plt.xlabel("x")
plt.ylabel("y")
plt.title("Left-hand and Right-hand Limits at x = 2 (Limit Exists)", fontsize=14)
plt.legend()

# Show grid
plt.grid(True)

# Display the plot
plt.show()
