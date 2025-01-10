import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Function and its derivatives
def f(x):
    return 3 * x**3 - 2 * x**2

def f_prime(x):
    return 9 * x**2 - 4 * x

def f_double_prime(x):
    return 18 * x - 4

# Newton-Raphson Method
def newton_raphson(initial_guess, tol=1e-6, max_iter=100):
    x = initial_guess
    for _ in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fx) < tol:  # Root found
            break
        if fpx == 0:  # Avoid division by zero
            raise ValueError("Derivative is zero. No solution found.")
        x = x - fx / fpx
    return x

# Find critical points
critical_points = np.roots([9, 0, -4])  # Solve 9x^2 - 4x = 0

# Determine concavity and inflection points
inflection_point = np.roots([18, -4])  # Solve 18x - 4 = 0

# 2D Visualization
x = np.linspace(-2, 2, 500)
y = f(x)
y_prime = f_prime(x)
y_double_prime = f_double_prime(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="f(x) = 3x^3 - 2x^2")
plt.plot(x, y_prime, label="f'(x)", linestyle="dashed")
plt.plot(x, y_double_prime, label="f''(x)", linestyle="dotted")

# Highlight critical points and inflection points
plt.scatter(critical_points, f(critical_points), color="red", label="Critical Points")
plt.scatter(inflection_point, f(inflection_point), color="green", label="Inflection Points")

plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("2D Visualization of Function and Derivatives")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# 3D Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

X = np.linspace(-2, 2, 500)
Y = f(X)
Z = f_prime(X)

ax.plot(X, Y, Z, label="f(x) and f'(x) curve", color="blue")
ax.scatter(critical_points, f(critical_points), f_prime(critical_points), color="red", label="Critical Points")
ax.scatter(inflection_point, f(inflection_point), f_prime(inflection_point), color="green", label="Inflection Points")

ax.set_title("3D Visualization of Function and Derivatives")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_zlabel("f'(x)")
ax.legend()
plt.show()


# Function and its derivatives
def f(x):
    return 3 * x**3 - 2 * x**2

def f_prime(x):
    return 9 * x**2 - 4 * x

def f_double_prime(x):
    return 18 * x - 4

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(-2, 2, 500)

# Initialize lines for animation
line_f, = ax.plot([], [], label="f(x) = 3x^3 - 2x^2", color="blue")
line_f_prime, = ax.plot([], [], label="f'(x)", linestyle="dashed", color="orange")
line_f_double_prime, = ax.plot([], [], label="f''(x)", linestyle="dotted", color="green")

# Highlight critical and inflection points
critical_points = np.roots([9, 0, -4])  # Solve 9x^2 - 4x = 0
inflection_point = np.roots([18, -4])  # Solve 18x - 4 = 0
critical_dots, = ax.plot([], [], 'ro', label="Critical Points")
inflection_dots, = ax.plot([], [], 'go', label="Inflection Points")

# Set up the plot limits and labels
ax.set_xlim(-2, 2)
ax.set_ylim(-5, 5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Animated Visualization of Function and Derivatives")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid()

# Animation update function
def update(frame):
    # Calculate current data
    x_frame = x[:frame]
    y_f = f(x_frame)
    y_f_prime = f_prime(x_frame)
    y_f_double_prime = f_double_prime(x_frame)
    
    # Update lines
    line_f.set_data(x_frame, y_f)
    line_f_prime.set_data(x_frame, y_f_prime)
    line_f_double_prime.set_data(x_frame, y_f_double_prime)
    
    # Update critical and inflection points
    critical_dots.set_data(critical_points, f(critical_points))
    inflection_dots.set_data(inflection_point, f(inflection_point))
    
    return line_f, line_f_prime, line_f_double_prime, critical_dots, inflection_dots

# Create the animation
frames = len(x)
ani = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)

plt.show()
