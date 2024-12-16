import numpy as np
import matplotlib.pyplot as plt

# Function to visualize MVT
def visualize_mvt():
    # Define a continuous and differentiable function
    def f(x):
        return x**2 - 3*x + 2  # Example: A parabola
    
    a, b = 0, 3  # Interval [a, b]
    x = np.linspace(a - 1, b + 1, 500)
    y = f(x)
    
    # Slope of the secant line
    secant_slope = (f(b) - f(a)) / (b - a)
    
    # Derivative of f(x)
    def f_prime(x):
        return 2*x - 3
    
    # Solve f'(c) = secant_slope
    c_mvt = (secant_slope + 3) / 2  # Solved from 2c - 3 = secant_slope

    # Secant line equation
    def secant_line(x):
        return secant_slope * (x - a) + f(a)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="f(x) = x² - 3x + 2")
    plt.scatter([a, b], [f(a), f(b)], color="red", label="Endpoints (a, f(a)) and (b, f(b))")
    plt.plot(x, secant_line(x), linestyle="--", color="orange", label="Secant Line")
    plt.scatter([c_mvt], [f(c_mvt)], color="green", label=f"Point c (MVT): x = {c_mvt:.2f}")
    plt.axvline(c_mvt, color="green", linestyle=":", alpha=0.7)

    plt.title("Visualization of the Mean Value Theorem (MVT)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()

# Function to visualize Rolle's Theorem
def visualize_rolles():
    # Define a continuous and differentiable function
    def f(x):
        return (x - 1)*(x - 3)  # Example: A parabola with f(a) = f(b) = 0

    a, b = 1, 3  # Interval [a, b]
    x = np.linspace(a - 1, b + 1, 500)
    y = f(x)

    # Derivative of f(x)
    def f_prime(x):
        return 2*x - 4

    # Solve f'(c) = 0
    c_rolle = 2  # Solved from 2c - 4 = 0

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="f(x) = (x - 1)(x - 3)")
    plt.scatter([a, b], [f(a), f(b)], color="red", label="Endpoints (a, f(a)) and (b, f(b))")
    plt.scatter([c_rolle], [f(c_rolle)], color="green", label=f"Point c (Rolle's): x = {c_rolle}")
    plt.axvline(c_rolle, color="green", linestyle=":", alpha=0.7)

    plt.title("Visualization of Rolle's Theorem")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()

# Visualize both
visualize_mvt()
visualize_rolles()


# Function to visualize Rolle's Theorem
def visualize_rolles_sqrt():
    # Define the function and its derivative
    def f(x):
        return np.sqrt(1 - x**2)
    
    # Interval [a, b]
    a, b = -1, 1  # f(-1) = f(1) = 0
    x = np.linspace(-1.1, 1.1, 500)  # A bit beyond the interval for better visualization
    y = np.sqrt(1 - x**2)

    # Derivative of f(x), excluding points where x = ±1 (non-differentiable)
    def f_prime(x):
        return -x / np.sqrt(1 - x**2)

    # Find c where f'(c) = 0
    c_rolle = 0  # f'(x) = -x / sqrt(1 - x^2), so f'(0) = 0

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="f(x) = √(1 - x²)", color="blue")
    plt.scatter([a, b], [f(a), f(b)], color="red", label="Endpoints (a, f(a)) and (b, f(b))")
    plt.scatter([c_rolle], [f(c_rolle)], color="green", label=f"Point c (Rolle's): x = {c_rolle}")
    plt.axvline(c_rolle, color="green", linestyle=":", alpha=0.7)

    plt.title("Visualization of Rolle's Theorem with f(x) = √(1 - x²)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()

# Visualize Rolle's Theorem
visualize_rolles_sqrt()

from matplotlib.animation import FuncAnimation

# Define the function and its derivative
def f(x):
    return np.sqrt(1 - x**2)

def f_prime(x):
    return -x / np.sqrt(1 - x**2)

# Interval [a, b]
a, b = -1, 1

# Generate x values for the plot
x = np.linspace(-1, 1, 500)

# Prepare the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.set_title("Rolle's Theorem Animation: f(x) = √(1 - x²)", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.grid()

# Plot the function
line, = ax.plot(x, f(x), color="blue", label="f(x) = √(1 - x²)")
points, = ax.plot([], [], 'ro', label="Endpoints (a, f(a)) and (b, f(b))")
tangent_line, = ax.plot([], [], 'g--', label="Tangent Line at x = c")
c_point, = ax.plot([], [], 'go', label="Point c: f'(c) = 0")

# Add legend
ax.legend()

# Initialization function
def init():
    points.set_data([a, b], [f(a), f(b)])  # Mark endpoints
    tangent_line.set_data([], [])
    c_point.set_data([], [])
    return line, points, tangent_line, c_point

# Animation update function
def update(frame):
    c = frame  # Current point c for the tangent line
    tangent_slope = f_prime(c)
    tangent_intercept = f(c) - tangent_slope * c
    tangent_x = np.linspace(c - 0.5, c + 0.5, 100)
    tangent_y = tangent_slope * tangent_x + tangent_intercept

    # Update the tangent line and point
    tangent_line.set_data(tangent_x, tangent_y)
    c_point.set_data([c], [f(c)])
    return line, points, tangent_line, c_point

# Frames for animation (values of c)
frames = np.linspace(-1, 1, 100)

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)

# Show the animation
plt.show()
