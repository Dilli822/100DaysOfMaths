import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Generate sequence data
n_values = np.arange(1, 100)  # Sequence indices
convergent_sequence = 1 / n_values  # Convergent sequence (approaching 0)
divergent_sequence = n_values  # Divergent sequence (grows unbounded)

# 2D Plot
def plot_2d():
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, convergent_sequence, label=r"$a_n = 1/n$ (Convergent: Approaches 0 as n → ∞)", marker="o", color="blue")
    plt.plot(n_values, divergent_sequence, label=r"$b_n = n$ (Divergent: Grows unbounded as n → ∞)", marker="x", color="red")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Reference Line: y = 0")
    plt.title("2D Visualization: Convergent and Divergent Sequences\nDetailed View of Sequence Behavior Over Index Range", fontsize=14)
    plt.xlabel("n (Index of the Sequence)", fontsize=12)
    plt.ylabel("Sequence Value", fontsize=12)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

# 3D Plot
def plot_3d():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(n_values, convergent_sequence, zs=0, zdir='z', label=r"$a_n = 1/n$ (Convergent)", color='blue')
    ax.scatter(n_values, divergent_sequence, zs=0, zdir='z', label=r"$b_n = n$ (Divergent)", color='red')
    ax.set_title("3D Visualization: Convergent and Divergent Sequences\nPerspective of Sequence Behavior in a Spatial Representation", fontsize=14)
    ax.set_xlabel("n (Index of the Sequence)", fontsize=12)
    ax.set_ylabel("Sequence Value", fontsize=12)
    ax.set_zlabel("z-axis (Fixed at 0 for Clarity)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    plt.show()

# Animation
def animate_sequences():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.set_title("Animated Visualization: Progression of Sequences Over Time\nDynamic Representation of Convergence and Divergence", fontsize=14)
    ax.set_xlabel("n (Index of the Sequence)", fontsize=12)
    ax.set_ylabel("Sequence Value", fontsize=12)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Reference Line: y = 0")
    line1, = ax.plot([], [], label=r"$a_n = 1/n$ (Convergent Sequence)", color="blue", marker="o")
    line2, = ax.plot([], [], label=r"$b_n = n$ (Divergent Sequence)", color="red", marker="x")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)

    def update(frame):
        line1.set_data(n_values[:frame], convergent_sequence[:frame])
        line2.set_data(n_values[:frame], divergent_sequence[:frame])
        return line1, line2

    ani = FuncAnimation(fig, update, frames=len(n_values), interval=50, blit=True)
    plt.show()

# Call the visualization functions
plot_2d()       # 2D Visualization
plot_3d()       # 3D Visualization
animate_sequences()  # Animated Visualization


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Infinite series examples and visualization

def plot_geometric_series():
    x = np.arange(1, 50)
    r = 0.5  # Common ratio
    partial_sums = [1 - r**n for n in x]  # Partial sums formula for geometric series

    plt.figure(figsize=(10, 6))
    plt.plot(x, partial_sums, label='Geometric Series Partial Sums (1 - r^n)', marker='o')
    plt.axhline(1 / (1 - r), color='r', linestyle='--', label='Sum to Infinity (1 / (1 - r)) (Convergent)')
    plt.xlabel('Number of Terms (n)')
    plt.ylabel('Value of Partial Sum')
    plt.title('Geometric Series Convergence in 2D: Partial Sums vs. Number of Terms')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_telescoping_series():
    x = np.arange(1, 50)
    partial_sums = [1 - 1 / (n + 1) for n in x]  # Telescoping series partial sums

    plt.figure(figsize=(10, 6))
    plt.plot(x, partial_sums, label='Telescoping Series Partial Sums (1 - 1/(n+1))', marker='o')
    plt.axhline(1, color='r', linestyle='--', label='Sum to Infinity (Convergent)')
    plt.xlabel('Number of Terms (n)')
    plt.ylabel('Value of Partial Sum')
    plt.title('Telescoping Series Convergence in 2D: Partial Sums vs. Number of Terms')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_3d_geometric_series():
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(1, 50, 50)
    r = 0.5  # Common ratio
    y = [1 - r**n for n in x]  # Partial sums
    z = x

    ax.plot(x, y, z, label='Geometric Series Partial Sums in 3D', color='b')
    ax.set_xlabel('Index (n) - Number of Terms')
    ax.set_ylabel('Value of Partial Sum')
    ax.set_zlabel('n-axis (Number of Terms)')
    ax.legend()
    ax.set_title('3D Visualization of Geometric Series Convergence')

    plt.show()

def animate_geometric_series():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, 50)
    r = 0.5  # Common ratio
    partial_sums = [1 - r**n for n in x]

    line, = ax.plot([], [], marker='o', label='Geometric Series Partial Sums Animation')
    ax.axhline(1 / (1 - r), color='r', linestyle='--', label='Sum to Infinity (1 / (1 - r))')

    def init():
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 2)
        ax.set_xlabel('Number of Terms (n)')
        ax.set_ylabel('Value of Partial Sum')
        ax.set_title('Animation of Geometric Series Convergence: Partial Sums Over Time')
        ax.legend(loc='upper right')
        return line,

    def update(frame):
        line.set_data(x[:frame], partial_sums[:frame])
        return line,

    ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True)
    plt.show()

# Execute all visualizations
plot_geometric_series()
plot_telescoping_series()
plot_3d_geometric_series()
animate_geometric_series()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math  # Import the math module for factorial

# Define the series functions
def maclaurin_series(x, terms=5):
    return sum((x**n) / math.factorial(n) for n in range(terms))

def taylor_series(x, a=1, terms=5):
    return sum(((x - a)**n) / math.factorial(n) for n in range(terms))

def telescoping_series(n):
    return 1 / n - 1 / (n + 1)

# Generate x values and series data
x = np.linspace(-2, 2, 200)
n_values = np.arange(1, 50)

maclaurin_values = maclaurin_series(x, terms=5)
taylor_values = taylor_series(x, a=0, terms=5)
telescoping_values = np.cumsum([telescoping_series(n) for n in n_values])

# 2D Plot
def plot_2d():
    plt.figure(figsize=(10, 6))
    plt.plot(x, maclaurin_values, label="Maclaurin Series (5 terms)", color="blue")
    plt.plot(x, taylor_values, label="Taylor Series around a=0 (5 terms)", color="green")
    plt.scatter(n_values, telescoping_values, label="Telescoping Series (Cumulative)", color="red")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Reference Line: y=0")
    plt.title("2D Visualization: Maclaurin, Taylor, and Telescoping Series\nDetailed View of Approximation and Convergence", fontsize=14)
    plt.xlabel("x (Input Value)", fontsize=12)
    plt.ylabel("Series Value", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

# 3D Plot
def plot_3d():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3D scatter for telescoping series
    ax.scatter(n_values, telescoping_values, zs=0, zdir='z', label="Telescoping Series (Cumulative)", color="red")

    # 3D lines for Maclaurin and Taylor
    ax.plot(x, maclaurin_values, zs=-2, zdir='z', label="Maclaurin Series (5 terms)", color="blue")
    ax.plot(x, taylor_values, zs=2, zdir='z', label="Taylor Series around a=0 (5 terms)", color="green")

    ax.set_title("3D Visualization: Maclaurin, Taylor, and Telescoping Series\nPerspective of Approximation and Convergence", fontsize=14)
    ax.set_xlabel("x (Input Value)", fontsize=12)
    ax.set_ylabel("Series Value", fontsize=12)
    ax.set_zlabel("z-axis (Fixed)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    plt.show()

# Animation for Telescoping Series
def animate_telescoping():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 1)
    ax.set_title("Animation: Telescoping Series Cumulative Sum\nDynamic Representation of Partial Sums", fontsize=14)
    ax.set_xlabel("n (Index of the Series)", fontsize=12)
    ax.set_ylabel("Cumulative Value", fontsize=12)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Reference Line: y=0")
    line, = ax.plot([], [], marker="o", label="Telescoping Series", color="red")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(n_values[:frame], telescoping_values[:frame])
        return line,

    ani = FuncAnimation(fig, update, frames=len(n_values), init_func=init, interval=100, blit=True)
    plt.show()

# Call the visualization functions
plot_2d()       # 2D Visualization
plot_3d()       # 3D Visualization
animate_telescoping()  # Animated Telescoping Series


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math  # Import the math module

# Define the function and its Maclaurin series approximation
def f(x):
    return np.sin(x)  # Example function: sine

def taylor_series(x, n_terms):
    """Compute the Taylor series of sin(x) centered at 0 (Maclaurin series)."""
    series_sum = 0
    for n in range(n_terms):
        term = ((-1)**n) * (x**(2*n+1)) / math.factorial(2*n+1)  # Use math.factorial instead of np.math.factorial
        series_sum += term
    return series_sum

# Set up the figure and axis
x = np.linspace(-2 * np.pi, 2 * np.pi, 400)
y = f(x)

fig, ax = plt.subplots()
ax.set_xlim(-2 * np.pi, 2 * np.pi)
ax.set_ylim(-1.5, 1.5)
line_true, = ax.plot(x, y, label="True Function (sin(x))", color='blue')
line_approx, = ax.plot(x, np.zeros_like(x), label="Taylor Approximation", color='red', linestyle='--')

# Add a title and legend
ax.set_title("Maclaurin Series Approximation of sin(x)")
ax.legend()

# Update function for the animation
def update(frame):
    n_terms = frame + 1  # Number of terms in the Taylor series
    y_approx = taylor_series(x, n_terms)
    line_approx.set_ydata(y_approx)  # Update the approximation curve
    return line_approx,

# Create the animation
ani = FuncAnimation(fig, update, frames=10, interval=500, blit=True)

# Show the animation
plt.show()
