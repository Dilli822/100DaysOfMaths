import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function to plot
def func(x):
    return 1 / (x + 1) + 2

# Set up the figure and axis
fig, ax = plt.subplots()
x = np.linspace(-5, 5, 400)
y = func(x)

# Remove the vertical asymptote for plotting purposes
x = x[x != -1]  # Exclude x = -1 to avoid division by zero

# Initialize the plot
line, = ax.plot([], [], lw=2, label=r"$f(x) = \frac{1}{x+1} + 2$")
ax.axvline(-1, color='red', linestyle='--', label="Vertical Asymptote (x=-1)")
ax.axhline(2, color='blue', linestyle='--', label="Horizontal Asymptote (y=2)")

# Plot styling
ax.set_xlim(-5, 5)
ax.set_ylim(-10, 10)
ax.set_title("Animated Function Plot with Asymptotes")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()

# Function to initialize the animation
def init():
    line.set_data([], [])
    return line,

# Function to update the animation at each frame
def update(frame):
    x_frame = x[:frame]
    y_frame = func(x_frame)
    line.set_data(x_frame, y_frame)
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval=30)

# Save or display the animation
plt.show()


# Functions
def horizontal_asymptote_func(x):
    return 3 / x + 2  # Horizontal asymptote at y = 2

def vertical_asymptote_func(x):
    return 1 / (x - 2)  # Vertical asymptote at x = 2

# x-values for plotting
x1 = np.linspace(-10, 10, 400)
x2 = np.linspace(1, 3, 400)  # Around x = 2 for the vertical asymptote

# Remove problematic points
x1 = x1[x1 != 0]  # Avoid division by zero for horizontal
x2 = x2[x2 != 2]  # Avoid division by zero for vertical

# Plot setup
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Horizontal asymptote plot
ax[0].plot(x1, horizontal_asymptote_func(x1), label=r"$f(x) = \frac{3}{x} + 2$")
ax[0].axhline(2, color='red', linestyle='--', label="Horizontal Asymptote $y=2$")
ax[0].set_title("Horizontal Asymptote Example")
ax[0].set_xlabel("x")
ax[0].set_ylabel("f(x)")
ax[0].legend()
ax[0].grid()

# Vertical asymptote plot
ax[1].plot(x2, vertical_asymptote_func(x2), label=r"$f(x) = \frac{1}{x-2}$")
ax[1].axvline(2, color='red', linestyle='--', label="Vertical Asymptote $x=2$")
ax[1].set_title("Vertical Asymptote Example")
ax[1].set_xlabel("x")
ax[1].set_ylabel("f(x)")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()


# Define functions
def horizontal_asymptote_func(x):
    return 3 / x + 2  # Horizontal asymptote at y = 2

def vertical_asymptote_func(x):
    return 1 / (x - 2)  # Vertical asymptote at x = 2

# x values for plotting
x_horizontal = np.linspace(-10, 10, 400)  # For horizontal asymptote
x_vertical = np.linspace(0, 4, 400)       # Around x = 2 for vertical asymptote

# Exclude undefined points
x_horizontal = x_horizontal[x_horizontal != 0]  # Exclude x = 0 (division by zero)
x_vertical = x_vertical[x_vertical != 2]        # Exclude x = 2 (division by zero)

# Create the plot
plt.figure(figsize=(10, 6))

# Horizontal asymptote
plt.plot(x_horizontal, horizontal_asymptote_func(x_horizontal), label=r"$f(x) = \frac{3}{x} + 2$", color='blue')
plt.axhline(2, color='blue', linestyle='--', label="Horizontal Asymptote $y=2$")

# Vertical asymptote
plt.plot(x_vertical, vertical_asymptote_func(x_vertical), label=r"$g(x) = \frac{1}{x-2}$", color='orange')
plt.axvline(2, color='orange', linestyle='--', label="Vertical Asymptote $x=2$")

# Add labels, legend, and grid
plt.title("Horizontal and Vertical Asymptotes")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

# Show the plot
plt.show()