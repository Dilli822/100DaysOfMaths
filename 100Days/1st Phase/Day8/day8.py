import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function f(x) = x
def f(x):
    return x

# The point of interest (a) and its limit (L)
a = 3
L = 5

# Define x values for plotting
x_vals = np.linspace(0, 6, 1000)

# Initial epsilon and delta values
epsilon = 0.5  # Fixed epsilon
delta = 1.0    # Initial delta

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 6)
ax.set_ylim(4, 6)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("f(x)", fontsize=12)

# Draw horizontal and vertical dashed lines
ax.axhline(L, color='black', linestyle='--', linewidth=1, label='y = L = 5')  # Horizontal line at L
ax.axvline(a, color='black', linestyle='--', linewidth=1, label='x = a = 3')  # Vertical line at a

# Plot the function
line, = ax.plot(x_vals, f(x_vals), label="f(x) = x", color='blue')

# Highlight epsilon bounds
epsilon_band = ax.fill_between(x_vals, L - epsilon, L + epsilon, color='yellow', alpha=0.3, label='ε-band')

# Highlight delta bounds
delta_lines = ax.vlines([], L - epsilon, L + epsilon, colors='purple', linewidth=1.5, label='δ-bounds')

# Add labels for epsilon and delta
epsilon_text_upper = ax.text(5.2, L + epsilon - 0.1, r"$5 + \epsilon$", color="green", fontsize=10)
epsilon_text_lower = ax.text(5.2, L - epsilon + 0.1, r"$5 - \epsilon$", color="green", fontsize=10)
delta_text_left = ax.text(a - delta - 0.5, 4.2, r"$3 - \delta$", color="purple", fontsize=10)
delta_text_right = ax.text(a + delta - 0.5, 4.2, r"$3 + \delta$", color="purple", fontsize=10)

# Update function for the animation
def update(frame):
    global delta
    delta = frame / 50  # Shrink delta over time
    valid_x = (x_vals > a - delta) & (x_vals < a + delta)  # x within the delta range
    x_delta_vals = x_vals[valid_x]

    # Update the vertical delta lines
    delta_lines.set_segments([[[x, L - epsilon], [x, L + epsilon]] for x in x_delta_vals])

    # Update delta text positions
    delta_text_left.set_position((a - delta - 0.5, 4.2))
    delta_text_right.set_position((a + delta - 0.5, 4.2))

    return line, epsilon_band, delta_lines, delta_text_left, delta_text_right

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(1, 101), blit=False, repeat=True)

# Add a legend
ax.legend(loc="upper left")

# Display the animation
plt.show()
