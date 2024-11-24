import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function f(x)
def f(x):
    return x**3 - 5*x + 6

# Parameters for limit definition
a = 1  # Point at which limit is defined
L = f(a)  # Limit value at x = a
epsilon = 0.2  # Epsilon range

# Domain for the graph
x = np.linspace(-3, 3, 500)
y = f(x)

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Visualization of Limit Definition", fontsize=14)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("f(x)", fontsize=12)
ax.axhline(L, color="red", linestyle="--", label=f"L = {L}")
ax.axhline(L + epsilon, color="green", linestyle="--", label=f"L + ε = {L + epsilon}")
ax.axhline(L - epsilon, color="green", linestyle="--", label=f"L - ε = {L - epsilon}")
line, = ax.plot(x, y, label="f(x) = x³ - 5x + 6")
highlight, = ax.plot([], [], color="orange", label="f(x) within ε bounds", lw=2)

# Add initial δ shading (this will be updated dynamically)
delta_shade = ax.fill_between(x, -10, 10, where=(x < a + 0.1) & (x > a - 0.1),
                              color="blue", alpha=0.2, label="δ range")

ax.legend()

# Animation function
def update(frame):
    delta = frame / 100  # Gradually increase δ
    x_delta = np.linspace(a - delta, a + delta, 100)
    y_delta = f(x_delta)

    # Update the shaded δ region
    ax.collections.clear()  # Clear previous shading
    ax.fill_between(x, -10, 10, where=(x < a + delta) & (x > a - delta),
                    color="blue", alpha=0.2)

    # Highlight the region where f(x) is within ε
    within_epsilon = (y_delta > L - epsilon) & (y_delta < L + epsilon)
    highlight.set_data(x_delta[within_epsilon], y_delta[within_epsilon])

    return highlight,

# Animation setup
frames = np.arange(1, 100, 2)  # Range for δ changes
anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

# Display the animation
plt.show()




