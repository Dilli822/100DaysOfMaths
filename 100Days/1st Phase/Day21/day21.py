import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define position and velocity functions
def s(t): # 4t^2 + 2t
    return 4 * t**2 + 2 * t  # Position function

def v(t):
    # 8t + 2
    return 8 * t + 2  # Velocity function (instantaneous velocity)

# Generate time values
t = np.linspace(0, 5, 500)  # Range of time
position = s(t)  # Position values

# Initialize the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 5)
ax.set_ylim(0, max(position) + 10)
ax.set_title("Position and Instantaneous Velocity")
ax.set_xlabel("Time (t)")
ax.set_ylabel("Position (s(t)) / Velocity (v(t))")

# Plot the position function
ax.plot(t, position, label="Position (s(t))", color="blue")

# Tangent line and point
tangent_line, = ax.plot([], [], color="red", label="Tangent Line (v(t))")
point, = ax.plot([], [], 'ro', label="Point of Tangency")

# Animation function
def animate(frame):
    t_point = frame / 10  # Time point for animation (scaled)
    s_point = s(t_point)  # Position at the current time
    slope = v(t_point)    # Instantaneous velocity (slope of tangent)

    # Define tangent line around the point
    tangent_t = np.linspace(t_point - 0.5, t_point + 0.5, 100)
    tangent_s = slope * (tangent_t - t_point) + s_point

    # Update the tangent line and point
    tangent_line.set_data(tangent_t, tangent_s)
    point.set_data([t_point], [s_point])  # Pass sequences (lists)

    return tangent_line, point

# Create animation
frames = 50  # Number of time steps
ani = FuncAnimation(fig, animate, frames=frames, interval=100, blit=True)

# Add legend and show the plot
ax.legend()
plt.show()
