import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 5)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(True)

# Continuous function: f(x) = x^2
x_cont = np.linspace(-2, 2, 500)
y_cont = x_cont**2

# Discontinuous function: g(x)
x_disc = np.linspace(-2, 2, 500)
y_disc = np.piecewise(x_disc, [x_disc < 0, x_disc >= 0], [1, 2])

# Plot initialization
line_cont, = ax.plot([], [], color='blue', label='Continuous: f(x) = x^2')
line_disc, = ax.plot([], [], color='red', linestyle='--', label='Discontinuous: g(x)')
ax.legend()

# Update function for animation
def update(frame):
    line_cont.set_data(x_cont[:frame], y_cont[:frame])
    line_disc.set_data(x_disc[:frame], y_disc[:frame])
    return line_cont, line_disc

# Animation
ani = FuncAnimation(fig, update, frames=len(x_cont), interval=20, blit=True)

# Save or display animation
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define functions
x = np.linspace(-2, 2, 500)
X, Y = np.meshgrid(x, x)

# Continuous: f(x, y) = sin(sqrt(x^2 + y^2))
Z_cont = np.sin(np.sqrt(X**2 + Y**2))

# Piecewise Continuous: g(x, y) = sqrt(|x| + |y|)
Z_piecewise = np.sqrt(np.abs(X) + np.abs(Y))

# Jump Discontinuity: h(x, y)
Z_jump = np.piecewise(X, [X + Y < 0, X + Y >= 0], [1, 2])

# Infinite Discontinuity: k(x, y) = 1/(x^2 + y^2)
Z_infinite = 1 / (X**2 + Y**2 + 0.01)  # Added 0.01 to avoid division by zero

# Removable Discontinuity: m(x, y) = (x^2 + y^2 - 1) / (sqrt(x^2 + y^2) - 1)
R = np.sqrt(X**2 + Y**2)
Z_removable = np.where(R != 1, (R**2 - 1) / (R - 1), np.nan)  # Handle removable point

# Create figure and subplots
fig = plt.figure(figsize=(15, 10))

# Continuous Function
ax1 = fig.add_subplot(231, projection='3d')
ax1.plot_surface(X, Y, Z_cont, cmap='viridis')
ax1.set_title('Continuous: f(x, y) = sin(sqrt(x^2 + y^2))')

# Piecewise Continuous Function
ax2 = fig.add_subplot(232, projection='3d')
ax2.plot_surface(X, Y, Z_piecewise, cmap='plasma')
ax2.set_title('Piecewise: g(x, y) = sqrt(|x| + |y|)')

# Jump Discontinuity
ax3 = fig.add_subplot(233, projection='3d')
ax3.plot_surface(X, Y, Z_jump, cmap='coolwarm')
ax3.set_title('Jump: h(x, y)')

# Infinite Discontinuity
ax4 = fig.add_subplot(234, projection='3d')
ax4.plot_surface(X, Y, Z_infinite, cmap='inferno')
ax4.set_title('Infinite: k(x, y) = 1/(x^2 + y^2)')

# Removable Discontinuity
ax5 = fig.add_subplot(235, projection='3d')
ax5.plot_surface(X, Y, Z_removable, cmap='cividis')
ax5.set_title('Removable: m(x, y)')

# Adjust layout and show plot
plt.tight_layout()
plt.show()
