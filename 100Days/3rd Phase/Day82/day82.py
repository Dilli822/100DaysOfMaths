import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Define function f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Compute Partial Derivatives
def df_dx(x, y):
    return 2*x  # Partial derivative w.r.t x

def df_dy(x, y):
    return 2*y  # Partial derivative w.r.t y

# Define point of tangency
a, b = 1, 1  # Chosen point (1,1)
z_tangent = f(a, b)  # Function value at (a, b)

# Define tangent plane equation
def tangent_plane(x, y):
    return z_tangent + df_dx(a, b) * (x - a) + df_dy(a, b) * (y - b)

# Create meshgrid for surface plot
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Initialize plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot function surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Tangent plane (initially hidden)
X_tan, Y_tan = np.meshgrid(np.linspace(a-1, a+1, 10), np.linspace(b-1, b+1, 10))
Z_tan = tangent_plane(X_tan, Y_tan)
tan_plane = ax.plot_surface(X_tan, Y_tan, Z_tan, color='red', alpha=0.6)

# Animate slicing process
slices = []

for i in range(1, 20):
    x_slice = np.full(10, a)
    y_vals = np.linspace(b - 1, b + 1, 10)
    z_vals = f(x_slice, y_vals)
    slices.append(ax.plot(x_slice, y_vals, z_vals, 'r-', lw=2))

    y_slice = np.full(10, b)
    x_vals = np.linspace(a - 1, a + 1, 10)
    z_vals = f(x_vals, y_slice)
    slices.append(ax.plot(x_vals, y_slice, z_vals, 'b-', lw=2))

def update(frame):
    if frame < len(slices):
        slices[frame][0].set_visible(True)

ani = animation.FuncAnimation(fig, update, frames=len(slices), interval=200)

# Labels
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Tangent Plane Animation")

plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define function f(x, y)
def f(x, y):
    return x**2 + y**2  # Simple quadratic function

# Partial derivatives (gradients)
def grad_f(x, y):
    df_dx = 2*x
    df_dy = 2*y
    return np.array([df_dx, df_dy])

# Gradient Descent Parameters
learning_rate = 0.1
num_iterations = 50

# Start at an initial point
x, y = 3, 3  # Initial guess
history = [(x, y)]  # Store points for visualization

# Perform Gradient Descent
for i in range(num_iterations):
    grad = grad_f(x, y)  # Compute gradient
    x -= learning_rate * grad[0]  # Update x
    y -= learning_rate * grad[1]  # Update y
    history.append((x, y))  # Store new point

# Convert history to arrays for plotting
history = np.array(history)

# Plot results
plt.figure(figsize=(8, 6))
X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = f(X, Y)
plt.contour(X, Y, Z, levels=30, cmap="viridis")
plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=4, label="Gradient Descent Path")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Gradient Descent to Find Minimum")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# Define Log-Loss function
def log_loss(y, y_hat):
    return - (y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))

# Generate probability values
y_hat_values = np.linspace(0.01, 0.99, 100)

# Compute log-loss for both classes
log_loss_1 = log_loss(1, y_hat_values)  # True label = 1
log_loss_0 = log_loss(0, y_hat_values)  # True label = 0

# Initialize plot
sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(7, 5))
line1, = ax.plot([], [], 'r-', lw=2, label="y = 1")
line2, = ax.plot([], [], 'b-', lw=2, label="y = 0")
ax.set_xlim(0, 1)
ax.set_ylim(0, 5)
ax.set_xlabel("Predicted Probability (ŷ)")
ax.set_ylabel("Log Loss")
ax.set_title("Log-Loss Optimization")
ax.legend()

# Update function for animation
def update(frame):
    line1.set_data(y_hat_values[:frame], log_loss_1[:frame])
    line2.set_data(y_hat_values[:frame], log_loss_0[:frame])
    return line1, line2

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(y_hat_values), interval=50, blit=True)

# Show animation
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Initialize parameters
theta = np.random.randn(2,1)  # Random start
learning_rate = 0.1
iterations = 50
m = len(X)

# Add bias term (x0 = 1)
X_b = np.c_[np.ones((m, 1)), X]

# Gradient Descent
history = []
for _ in range(iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients
    history.append(theta.copy())  # Store updates for visualization

# Plot convergence
history = np.array(history).squeeze()
plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=4, label="Optimization Path")
plt.xlabel("Theta 0 (Bias)")
plt.ylabel("Theta 1 (Slope)")
plt.title("Gradient Descent for Linear Regression")
plt.legend()
plt.show()

print(f"Final Parameters: {theta.ravel()}")


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate synthetic dataset
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X - 2 * X**2 + np.random.randn(100, 1)  # Quadratic trend with noise

# Add higher-order terms for quadratic and cubic models
X_poly = np.c_[X, X**2, X**3]
X_b = np.c_[np.ones((len(X), 1)), X_poly]  # Bias term for all models

# Model degrees and initialization
degrees = [1, 2, 3]  # Linear, Quadratic, Cubic
theta = {deg: np.random.randn(deg + 1, 1) for deg in degrees}
learning_rate = 0.1
epochs = 100
history = {deg: [] for deg in degrees}  # Store updates for visualization

# Gradient Descent for all models
for deg in degrees:
    X_train = X_b[:, :deg+1]  # Select columns based on model degree
    for _ in range(epochs):
        gradients = (2 / len(X)) * X_train.T @ (X_train @ theta[deg] - y)
        theta[deg] -= learning_rate * gradients
        history[deg].append(theta[deg].copy())

# Visualization setup
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

X_range = np.linspace(0, 2, 100).reshape(-1, 1)
X_poly_range = np.c_[X_range, X_range**2, X_range**3]
X_b_range = np.c_[np.ones((len(X_range), 1)), X_poly_range]

def update(frame):
    for i, deg in enumerate(degrees):
        ax[i].cla()
        ax[i].scatter(X, y, color="blue", alpha=0.5, label="Data")
        theta_current = history[deg][frame]
        X_train = X_b_range[:, :deg+1]
        y_pred = X_train @ theta_current
        ax[i].plot(X_range, y_pred, color="red", linewidth=2, label=f"{deg}-Power Fit")
        ax[i].set_title(f"{deg}-Power Optimization Step {frame+1}")
        ax[i].legend()
        ax[i].set_xlabel("X")
        ax[i].set_ylabel("Y")

ani = animation.FuncAnimation(fig, update, frames=epochs, interval=100)
plt.show()
