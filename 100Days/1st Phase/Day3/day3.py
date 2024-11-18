import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the cost function: f(x, y) = x^2 + y^2 (simple quadratic function)
def cost_function(x, y):
    return x**2 + y**2

# Define the gradient of the cost function
def gradient(x, y):
    return np.array([2 * x, 2 * y])

# Initialize parameters
x_start, y_start = 3.0, 4.0  # Starting point
learning_rate = 0.1
iterations = 50

# Track the path of gradient descent
path_x = [x_start]
path_y = [y_start]
path_z = [cost_function(x_start, y_start)]

# Gradient Descent Loop
x, y = x_start, y_start
for _ in range(iterations):
    grad = gradient(x, y)
    x -= learning_rate * grad[0]
    y -= learning_rate * grad[1]
    path_x.append(x)
    path_y.append(y)
    path_z.append(cost_function(x, y))

# Create a grid of points for surface plot
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = cost_function(X, Y)

# Plotting the 3D Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# Plot the gradient descent path
ax.plot(path_x, path_y, path_z, color='red', marker='o', markersize=4, label='Gradient Descent Path')

# Mark the starting point
ax.scatter(path_x[0], path_y[0], path_z[0], color='blue', s=50, label='Start Point')
# Mark the minimum point (0,0,0)
ax.scatter(0, 0, 0, color='green', s=50, label='Minimum')

# Labels and Title
ax.set_title('3D Visualization of Gradient Descent', fontsize=14)
ax.set_xlabel('Parameter x')
ax.set_ylabel('Parameter y')
ax.set_zlabel('Cost Function Value')
ax.legend()

# Display plot
plt.show()
