import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function
def f(x, y):
    return x**2 * y + np.sin(x * y)

# Define the partial derivatives
def partial_fx(x, y):
    return 2 * x * y + y * np.cos(x * y)

def partial_fy(x, y):
    return x**2 + x * np.cos(x * y)

# Create a meshgrid for visualization
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

# Calculate function values and partial derivatives
Z = f(X, Y)
Z_fx = partial_fx(X, Y)
Z_fy = partial_fy(X, Y)

# Plot the 3D surface of the function
fig = plt.figure(figsize=(14, 10))

# Plot the function f(x, y)
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('Function f(x, y)', fontsize=14)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')

# Plot the partial derivative with respect to x
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_fx, cmap='coolwarm', alpha=0.8)
ax2.set_title('Partial Derivative ∂f/∂x', fontsize=14)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('∂f/∂x')

# Plot the partial derivative with respect to y
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z_fy, cmap='plasma', alpha=0.8)
ax3.set_title('Partial Derivative ∂f/∂y', fontsize=14)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('∂f/∂y')

plt.tight_layout()
plt.show()


# Define the function
def f(x, y):
    return x**2 * y + np.sin(x * y)

# Define the partial derivatives
def partial_fx(x, y):
    return 2 * x * y + y * np.cos(x * y)

def partial_fy(x, y):
    return x**2 + x * np.cos(x * y)

# Create a meshgrid for 3D visualization
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

# Calculate function values and partial derivatives
Z = f(X, Y)
Z_fx = partial_fx(X, Y)
Z_fy = partial_fy(X, Y)

# 2D slices for visualization
x_fixed = 1  # Fix x at 1
y_fixed = 1  # Fix y at 1
y_slice = y
x_slice = x
f_fixed_x = f(x_fixed, y_slice)  # Slice along y-axis
f_fixed_y = f(x_slice, y_fixed)  # Slice along x-axis
fx_fixed_x = partial_fx(x_fixed, y_slice)
fy_fixed_y = partial_fy(x_slice, y_fixed)

# Plot the 3D surface of the function
fig = plt.figure(figsize=(16, 10))

# 3D plots
ax1 = fig.add_subplot(231, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('Function f(x, y)', fontsize=14)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')

ax2 = fig.add_subplot(232, projection='3d')
ax2.plot_surface(X, Y, Z_fx, cmap='coolwarm', alpha=0.8)
ax2.set_title('Partial Derivative ∂f/∂x', fontsize=14)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('∂f/∂x')

ax3 = fig.add_subplot(233, projection='3d')
ax3.plot_surface(X, Y, Z_fy, cmap='plasma', alpha=0.8)
ax3.set_title('Partial Derivative ∂f/∂y', fontsize=14)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('∂f/∂y')

# 2D line plots
ax4 = fig.add_subplot(234)
ax4.plot(y_slice, f_fixed_x, label='f(x=1, y)', color='blue')
ax4.plot(y_slice, fx_fixed_x, label='∂f/∂x (x=1, y)', color='red')
ax4.set_title('2D Slice at x=1', fontsize=14)
ax4.set_xlabel('y')
ax4.set_ylabel('Function / Derivative')
ax4.legend()

ax5 = fig.add_subplot(235)
ax5.plot(x_slice, f_fixed_y, label='f(x, y=1)', color='blue')
ax5.plot(x_slice, fy_fixed_y, label='∂f/∂y (x, y=1)', color='green')
ax5.set_title('2D Slice at y=1', fontsize=14)
ax5.set_xlabel('x')
ax5.set_ylabel('Function / Derivative')
ax5.legend()

plt.tight_layout()
plt.show()

# Define the function
def f(x, y, z):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (x**2 + y**2 + z**2) / (x + y + z)
        result[np.isnan(result)] = 0  # Define the function as 0 at (0, 0, 0)
        return result

# Create a grid for visualization
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
z = np.linspace(-1, 1, 100)

X, Y, Z = np.meshgrid(x, y, z)

# Evaluate the function
F = f(X, Y, Z)

# 3D Visualization of slices (z=0, y=0, x=0)
fig = plt.figure(figsize=(18, 12))

# Slice where z = 0
ax1 = fig.add_subplot(231, projection='3d')
Z_fixed = 0
F_z0 = f(X, Y, Z_fixed)
ax1.plot_surface(X[:, :, 0], Y[:, :, 0], F_z0[:, :, 0], cmap='viridis')
ax1.set_title('f(x, y, z=0)', fontsize=14)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y, z)')

# Slice where y = 0
ax2 = fig.add_subplot(232, projection='3d')
Y_fixed = 0
F_y0 = f(X[:, 0, :], Y_fixed, Z[:, 0, :])
ax2.plot_surface(X[:, 0, :], Z[:, 0, :], F_y0, cmap='plasma')
ax2.set_title('f(x, y=0, z)', fontsize=14)
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax2.set_zlabel('f(x, y, z)')

# Slice where x = 0
ax3 = fig.add_subplot(233, projection='3d')
X_fixed = 0
F_x0 = f(X_fixed, Y[0, :, :], Z[0, :, :])
ax3.plot_surface(Y[0, :, :], Z[0, :, :], F_x0, cmap='coolwarm')
ax3.set_title('f(x=0, y, z)', fontsize=14)
ax3.set_xlabel('y')
ax3.set_ylabel('z')
ax3.set_zlabel('f(x, y, z)')

# 2D visualization along paths approaching the origin
path = np.linspace(-1, 1, 100)
fx_path = f(path, path, path)  # Along x=y=z
fy_path = f(path, 0, 0)        # Along y=z=0
fz_path = f(0, 0, path)        # Along x=y=0

ax4 = fig.add_subplot(234)
ax4.plot(path, fx_path, label='x=y=z', color='blue')
ax4.plot(path, fy_path, label='y=z=0', color='red')
ax4.plot(path, fz_path, label='x=y=0', color='green')
ax4.set_title('Limit Paths to (0, 0, 0)', fontsize=14)
ax4.set_xlabel('Path')
ax4.set_ylabel('f(x, y, z)')
ax4.legend()

# Continuity check near origin (scatter points)
points_x = np.random.uniform(-0.1, 0.1, 500)
points_y = np.random.uniform(-0.1, 0.1, 500)
points_z = np.random.uniform(-0.1, 0.1, 500)
values = f(points_x, points_y, points_z)

ax5 = fig.add_subplot(235, projection='3d')
scatter = ax5.scatter(points_x, points_y, points_z, c=values, cmap='coolwarm')
ax5.set_title('Continuity Check Near Origin', fontsize=14)
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_zlabel('z')
fig.colorbar(scatter, ax=ax5, shrink=0.6)

plt.tight_layout()
plt.show()