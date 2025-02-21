import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Partial Derivative for a function of one variable: f(x) = x^2
x = np.linspace(-5, 5, 100)
f_x = x**2
df_x = 2*x  # Derivative

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(x, f_x, label="f(x) = x^2")
plt.plot(x, df_x, label="df/dx = 2x", linestyle="dashed")
plt.legend()
plt.title("1 Variable Partial Derivative")
plt.xlabel("x")
plt.ylabel("f(x)")

# 2. Partial Derivative for a function of two variables: f(x, y) = x^2 + y^2
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
dZ_dx = 2*X  # ∂f/∂x
dZ_dy = 2*Y  # ∂f/∂y

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)
ax.quiver(X, Y, Z, dZ_dx, dZ_dy, 0, color='r', length=1, normalize=True)
ax.set_title("2 Variable Partial Derivatives")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("f(X,Y)")

# 3. Partial Derivative for a function of three variables: f(x, y, z) = x^2 + y^2 + z^2
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
z = np.linspace(-5, 5, 10)
X, Y, Z = np.meshgrid(x, y, z)
dF_dx = 2*X  # ∂f/∂x
dF_dy = 2*Y  # ∂f/∂y
dF_dz = 2*Z  # ∂f/∂z

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, dF_dx, dF_dy, dF_dz, length=1, normalize=True, color="r")
ax.set_title("3 Variable Partial Derivatives")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate 3D Gaussian Data
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))  # 3D Normal Distribution

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot Gaussian Surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# Define the Tangent Plane Function
def tangent_plane(x, y, a, b):
    """Equation of a tangent plane at (a, b)"""
    z0 = np.exp(-(a**2 + b**2))  # Height of Gaussian at (a, b)
    dzdx = -2 * a * z0  # Partial derivative w.r.t x
    dzdy = -2 * b * z0  # Partial derivative w.r.t y
    return z0 + dzdx * (x - a) + dzdy * (y - b)

# Define Different Slices by Changing (a, b)
slices = [(0, 0), (1, 1), (-1, -1), (2, 0), (0, -2)]  # Different slicing points

for (a, b) in slices:
    Z_plane = tangent_plane(X, Y, a, b)
    ax.plot_surface(X, Y, Z_plane, color='red', alpha=0.5)

# Labels and View Angle
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis (Gaussian)")
ax.set_title("3D Gaussian Distribution with Tangent Plane Slices")
ax.view_init(elev=25, azim=35)  # Adjust viewing angle

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate 3D Gaussian Data
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))  # 3D Normal Distribution

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot Gaussian Surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# Define Tangent Plane in One Direction (∂f/∂x at a fixed y)
a, b = 0, 0  # Point of tangency
z0 = np.exp(-(a**2 + b**2))  # Function value at (a, b)

dzdx = -2 * a * z0  # Partial derivative w.r.t x
Z_tangent_x = z0 + dzdx * (X - a)  # Plane in x direction (holding y constant)

# Plot the Tangent Plane in X-Direction Only
ax.plot_surface(X, Y, Z_tangent_x, color='red', alpha=0.5)

# Labels and View Angle
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis (Gaussian)")
ax.set_title("3D Gaussian Distribution with ∂f/∂x Tangent Plane Slice")
ax.view_init(elev=25, azim=35)  # Adjust viewing angle

plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define function f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Partial derivatives
def df_dx(x, y):
    return 2*x

def df_dy(x, y):
    return 2*y

# Create meshgrid
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# Compute partial derivatives
dfx = df_dx(X, Y)
dfy = df_dy(X, Y)

# Plot contour of function
plt.figure(figsize=(7, 7))
contour = plt.contour(X, Y, f(X, Y), levels=20, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)

# Plot gradient vectors (partial derivatives)
plt.quiver(X, Y, dfx, dfy, color='red', angles='xy', scale_units='xy', scale=10)

# Labels
plt.title("2D Partial Derivatives of f(x,y) = x² + y²")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate 3D Gaussian Data
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))  # 3D Normal Distribution

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot Gaussian Surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# Define Tangent Plane in One Direction (∂f/∂y at a fixed x)
a, b = 0, 0  # Point of tangency
z0 = np.exp(-(a**2 + b**2))  # Function value at (a, b)

dzdy = -2 * b * z0  # Partial derivative w.r.t y
Z_tangent_y = z0 + dzdy * (Y - b)  # Plane in y direction (holding x constant)

# Plot the Tangent Plane in Y-Direction Only
ax.plot_surface(X, Y, Z_tangent_y, color='red', alpha=0.5)

# Labels and View Angle
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis (Gaussian)")
ax.set_title("3D Gaussian Distribution with ∂f/∂y Tangent Plane Slice")
ax.view_init(elev=25, azim=35)  # Adjust viewing angle

plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define functions
def f1(x, y):  # Linear function
    return 2*x + 3*y

def df1_dx(x, y):
    return np.full_like(x, 2)  # Partial derivative is constant 2

def df1_dy(x, y):
    return np.full_like(y, 3)  # Partial derivative is constant 3

def f2(x, y):  # Exponential function
    return np.exp(x) * np.sin(y)

def df2_dx(x, y):
    return np.exp(x) * np.sin(y)

def df2_dy(x, y):
    return np.exp(x) * np.cos(y)

def f3(x, y):  # Saddle function
    return x**2 - y**2

def df3_dx(x, y):
    return 2*x

def df3_dy(x, y):
    return -2*y

# Generate grid
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# List of functions
functions = [
    (f1, df1_dx, df1_dy, "Linear: f(x,y) = 2x + 3y"),
    (f2, df2_dx, df2_dy, "Exponential: f(x,y) = e^x sin(y)"),
    (f3, df3_dx, df3_dy, "Saddle: f(x,y) = x² - y²")
]

# Plot each function
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (func, dfdx, dfdy, title) in zip(axes, functions):
    Z = func(X, Y)
    dX = dfdx(X, Y)
    dY = dfdy(X, Y)

    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.quiver(X, Y, dX, dY, color='red', angles='xy', scale_units='xy', scale=10)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Function: f(x, y) = x^2 - y^2
def f(x, y):
    return x**2 - y**2

# Partial derivatives
def df_dx(x, y):
    return 2*x

def df_dy(x, y):
    return -2*y

# Create grid
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# Compute partial derivatives
dfx = df_dx(X, Y)
dfy = df_dy(X, Y)

# Plot contour and gradient arrows
plt.figure(figsize=(7, 7))
contour = plt.contour(X, Y, f(X, Y), levels=20, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.quiver(X, Y, dfx, dfy, color='red')  # Arrows for gradient

# Labels
plt.title("Partial Derivatives of f(x,y) = x² - y²")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define function f(x, y) = x² - y²
def f(x, y):
    return x**2 - y**2

# Compute derivatives
def df_dx(x, y):
    return 2*x

def df_dy(x, y):
    return -2*y

# Create grid
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Compute function values and derivatives
Z = f(X, Y)
dfx = df_dx(X, Y)
dfy = df_dy(X, Y)

# Plot contour and gradient arrows
plt.figure(figsize=(7, 7))
contour = plt.contour(X, Y, Z, levels=20, cmap='coolwarm')
plt.clabel(contour, inline=True, fontsize=8)
plt.quiver(X, Y, dfx, dfy, color='black')  # Gradient vectors
plt.scatter(0, 0, color='red', s=100, label="Saddle Point (0,0)")

# Labels
plt.title("Saddle Point & Partial Derivatives of f(x,y) = x² - y²")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define function f(x, y) = x² - y²
def f(x, y):
    return x**2 - y**2

# Compute partial derivatives
def df_dx(x, y):
    return 2*x

def df_dy(x, y):
    return -2*y

# Create grid
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Compute gradients
dfx = df_dx(X, Y)
dfy = df_dy(X, Y)

# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7)

# Plot gradient arrows
ax.quiver(X, Y, Z, dfx, dfy, 0, color='black', length=0.3, normalize=True)

# Highlight saddle point
ax.scatter(0, 0, f(0, 0), color='red', s=100, label="Saddle Point (0,0)")

# Labels
ax.set_title("Saddle Point & Partial Derivatives (3D)")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("f(x, y) = x² - y²")
ax.legend()

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function: f(x, y) = x² - y²
def f(x, y):
    return x**2 - y**2

# Partial derivatives
def df_dx(x, y):
    return 2*x

def df_dy(x, y):
    return -2*y

# Create grid
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Compute partial derivatives
dfx = df_dx(X, Y)
dfy = df_dy(X, Y)

# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Plot gradient arrows
ax.quiver(X, Y, Z, dfx, dfy, 0, color='red', length=0.7, normalize=True)

# Labels
ax.set_title("3D Partial Derivatives of f(x,y) = x² - y²")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("f(x, y)")
ax.view_init(elev=30, azim=120)  # Adjust view angle

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define functions
def f1(x, y):  # Linear function
    return 2*x + 3*y

def df1_dx(x, y):
    return np.full_like(x, 2)

def df1_dy(x, y):
    return np.full_like(y, 3)

def f2(x, y):  # Exponential function
    return np.exp(x) * np.sin(y)

def df2_dx(x, y):
    return np.exp(x) * np.sin(y)

def df2_dy(x, y):
    return np.exp(x) * np.cos(y)

def f3(x, y):  # Saddle function
    return x**2 - y**2

def df3_dx(x, y):
    return 2*x

def df3_dy(x, y):
    return -2*y

# Generate grid
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# List of functions
functions = [
    (f1, df1_dx, df1_dy, "Linear: f(x,y) = 2x + 3y"),
    (f2, df2_dx, df2_dy, "Exponential: f(x,y) = e^x sin(y)"),
    (f3, df3_dx, df3_dy, "Saddle: f(x,y) = x² - y²")
]

# Plot each function in 3D
fig = plt.figure(figsize=(18, 6))

for i, (func, dfdx, dfdy, title) in enumerate(functions, 1):
    ax = fig.add_subplot(1, 3, i, projection='3d')
    Z = func(X, Y)
    dX = dfdx(X, Y)
    dY = dfdy(X, Y)

    # Plot surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    # Plot gradient arrows
    ax.quiver(X, Y, Z, dX, dY, 0, color='red', length=0.3, normalize=True)

    # Labels
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("f(x, y)")
    ax.view_init(elev=30, azim=120)  # Adjust view angle

plt.tight_layout()
plt.show()
