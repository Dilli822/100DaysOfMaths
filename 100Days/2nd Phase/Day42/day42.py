import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, integrate, sin, cos, exp, lambdify

# Define the variable and function
x = symbols('x')
f = x**2
F = integrate(f, x)  # Indefinite integral

# Convert sympy expressions to numerical functions
f_lam = lambdify(x, f, 'numpy')
F_lam = lambdify(x, F, 'numpy')

# Generate data points for 2D visualization
x_vals = np.linspace(-10, 10, 400)
f_vals = f_lam(x_vals)
F_vals = F_lam(x_vals)

# Plot the function and its integral
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f_vals, label=r"$f(x) = x^2$", color="blue")
plt.plot(x_vals, F_vals, label=r"$F(x) = \frac{x^3}{3} + C$", color="orange", linestyle="--")
plt.axhline(0, color='black', linewidth=0.5, linestyle=":")
plt.axvline(0, color='black', linewidth=0.5, linestyle=":")
plt.title("Function and Its Anti-derivative")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# 3D Visualization
from mpl_toolkits.mplot3d import Axes3D

# Define a symbolic 3D function and convert to numerical
f_3d = sin(x) + x
f_3d_lam = lambdify(x, f_3d, 'numpy')

# Generate data points for 3D visualization
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = f_3d_lam(X)  # Calculate Z using the symbolic function

# Plot the 3D function
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

ax.set_title("3D Visualization of Function")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("f(x)")

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, integrate, lambdify, sin

# Define the variable and function
x = symbols('x')
f = x**2 + sin(x)  # Same function for derivative and anti-derivative
f_derivative = diff(f, x)  # Derivative
f_antiderivative = integrate(f, x)  # Anti-derivative

# Convert to numerical functions using lambdify
f_lam = lambdify(x, f, 'numpy')
f_derivative_lam = lambdify(x, f_derivative, 'numpy')
f_antiderivative_lam = lambdify(x, f_antiderivative, 'numpy')

# Generate x values and compute y values
x_vals = np.linspace(-10, 10, 400)
f_vals = f_lam(x_vals)
f_derivative_vals = f_derivative_lam(x_vals)
f_antiderivative_vals = f_antiderivative_lam(x_vals)

# 2D Plot: Function, Derivative, and Anti-Derivative
plt.figure(figsize=(12, 6))
plt.plot(x_vals, f_vals, label=r"$f(x) = x^2 + \sin(x)$", color="blue")
plt.plot(x_vals, f_derivative_vals, label=r"$f'(x)$ (Derivative)", color="green")
plt.plot(x_vals, f_antiderivative_vals, label=r"$F(x)$ (Anti-Derivative)", color="orange", linestyle="--")
plt.axhline(0, color='black', linewidth=0.5, linestyle=":")
plt.axvline(0, color='black', linewidth=0.5, linestyle=":")
plt.title("Function, Derivative, and Anti-Derivative (2D Visualization)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# 3D Plot: Function, Derivative, and Anti-Derivative
from mpl_toolkits.mplot3d import Axes3D

# Create meshgrid for 3D visualization
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)

# Compute Z values for the 3D plots
Z_func = f_lam(X)
Z_derivative = f_derivative_lam(X)
Z_antiderivative = f_antiderivative_lam(X)

# Plot the 3D Function
fig = plt.figure(figsize=(18, 6))

# 3D: Original Function
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z_func, cmap='viridis', alpha=0.8)
ax1.set_title("Original Function f(x)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("f(x)")

# 3D: Derivative
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_derivative, cmap='plasma', alpha=0.8)
ax2.set_title("Derivative f'(x)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("f'(x)")

# 3D: Anti-Derivative
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z_antiderivative, cmap='cividis', alpha=0.8)
ax3.set_title("Anti-Derivative F(x)")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("F(x)")

plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, integrate, lambdify

# Define the variable and function
x = symbols('x')
f = x**2  # Function f(x) = x^2
f_derivative = diff(f, x)  # Derivative
f_antiderivative = integrate(f, x)  # Anti-derivative

# Convert to numerical functions using lambdify
f_lam = lambdify(x, f, 'numpy')
f_derivative_lam = lambdify(x, f_derivative, 'numpy')
f_antiderivative_lam = lambdify(x, f_antiderivative, 'numpy')

# Generate x values and compute y values
x_vals = np.linspace(-10, 10, 400)
f_vals = f_lam(x_vals)
f_derivative_vals = f_derivative_lam(x_vals)
f_antiderivative_vals = f_antiderivative_lam(x_vals)

# 2D Plot: Function, Derivative, and Anti-Derivative
plt.figure(figsize=(12, 6))
plt.plot(x_vals, f_vals, label=r"$f(x) = x^2$", color="blue")
plt.plot(x_vals, f_derivative_vals, label=r"$f'(x) = 2x$ (Derivative)", color="green")
plt.plot(x_vals, f_antiderivative_vals, label=r"$F(x) = \frac{x^3}{3} + C$ (Anti-Derivative)", color="orange", linestyle="--")
plt.axhline(0, color='black', linewidth=0.5, linestyle=":")
plt.axvline(0, color='black', linewidth=0.5, linestyle=":")
plt.title("Function, Derivative, and Anti-Derivative (2D Visualization)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# 3D Plot: Function, Derivative, and Anti-Derivative
from mpl_toolkits.mplot3d import Axes3D

# Create meshgrid for 3D visualization
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)

# Compute Z values for the 3D plots
Z_func = f_lam(X)
Z_derivative = f_derivative_lam(X)
Z_antiderivative = f_antiderivative_lam(X)

# Plot the 3D Function
fig = plt.figure(figsize=(18, 6))

# 3D: Original Function
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z_func, cmap='viridis', alpha=0.8)
ax1.set_title("Original Function f(x) = x^2")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("f(x)")

# 3D: Derivative
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_derivative, cmap='plasma', alpha=0.8)
ax2.set_title("Derivative f'(x) = 2x")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("f'(x)")

# 3D: Anti-Derivative
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z_antiderivative, cmap='cividis', alpha=0.8)
ax3.set_title("Anti-Derivative F(x) = x^3/3 + C")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("F(x)")

plt.tight_layout()
plt.show()
