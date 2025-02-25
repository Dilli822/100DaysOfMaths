import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivatives
def f(x):
    return x**3 - 3*x**2 + 2

def f_prime(x):
    return 3*x**2 - 6*x

def f_double_prime(x):
    return 6*x - 6

# Define x range
x = np.linspace(-1, 3, 400)
y = f(x)
y_prime = f_prime(x)
y_double_prime = f_double_prime(x)

# Find inflection points (where second derivative is zero)
inflection_x = np.roots([6, -6])  # Solving 6x - 6 = 0
inflection_y = f(inflection_x)

# Plot the function and its derivatives
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label='$f(x) = x^3 - 3x^2 + 2$', linewidth=2)
ax.plot(x, y_prime, label="$f'(x)$", linestyle='dashed')
ax.plot(x, y_double_prime, label="$f''(x)$", linestyle='dotted')

# Highlight concavity
ax.fill_between(x, y, where=(y_double_prime > 0), color='lightblue', alpha=0.3, label='Concave Up')
ax.fill_between(x, y, where=(y_double_prime < 0), color='lightcoral', alpha=0.3, label='Concave Down')

# Mark inflection points
ax.scatter(inflection_x, inflection_y, color='black', zorder=3, label='Inflection Points')

# Labels and legend
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.legend()
ax.set_title('Function, Derivatives, and Concavity')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.grid()
plt.show()

# Newton's Method Implementation
def newton_method(f, f_prime, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fpx) < tol:  # Avoid division by zero
            break
        x_new = x - fx / fpx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# Example usage of Newton's method
root = newton_method(f, f_prime, x0=2.5)
print(f"Root found at x = {root}")

# Notations for Second Derivatives and Hessian Matrix
from sympy import symbols, diff, Matrix

x, y = symbols('x y')
f_sym = x**2 + y**2 + x*y
f_xx = diff(f_sym, x, x)
f_yy = diff(f_sym, y, y)
f_xy = diff(f_sym, x, y)

hessian_matrix = Matrix([[f_xx, f_xy], [f_xy, f_yy]])
print("Hessian Matrix:")
print(hessian_matrix)

import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return x**3 - 3*x**2 + 2

def f_prime(x):
    return 3*x**2 - 6*x

def newton_method(f, f_prime, x0, iterations=5):
    x = x0
    xs = [x0]
    for _ in range(iterations):
        f_prime_val = f_prime(x)
        if f_prime_val == 0:  # Avoid division by zero
            print("Derivative is zero, stopping Newton's method.")
            break
        x = x - f(x) / f_prime_val
        xs.append(x)
    return xs

# Initial guess
x0 = 2
xs = newton_method(f, f_prime, x0)

x = np.linspace(-2, 4, 400)
y = f(x)

plt.plot(x, y, label='f(x)', color='b')
plt.scatter(xs, [f(xi) for xi in xs], color='r', label='Newton\'s Steps')
plt.title('Newton\'s Method Root-Finding')
plt.legend()
plt.grid(True)
plt.show()


def f_second_derivative(x):
    return 6*x - 6

x = np.linspace(-2, 4, 400)
y = f_second_derivative(x)

plt.plot(x, y, label="f''(x)", color='r')
plt.axhline(0, color='k', linestyle='--')
plt.fill_between(x, y, where=(y > 0), color='green', alpha=0.3, label='Concave Up')
plt.fill_between(x, y, where=(y < 0), color='red', alpha=0.3, label='Concave Down')
plt.legend()
plt.title("Concavity of f(x)")
plt.grid(True)
plt.show()


from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return x**3 - 3*x**2 + 2*x*y

x = np.linspace(-2, 4, 100)
y = np.linspace(-2, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title("Surface plot of f(x, y)")
plt.show()


import numpy as np
from numpy.linalg import eig

H = np.array([[2, 1], [1, 2]])  # Example Hessian matrix

eigenvalues, _ = eig(H)

if np.all(eigenvalues > 0):
    print("Local Minimum")
elif np.all(eigenvalues < 0):
    print("Local Maximum")
else:
    print("Saddle Point")


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Define the functions and their Jacobian matrix
def f(x, y):
    return x**2 + y**2 - 4  # Example function 1: Circle equation

def g(x, y):
    return x**2 - y - 1  # Example function 2: Parabola equation

def jacobian(x, y):
    # Jacobian matrix [df/dx, df/dy; dg/dx, dg/dy]
    df_dx = 2*x
    df_dy = 2*y
    dg_dx = 2*x
    dg_dy = -1
    return np.array([[df_dx, df_dy], [dg_dx, dg_dy]])

def newton_method_2d(f, g, jacobian, x0, y0, iterations=5):
    # Starting point (x0, y0)
    point = np.array([x0, y0])
    points = [point]
    
    for _ in range(iterations):
        # Evaluate the functions and Jacobian at the current point
        F = np.array([f(point[0], point[1]), g(point[0], point[1])])
        J = jacobian(point[0], point[1])
        
        # Solve for the update step using the inverse of the Jacobian
        delta = inv(J).dot(F)
        
        # Update the point
        point = point - delta
        points.append(point)
        
    return np.array(points)

# Initial guess
x0, y0 = 2, 1

# Perform Newton's method
points = newton_method_2d(f, g, jacobian, x0, y0, iterations=10)

# Plotting the results
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z1 = f(X, Y)  # Function 1: Circle
Z2 = g(X, Y)  # Function 2: Parabola

plt.figure(figsize=(8, 8))
plt.contour(X, Y, Z1, levels=[0], colors='blue', label='f(x, y) = 0')
plt.contour(X, Y, Z2, levels=[0], colors='red', label='g(x, y) = 0')

# Plot the Newton's method iterations
plt.plot(points[:, 0], points[:, 1], marker='o', color='black', label="Newton's Method Steps")
plt.title("Newton's Method for Two Variables")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sympy import symbols, diff, Matrix, lambdify

# Define the function and its derivatives
def f(x):
    return x**3 - 3*x**2 + 2

def f_prime(x):
    return 3*x**2 - 6*x

def f_double_prime(x):
    return 6*x - 6

# Define x range
x = np.linspace(-1, 3, 400)
y = f(x)
y_prime = f_prime(x)
y_double_prime = f_double_prime(x)

# Find inflection points (where second derivative is zero)
inflection_x = np.roots([6, -6])  # Solving 6x - 6 = 0
inflection_y = f(inflection_x)

# Plot the function and its derivatives
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label='$f(x) = x^3 - 3x^2 + 2$', linewidth=2)
ax.plot(x, y_prime, label="$f'(x)$", linestyle='dashed')
ax.plot(x, y_double_prime, label="$f''(x)$", linestyle='dotted')

# Highlight concavity
ax.fill_between(x, y, where=(y_double_prime > 0), color='lightblue', alpha=0.3, label='Concave Up')
ax.fill_between(x, y, where=(y_double_prime < 0), color='lightcoral', alpha=0.3, label='Concave Down')

# Mark inflection points
ax.scatter(inflection_x, inflection_y, color='black', zorder=3, label='Inflection Points')

# Labels and legend
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.legend()
ax.set_title('Function, Derivatives, and Concavity')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.grid()
plt.show()

# Newton's Method for Two Variables
def newton_method_2d(f_grad, f_hessian, x0, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad = np.array(f_grad(*x), dtype=float)
        hess = np.array(f_hessian(*x), dtype=float)
        if np.linalg.det(hess) == 0:
            break  # Avoid singular Hessian
        delta_x = np.linalg.solve(hess, -grad)
        x_new = x + delta_x
        if np.linalg.norm(delta_x) < tol:
            return x_new
        x = x_new
    return x

# Define function f(x, y) = x^2 + y^2 + x*y
x, y = symbols('x y')
f_sym = x**2 + y**2 + x*y
f_grad = [diff(f_sym, var) for var in (x, y)]
f_hessian = Matrix([[diff(g, var) for var in (x, y)] for g in f_grad])

# Convert to numerical functions
f_grad_func = lambdify((x, y), f_grad, 'numpy')
f_hessian_func = lambdify((x, y), f_hessian, 'numpy')

# Find critical point
critical_point = newton_method_2d(f_grad_func, f_hessian_func, x0=[1, 1])
print(f"Critical Point found at: {critical_point}")

# 3D Visualization of Function
X, Y = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
Z = X**2 + Y**2 + X*Y

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.scatter(*critical_point, critical_point[0]**2 + critical_point[1]**2 + critical_point[0]*critical_point[1], color='red', s=100, label='Critical Point')
ax.set_title('Newton\'s Method for Two Variables')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy import symbols, diff, lambdify

# Define a new function and its derivatives
def f(x):
    return np.sin(x) + 0.5*x**2

def f_prime(x):
    return np.cos(x) + x

def f_double_prime(x):
    return -np.sin(x) + 1

# Define x range
x = np.linspace(-4, 4, 400)
y = f(x)
y_prime = f_prime(x)
y_double_prime = f_double_prime(x)

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label='$f(x) = \sin(x) + 0.5x^2$', linewidth=2)
ax.plot(x, y_prime, label="$f'(x)$", linestyle='dashed')
ax.plot(x, y_double_prime, label="$f''(x)$", linestyle='dotted')
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.legend()
ax.set_title('Function, First and Second Derivatives')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.grid()

# Animation function
# Animation function
def update(frame):
    ax.clear()
    ax.plot(x, y, label='$f(x) = \sin(x) + 0.5x^2$', linewidth=2)
    ax.plot(x, y_prime, label="$f'(x)$", linestyle='dashed')
    ax.plot(x, y_double_prime, label="$f''(x)$", linestyle='dotted')
    ax.scatter(x[frame], y[frame], color='red', s=100, label='Point on f(x)')
    ax.scatter(x[frame], y_prime[frame], color='blue', s=100, label='Point on f\'(x)')
    ax.scatter(x[frame], y_double_prime[frame], color='green', s=100, label="Point on f'(x)")
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.legend()
    ax.set_title(f'Frame {frame}: First vs Second Derivative Animation')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.grid()


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(x), interval=50)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create grid for x, y
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)

# Functions for the plots
def f_min(x, y):
    return x**2 + y**2

def f_max(x, y):
    return -x**2 - y**2

def f_saddle(x, y):
    return x**2 - y**2

# Plotting the 3 functions

fig = plt.figure(figsize=(18, 6))

# Plot 1: Function with Minima
ax1 = fig.add_subplot(131, projection='3d')
Z1 = f_min(X, Y)
ax1.plot_surface(X, Y, Z1, cmap='viridis', edgecolor='none')
ax1.set_title('Function with Minima: $f(x, y) = x^2 + y^2$')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(x, y)')

# Plot 2: Function with Maxima
ax2 = fig.add_subplot(132, projection='3d')
Z2 = f_max(X, Y)
ax2.plot_surface(X, Y, Z2, cmap='plasma', edgecolor='none')
ax2.set_title('Function with Maxima: $f(x, y) = -x^2 - y^2$')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('f(x, y)')

# Plot 3: Function with Saddle Point
ax3 = fig.add_subplot(133, projection='3d')
Z3 = f_saddle(X, Y)
ax3.plot_surface(X, Y, Z3, cmap='inferno', edgecolor='none')
ax3.set_title('Function with Saddle Point: $f(x, y) = x^2 - y^2$')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('f(x, y)')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the function and its second derivative
def f(x):
    return x**3 - 6*x**2 + 9*x

def f_second_derivative(x):
    return 6*x - 12

# Create an array of x values
x = np.linspace(-2, 5, 400)

# Calculate y values for the function and its second derivative
y = f(x)
y_second_derivative = f_second_derivative(x)

# Plot the function and its second derivative
plt.figure(figsize=(10, 6))

# Plot the original function
plt.plot(x, y, label='$f(x) = x^3 - 6x^2 + 9x$', color='blue', linestyle='-', linewidth=2)

# Plot the second derivative
plt.plot(x, y_second_derivative, label="$f''(x) = 6x - 12$", color='red', linestyle='--', linewidth=2)

# Add labels and title
plt.title("Function and its Second Derivative")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)

# Show a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function and its second derivative
def f(x, y):
    return x**3 + y**3 - 6*x**2 - 6*y**2 + 9*x + 9*y

def f_second_derivative_x(x, y):
    return 6*x - 12

def f_second_derivative_y(x, y):
    return 6*y - 12

# Create meshgrid for x and y values
x = np.linspace(-2, 5, 400)
y = np.linspace(-2, 5, 400)
X, Y = np.meshgrid(x, y)

# Calculate Z values for the function and second derivatives
Z = f(X, Y)
Z_x_second = f_second_derivative_x(X, Y)
Z_y_second = f_second_derivative_y(X, Y)

# Plot the function and its second derivatives
fig = plt.figure(figsize=(14, 6))

# 1st subplot: Function surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Function: $f(x, y) = x^3 + y^3 - 6x^2 - 6y^2 + 9x + 9y$')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('f(x, y)')

# 2nd subplot: Second derivative w.r.t x
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_x_second, cmap='plasma')
ax2.set_title("Second Derivative w.r.t X: $f''_x(x, y) = 6x - 12$")
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('f''(x, y)')

# Show the plot
plt.tight_layout()
plt.show()
