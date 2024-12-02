import numpy as np
import matplotlib.pyplot as plt

# Plot settings
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Example 1: y = x^2 (a function)
x1 = np.linspace(-2, 2, 500)
y1 = x1**2
ax[0].plot(x1, y1, label="$y = x^2$")
ax[0].axvline(x=1, color='red', linestyle='--', label="Vertical Line (x=1)")
ax[0].set_title("Passes Vertical Line Test")
ax[0].legend()

# Example 2: x^2 + y^2 = 1 (not a function)
theta = np.linspace(0, 2 * np.pi, 500)
x2 = np.cos(theta)
y2 = np.sin(theta)
ax[1].plot(x2, y2, label="$x^2 + y^2 = 1$")
ax[1].axvline(x=0.5, color='red', linestyle='--', label="Vertical Line (x=0.5)")
ax[1].set_title("Fails Vertical Line Test")
ax[1].legend()

plt.tight_layout()
plt.show()

# Define the function f(x) = sqrt(x)
x = np.linspace(0, 10, 500)
y = np.sqrt(x)

# Plot the function
plt.plot(x, y, label="$f(x) = \sqrt{x}$")
plt.title("Domain and Range of $f(x) = \sqrt{x}$")
plt.xlabel("x (Domain)")
plt.ylabel("f(x) (Range)")
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
plt.grid(True)
plt.legend()
plt.show()

# Define functions
def quadratic(x):
    return x**2

def reciprocal(x):
    return 1 / x

def square_root(x):
    return np.sqrt(x)

# Define ranges for each function
x_quadratic = np.linspace(-10, 10, 500)
y_quadratic = quadratic(x_quadratic)

x_reciprocal = np.linspace(-10, -0.1, 500)
x_reciprocal = np.append(x_reciprocal, np.linspace(0.1, 10, 500))
y_reciprocal = reciprocal(x_reciprocal)

x_sqrt = np.linspace(0, 10, 500)
y_sqrt = square_root(x_sqrt)

# Plot
plt.figure(figsize=(12, 6))

# Quadratic plot
plt.subplot(1, 3, 1)
plt.plot(x_quadratic, y_quadratic, label="$f(x) = x^2$")
plt.title("Quadratic: Domain $(-\infty, \infty)$, Range $[0, \infty)$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
plt.grid(True)
plt.legend()

# Reciprocal plot
plt.subplot(1, 3, 2)
plt.plot(x_reciprocal, y_reciprocal, label="$f(x) = \\frac{1}{x}$", color="orange")
plt.title("Reciprocal: Domain $(-\infty, 0) \\cup (0, \infty)$, Range $(-\infty, 0) \\cup (0, \infty)$")
plt.xlabel("x")
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
plt.grid(True)
plt.legend()

# Square root plot
plt.subplot(1, 3, 3)
plt.plot(x_sqrt, y_sqrt, label="$f(x) = \\sqrt{x}$", color="green")
plt.title("Square Root: Domain $[0, \infty)$, Range $[0, \infty)$")
plt.xlabel("x")
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
