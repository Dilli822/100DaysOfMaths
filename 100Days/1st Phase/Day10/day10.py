import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return x**2 - 4

# Generate x values
x = np.linspace(-3, 3, 500)  # Range from -3 to 3 with 500 points
y = f(x)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$f(x) = x^2 - 4$", color='blue')

# Highlight regions where f(x) >= 0
plt.fill_between(x, y, 0, where=(y >= 0), color='green', alpha=0.3, label=r"$f(x) \geq 0$")

# Highlight regions where f(x) < 0
plt.fill_between(x, y, 0, where=(y < 0), color='red', alpha=0.3, label=r"$f(x) < 0$")

# Add axes, labels, and a legend
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.title("Visualization of the Inequality $x^2 - 4 \geq 0$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()

# Show the solution
plt.show()


# Define the function f(x) = |x^3 - 5x + 6| - 2
def f(x):
    return np.abs(x**3 - 5*x + 6) - 2

# Generate x values
x = np.linspace(-2, 2, 500)
y = f(x)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$f(x) = |x^3 - 5x + 6| - 2$", color='blue')

# Highlight the region where f(x) < 0
plt.fill_between(x, y, 0, where=(y < 0), color='green', alpha=0.3, label=r"$f(x) < 0$")

# Add axes, labels, and legend
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.title(r"Visualization of $|x^3 - 5x + 6| - 2 < 0$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()

# Show the plot
plt.show()


# Define the function and parameters
def f(x):
    return x**3 - 5*x + 6

L = 2
epsilon = 0.2

# Generate x values
x = np.linspace(0.7, 1.3, 500)
y = f(x)

# Calculate the bounds
upper_bound = L + epsilon
lower_bound = L - epsilon

# Plot the function and bounds
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$f(x) = x^3 - 5x + 6$", color='blue')
plt.axhline(upper_bound, color='green', linestyle='--', label=r"$L + \epsilon$")
plt.axhline(lower_bound, color='red', linestyle='--', label=r"$L - \epsilon$")

# Highlight the region satisfying the condition
plt.fill_between(x, y, where=(y >= lower_bound) & (y <= upper_bound), color='orange', alpha=0.3, label=r"$|f(x) - L| < \epsilon$")

# Add labels and legend
plt.title(r"Visualization of $|f(x) - L| < \epsilon$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()

# Show the plot
plt.show()
