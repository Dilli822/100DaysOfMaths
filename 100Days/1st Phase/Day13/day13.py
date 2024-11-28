import numpy as np
import matplotlib.pyplot as plt

# Define G(x)
def G(x):
    return (x**2 + 1) / (2 * x**2 - x - 1)

# Define the domain excluding points of discontinuity
x = np.linspace(-10, 10, 1000)
x_discontinuity = [-1/2, 1]  # Discontinuities
x = x[~np.isin(x, x_discontinuity)]

# Plot G(x)
plt.figure(figsize=(8, 6))
plt.plot(x, G(x), label=r"$G(x) = \frac{x^2 + 1}{2x^2 - x - 1}$", color="blue")
plt.axvline(x=-1/2, color="red", linestyle="--", label="Discontinuity at x=-1/2")
plt.axvline(x=1, color="red", linestyle="--", label="Discontinuity at x=1")

# Labels and legends
plt.title("Plot of G(x) with Discontinuities Highlighted", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("G(x)", fontsize=12)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()


# Define the function
def f(x):
    return (2*x + 3) / (x - 2)

# Create a range of x values, excluding the vertical asymptote at x=2
x = np.linspace(-10, 10, 500)
x = x[x != 2]  # Exclude x=2 to avoid division by zero

# Calculate y values
y = f(x)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$f(x) = \frac{2x+3}{x-2}$", color='blue')
plt.axvline(x=2, color='red', linestyle='--', label='Vertical Asymptote $x=2$')  # Vertical asymptote
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)  # Horizontal axis
plt.axhline(y=2, color='green', linestyle='--', label='Horizontal Asymptote $y=2$')  # Horizontal asymptote

# Add labels and title
plt.title("Graph of $f(x) = \\frac{2x+3}{x-2}$", fontsize=14)
plt.xlabel("$x$", fontsize=12)
plt.ylabel("$f(x)$", fontsize=12)
plt.ylim(-10, 10)  # Limit y-axis for better visualization
plt.legend()
plt.grid()

# Show the plot
plt.show()


# 1. Step Function Visualization
x1 = np.linspace(1, 2, 100)
x2 = np.linspace(2, 4, 100)
y1 = 2 * x1 - 2  # For 1 <= x <= 2
y2 = np.full_like(x2, 3)  # For 2 < x <= 4

plt.figure(figsize=(10, 6))
plt.plot(x1, y1, label="f(x) = 2x - 2 (1 ≤ x ≤ 2)", color="blue")
plt.plot(x2, y2, label="f(x) = 3 (2 < x ≤ 4)", color="orange")

# Marking the discontinuity at x=2
plt.scatter([2], [2], color="blue", edgecolor="black", s=100, label="LHL at x=2")
plt.scatter([2], [3], color="orange", edgecolor="black", s=100, label="RHL at x=2")

plt.title("Step Function Visualization", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("f(x)", fontsize=12)
plt.axvline(2, color="red", linestyle="--", label="x=2 (discontinuity)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Polynomial Root Visualization
x = np.linspace(1, 2, 500)
f = 4 * x**3 - 6 * x**2 + 3 * x - 2

plt.figure(figsize=(10, 6))
plt.plot(x, f, label="f(x) = 4x³ - 6x² + 3x - 2", color="green")

# Marking points f(1) and f(2)
plt.scatter([1], [4*1**3 - 6*1**2 + 3*1 - 2], color="red", label="f(1)")
plt.scatter([2], [4*2**3 - 6*2**2 + 3*2 - 2], color="blue", label="f(2)")

plt.axhline(0, color="black", linestyle="--", label="y = 0 (root)")
plt.title("Polynomial Root Visualization", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("f(x)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()