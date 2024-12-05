import numpy as np
import matplotlib.pyplot as plt

# Define the curve function and its derivative
def f(x):
    return x**2  # Example: A parabola

def df(x):
    return 2 * x  # Derivative of the parabola

# Generate data for the curve
x = np.linspace(-2, 2, 400)
y = f(x)

# Define the tangent point
x_tangent = 1
y_tangent = f(x_tangent)
slope = df(x_tangent)

# Tangent line equation
def tangent_line(x):
    return slope * (x - x_tangent) + y_tangent

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the curve
plt.plot(x, y, label="Curve: $y=x^2$", color='blue')

# Plot the tangent point
plt.scatter([x_tangent], [y_tangent], color='red', label=f"Tangent Point ({x_tangent}, {y_tangent})")

# Plot the tangent line
plt.plot(x, tangent_line(x), color='green', linestyle='--', label="Tangent Line")

# Add labels and legend
plt.title("Tangent Line and Slope Visualization", fontsize=16)
plt.xlabel("x-axis", fontsize=12)
plt.ylabel("y-axis", fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)  # x-axis
plt.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)  # y-axis
plt.legend()
plt.grid(alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()


# Define a continuous curve y = f(x)
def f(x):
    return x**3 - 3*x**2 + 2*x + 1  # A smooth cubic function for demonstration

# Define points P and Q on the curve
x_p = 1.0  # x-coordinate of point P
x_q = 2.0  # x-coordinate of point Q (P + Δx)
y_p = f(x_p)  # y-coordinate of point P
y_q = f(x_q)  # y-coordinate of point Q

# Compute slope of the secant line
slope = (y_q - y_p) / (x_q - x_p)

# Define the tangent line using the slope
x_tangent = np.linspace(0, 3, 100)  # Range for the tangent line
y_tangent = slope * (x_tangent - x_p) + y_p

# Plot the curve
x_vals = np.linspace(0, 3, 500)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals, label="Curve: $y = f(x)$", color="blue")

# Plot points P and Q
plt.scatter([x_p, x_q], [y_p, y_q], color="red", zorder=5, label="Points P and Q")

# Plot the tangent line
plt.plot(x_tangent, y_tangent, label="Tangent Line", color="orange", linestyle="--")

# Annotate the points and increments
plt.annotate("P", (x_p, y_p), textcoords="offset points", xytext=(-15, -10), ha="center")
plt.annotate("Q", (x_q, y_q), textcoords="offset points", xytext=(10, -10), ha="center")
plt.annotate(r"$\Delta x$", ((x_p + x_q) / 2, y_p - 0.5), ha="center")
plt.annotate(r"$\Delta y$", (x_q + 0.1, (y_p + y_q) / 2), ha="center")

# Highlight the horizontal (Δx) and vertical (Δy) lines
plt.plot([x_p, x_q], [y_p, y_p], color="green", linestyle=":", label=r"$\Delta x$")
plt.plot([x_q, x_q], [y_p, y_q], color="purple", linestyle=":", label=r"$\Delta y$")

# Customize the plot
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.title("Tangent Line to a Curve")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.grid(alpha=0.3)
plt.show()

# Define a generic continuous curve
def f(x):
    return x**3 - 3*x**2 + 2  # Example curve

# Points on the curve
x_p = 1.5  # Point P (x, y) on the curve
x_q = 2.0  # Point Q (x', y') on the curve
y_p = f(x_p)
y_q = f(x_q)

# Calculate the slope (dy/dx)
delta_x = x_q - x_p
delta_y = y_q - y_p
slope = delta_y / delta_x

# Tangent line equation: y = mx + c
# Slope m = slope, and c = y - mx
c = y_p - slope * x_p
tangent_line = lambda x: slope * x + c

# Plotting the curve, points, and tangent line
x_vals = np.linspace(0, 3, 500)  # Range of x values for the curve
y_vals = f(x_vals)

plt.figure(figsize=(8, 6))

# Plot the curve
plt.plot(x_vals, y_vals, label="Curve: $y=f(x)$", color="blue")

# Plot the tangent line
tangent_x = np.linspace(1, 2.5, 100)  # Tangent segment
plt.plot(tangent_x, tangent_line(tangent_x), label="Tangent Line at P", color="red", linestyle="--")

# Plot points P and Q
plt.scatter([x_p, x_q], [y_p, y_q], color="green", zorder=5, label="Points P and Q")
plt.text(x_p, y_p, " P", fontsize=12, color="green")
plt.text(x_q, y_q, " Q", fontsize=12, color="green")

# Mark increments Δx and Δy
plt.arrow(x_p, y_p, delta_x, 0, color="orange", head_width=0.1, length_includes_head=True)
plt.arrow(x_q, y_p, 0, delta_y, color="purple", head_width=0.1, length_includes_head=True)
plt.text(x_p + delta_x / 2, y_p - 0.5, r"$\Delta x$", fontsize=12, color="orange")
plt.text(x_q + 0.1, y_p + delta_y / 2, r"$\Delta y$", fontsize=12, color="purple")

# Labels and legend
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.title("Tangent Line to a Curve")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.grid(True)

plt.show()
