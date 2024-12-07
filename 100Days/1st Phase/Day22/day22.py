import numpy as np
import matplotlib.pyplot as plt

# Define the x range
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

# Define functions and their derivatives
sin_x = np.sin(x)
cos_x = np.cos(x)
tan_x = np.tan(x)

d_sin_x = np.cos(x)  # Derivative of sin(x)
d_cos_x = -np.sin(x)  # Derivative of cos(x)
d_tan_x = 1 / np.cos(x)**2  # Derivative of tan(x)

# Create plots
plt.figure(figsize=(12, 8))

# Plot sin(x) and its derivative
plt.subplot(3, 1, 1)
plt.plot(x, sin_x, label="sin(x)", color="blue")
plt.plot(x, d_sin_x, label="cos(x)", linestyle="--", color="orange")
plt.title("sin(x) and its Derivative")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid(True)

# Plot cos(x) and its derivative
plt.subplot(3, 1, 2)
plt.plot(x, cos_x, label="cos(x)", color="green")
plt.plot(x, d_cos_x, label="-sin(x)", linestyle="--", color="red")
plt.title("cos(x) and its Derivative")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid(True)

# Plot tan(x) and its derivative
plt.subplot(3, 1, 3)
plt.plot(x, tan_x, label="tan(x)", color="purple")
plt.plot(x, d_tan_x, label="sec^2(x)", linestyle="--", color="brown")
plt.ylim(-10, 10)  # Limit y-axis to avoid infinities
plt.title("tan(x) and its Derivative")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

## Fundamental Formula on Differentiation

# Define x range
x = np.linspace(-2, 2, 500)

# Power Rule Example: f(x) = x^3
f1 = x**3
df1 = 3 * x**2  # Derivative of x^3

# Sum Rule Example: f(x) = x^2 + 3x
f2 = x**2 + 3*x
df2 = 2*x + 3  # Derivative of x^2 + 3x

# Product Rule Example: f(x) = x * sin(x)
f3 = x * np.sin(x)
df3 = np.sin(x) + x * np.cos(x)  # Product rule: (x)'sin(x) + x(cos(x))

# Quotient Rule Example: f(x) = sin(x) / x
f4 = np.sin(x) / x
df4 = (x * np.cos(x) - np.sin(x)) / x**2  # Quotient rule

# General Power Rule Example: f(x) = (x^2 + 1)^2
f5 = (x**2 + 1)**2
df5 = 2 * (x**2 + 1) * (2 * x)  # Chain rule in action

# Chain Rule Example: f(x) = sin(x^2)
f6 = np.sin(x**2)
df6 = 2 * x * np.cos(x**2)  # Chain rule

# Create plots
plt.figure(figsize=(15, 12))

# Power Rule
plt.subplot(3, 2, 1)
plt.plot(x, f1, label="f(x) = x^3", color="blue")
plt.plot(x, df1, label="f'(x) = 3x^2", linestyle="--", color="orange")
plt.title("Power Rule")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid(True)

# Sum Rule
plt.subplot(3, 2, 2)
plt.plot(x, f2, label="f(x) = x^2 + 3x", color="green")
plt.plot(x, df2, label="f'(x) = 2x + 3", linestyle="--", color="red")
plt.title("Sum Rule")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid(True)

# Product Rule
plt.subplot(3, 2, 3)
plt.plot(x, f3, label="f(x) = x*sin(x)", color="purple")
plt.plot(x, df3, label="f'(x) = sin(x) + x*cos(x)", linestyle="--", color="brown")
plt.title("Product Rule")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid(True)

# Quotient Rule
plt.subplot(3, 2, 4)
plt.plot(x, f4, label="f(x) = sin(x)/x", color="cyan")
plt.plot(x, df4, label="f'(x)", linestyle="--", color="magenta")
plt.title("Quotient Rule")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid(True)

# General Power Rule
plt.subplot(3, 2, 5)
plt.plot(x, f5, label="f(x) = (x^2 + 1)^2", color="pink")
plt.plot(x, df5, label="f'(x) = 2(x^2+1)(2x)", linestyle="--", color="navy")
plt.title("General Power Rule")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid(True)

# Chain Rule
plt.subplot(3, 2, 6)
plt.plot(x, f6, label="f(x) = sin(x^2)", color="gold")
plt.plot(x, df6, label="f'(x) = 2x*cos(x^2)", linestyle="--", color="darkgreen")
plt.title("Chain Rule")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
