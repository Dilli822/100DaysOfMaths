import numpy as np
import matplotlib.pyplot as plt

# Function y = a^x and its derivative
a = 2  # Base of the exponential function
x = np.linspace(-4, 4, 500)
y = a**x
dy_dx = np.log(a) * a**x

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$y = a^x$', color='blue')
plt.plot(x, dy_dx, label=r"$\frac{dy}{dx} = \ln(a) \cdot a^x$", color='red', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle='dotted')
plt.axvline(0, color='black', linewidth=0.5, linestyle='dotted')
plt.title(r'Function and Derivative of $y = a^x$', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()

# Hyperbolic functions: sinh(x), cosh(x), tanh(x)
x = np.linspace(-6, 6, 500)
sinh_x = np.sinh(x)
cosh_x = np.cosh(x)
tanh_x = np.tanh(x)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x, sinh_x, label=r'$\sinh(x)$', color='blue')
plt.plot(x, cosh_x, label=r'$\cosh(x)$', color='green')
plt.plot(x, tanh_x, label=r'$\tanh(x)$', color='orange')
plt.axhline(0, color='black', linewidth=0.5, linestyle='dotted')
plt.axvline(0, color='black', linewidth=0.5, linestyle='dotted')
plt.title('Hyperbolic Functions', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()


# Parabola y = x^2 and tangent at x = 1
x = np.linspace(-4, 4, 500)
y = x**2
tangent_line = 2 * (x - 1) + 1  # Slope = 2, passes through (1, 1)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$y = x^2$', color='blue')
plt.plot(x, tangent_line, label='Tangent at (1, 1)', color='red', linestyle='--')
plt.scatter(1, 1, color='black', label='Point (1, 1)', zorder=5)
plt.axhline(0, color='black', linewidth=0.5, linestyle='dotted')
plt.axvline(0, color='black', linewidth=0.5, linestyle='dotted')
plt.title('Parabola and Tangent Line', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()
