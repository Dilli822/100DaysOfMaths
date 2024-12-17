import numpy as np
import matplotlib.pyplot as plt

# Define a range for x values
a, b = -5, 5  # Interval [a, b]
x = np.linspace(a, b, 500)

# Case 1: Constant function (derivative = 0)
constant1 = 3  # Example constant value
constant2 = -2  # Another constant value
f1 = lambda x: np.full_like(x, constant1)
f2 = lambda x: np.full_like(x, constant2)

# Case 2: Increasing function (f'(x) > 0)
inc_f = lambda x: x**2 + 2*x  # Derivative is positive on part of the range
inc_f_prime = lambda x: 2*x + 2

# Case 3: Decreasing function (f'(x) < 0)
dec_f = lambda x: -x**2 + 2*x  # Derivative is negative on part of the range
dec_f_prime = lambda x: -2*x + 2

# Compute values
f1_values = f1(x)
f2_values = f2(x)
inc_f_values = inc_f(x)
dec_f_values = dec_f(x)

# Plot
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Functions with zero derivatives (constant functions)
axs[0].plot(x, f1_values, label=f"f(x) = {constant1}", color='blue')
axs[0].plot(x, f2_values, label=f"f(x) = {constant2}", color='green')
axs[0].set_title("Functions with Zero Derivatives (Constant Functions)")
axs[0].legend()
axs[0].grid(True)

# Plot 2: Increasing function (f'(x) > 0)
axs[1].plot(x, inc_f_values, label="$f(x) = x^2 + 2x$", color='orange')
axs[1].plot(x, inc_f_prime(x), label="$f'(x) = 2x + 2$", linestyle='--', color='red')
axs[1].set_title("Increasing Function (f'(x) > 0)")
axs[1].legend()
axs[1].grid(True)

# Plot 3: Decreasing function (f'(x) < 0)
axs[2].plot(x, dec_f_values, label="$f(x) = -x^2 + 2x$", color='purple')
axs[2].plot(x, dec_f_prime(x), label="$f'(x) = -2x + 2$", linestyle='--', color='red')
axs[2].set_title("Decreasing Function (f'(x) < 0)")
axs[2].legend()
axs[2].grid(True)

# Display
plt.tight_layout()
plt.show()
