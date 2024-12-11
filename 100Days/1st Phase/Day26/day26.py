import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivatives
# Function: f(x) = e^(x+1) + 1
f = lambda x: np.exp(x + 1) + 1
f_prime = lambda x: np.exp(x + 1)
f_double_prime = lambda x: np.exp(x + 1)

# Define x values for visualization
x = np.linspace(-2, 2, 500)

# Calculate function values
f_values = f(x)
f_prime_values = f_prime(x)
f_double_prime_values = f_double_prime(x)

# Plot the function and its derivatives
plt.figure(figsize=(12, 6))

# Plot f(x)
plt.plot(x, f_values, label='$f(x) = e^{x+1} + 1$', color='blue')

# Plot f'(x)
plt.plot(x, f_prime_values, label="$f'(x) = e^{x+1}$", color='green', linestyle='--')

# Plot f''(x)
plt.plot(x, f_double_prime_values, label="$f''(x) = e^{x+1}$", color='red', linestyle=':')

# Highlight x = 0
plt.axvline(0, color='black', linestyle='-.', label='x = 0')
plt.axhline(0, color='black', linewidth=0.5)

# Add title and labels
plt.title("Visualization of Function, First Derivative, and Second Derivative", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)

# Add legend
plt.legend(fontsize=10)

# Show grid and plot
plt.grid(alpha=0.3)
plt.show()

# Demonstrating the quotient rule and chain rule
# Function: h(x) = (e^x + 1) / (x + 1)
quotient_numerator = lambda x: np.exp(x) + 1
quotient_denominator = lambda x: x + 1

# Quotient Rule: h'(x) = [g'(x)h(x) - g(x)h'(x)] / [h(x)]^2
h_prime = lambda x: (
    (np.exp(x) * (x + 1) - (np.exp(x) + 1) * 1) / (x + 1) ** 2
)

# Chain Rule Example: y = e^(g(x)), g(x) = x + 1
# y' = e^(g(x)) * g'(x)
chain_rule = lambda x: np.exp(x + 1) * 1

print("Quotient Rule Derivative at x=1:", h_prime(1))
print("Chain Rule Derivative at x=1:", chain_rule(1))


import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivatives
# Function: f(x) = |x|
def f(x):
    return np.abs(x)

def f_prime(x):
    return np.where(x > 0, 1, np.where(x < 0, -1, np.nan))  # Derivative undefined at x=0

# Define x values for visualization
x = np.linspace(-2, 2, 500)

# Calculate function values and derivatives
f_values = f(x)
f_prime_values = f_prime(x)

# Plot the function and its derivatives
plt.figure(figsize=(12, 6))

# Plot f(x)
plt.plot(x, f_values, label='$f(x) = |x|$', color='blue')

# Highlight x = 0 and plot f'(x)
plt.axvline(0, color='black', linestyle='-.', label='x = 0 (not differentiable)')
plt.plot(x[x != 0], f_prime_values[x != 0], label="$f'(x)$", color='green', linestyle='--')

# Add title and labels
plt.title("Visualization of $f(x) = |x|$ and its Derivative", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)

# Add legend
plt.legend(fontsize=10)

# Show grid and plot
plt.grid(alpha=0.3)
plt.show()

# Theorem demonstration
print("Theorem: f(x) = |x| is not differentiable at x = 0.")
print("Reason: The left-hand derivative (-1) does not equal the right-hand derivative (1) at x = 0.")
