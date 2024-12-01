import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return (x**2 + 2*x + 3) / (x - 1)

# Polynomial long division: (x^2 + 2x + 3) ÷ (x - 1)
# Quotient: x + 3 (This is the slant asymptote)

# Define the slant asymptote
def slant_asymptote(x):
    return x + 3

# Create x values for plotting
x = np.linspace(-10, 10, 1000)
x = x[x != 1]  # Avoid x = 1 where the function is undefined

# Plot the function and slant asymptote
plt.figure(figsize=(8, 6))
plt.plot(x, f(x), label="f(x) = (x^2 + 2x + 3) / (x - 1)", color="blue")
plt.plot(x, slant_asymptote(x), label="Slant Asymptote: y = x + 3", color="red", linestyle="--")

# Highlight the vertical asymptote at x = 1
plt.axvline(1, color="green", linestyle=":", label="Vertical Asymptote: x = 1")

# Add labels and legend
plt.title("Function with Slant Asymptote")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color="black", linewidth=0.5)
plt.legend()
plt.grid()

# Show the plot
plt.show()



# Define the function and its slant asymptote
def f(x):
    return (2 * x**2 - 3 * x) / (4 * x + 2)

def slant_asymptote(x):
    return 2 * x - 1  # Derived from long division of the numerator and denominator

# Create an x-range, excluding values where the function might be undefined
x = np.linspace(-10, 10, 500)
x = x[x != -0.5]  # Exclude vertical asymptote at x = -0.5

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x, f(x), label="f(x) = (2x² - 3x) / (4x + 2)", color="blue")
plt.plot(x, slant_asymptote(x), label="Slant Asymptote: y = 2x - 1", linestyle="--", color="red")

# Add labels, legend, and grid
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("x", fontsize=12)
plt.ylabel("f(x)", fontsize=12)
plt.title("Function and Slant Asymptote", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Show the plot
plt.show()
