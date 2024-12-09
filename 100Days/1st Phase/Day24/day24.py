import numpy as np
import matplotlib.pyplot as plt

# Define a range of x values
x = np.linspace(-2, 2, 500)

# Define some sample functions
f = x**2  # Example for power rule
g = np.sin(x)  # Example for chain rule
h = np.exp(x)  # Additional function for combining rules

# Derivatives for visualization
df = 2 * x  # Derivative of x^2
dg = np.cos(x)  # Derivative of sin(x)
dh = np.exp(x)  # Derivative of exp(x)

# Rules for combinations
sum_func = f + g
sum_derivative = df + dg

product_func = f * g
product_derivative = f * dg + g * df

quotient_func = f / h
quotient_derivative = (df * h - f * dh) / h**2

chain_func = np.sin(x**2)
chain_derivative = 2 * x * np.cos(x**2)  # Chain rule: d/dx[sin(x^2)] = cos(x^2) * 2x

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(12, 16))
axs = axs.flatten()

# Power Rule
axs[0].plot(x, f, label="f(x) = x^2")
axs[0].plot(x, df, label="f'(x) = 2x", linestyle="--")
axs[0].set_title("Power Rule")
axs[0].legend()

# Sum Rule
axs[1].plot(x, sum_func, label="f(x) + g(x)")
axs[1].plot(x, sum_derivative, label="(f+g)'(x)", linestyle="--")
axs[1].set_title("Sum Rule")
axs[1].legend()

# Product Rule
axs[2].plot(x, product_func, label="f(x) * g(x)")
axs[2].plot(x, product_derivative, label="(f*g)'(x)", linestyle="--")
axs[2].set_title("Product Rule")
axs[2].legend()

# Quotient Rule
axs[3].plot(x, quotient_func, label="f(x) / h(x)")
axs[3].plot(x, quotient_derivative, label="(f/h)'(x)", linestyle="--")
axs[3].set_title("Quotient Rule")
axs[3].legend()

# Chain Rule
axs[4].plot(x, chain_func, label="sin(x^2)")
axs[4].plot(x, chain_derivative, label="(sin(x^2))'", linestyle="--")
axs[4].set_title("Chain Rule")
axs[4].legend()

# Extra space for summary
axs[5].axis("off")
axs[5].text(0.5, 0.5, 
    "This visualization demonstrates:\n"
    "- Power Rule: f(x) = x^2, f'(x) = 2x\n"
    "- Sum Rule: (f+g)' = f' + g'\n"
    "- Product Rule: (fg)' = f'g + fg'\n"
    "- Quotient Rule: (f/h)' = (f'h - fh') / h^2\n"
    "- Chain Rule: d/dx sin(x^2) = 2x * cos(x^2)",
    ha="center", va="center", fontsize=12)

plt.tight_layout()
plt.show()


import matplotlib.animation as animation

# Set up the figure and subplots for animation
fig, ax = plt.subplots(figsize=(8, 6))

# Initial setup for animation
line_original, = ax.plot([], [], label="Original Function")
line_derivative, = ax.plot([], [], linestyle="--", label="Derivative")
title = ax.text(0.5, 1.05, "", ha="center", va="center", transform=ax.transAxes, fontsize=14)

# General animation setup
def init():
    ax.set_xlim(-2, 2)
    ax.set_ylim(-5, 5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()
    return line_original, line_derivative, title

# Animation function
def animate(i):
    rules = [
        ("Power Rule: f(x) = x^2", f, df),
        ("Sum Rule: f(x) + g(x)", sum_func, sum_derivative),
        ("Product Rule: f(x) * g(x)", product_func, product_derivative),
        ("Quotient Rule: f(x) / h(x)", quotient_func, quotient_derivative),
        ("Chain Rule: sin(x^2)", chain_func, chain_derivative)
    ]
    
    if i < len(rules):
        rule_name, original_func, derivative_func = rules[i]
        line_original.set_data(x, original_func)
        line_derivative.set_data(x, derivative_func)
        title.set_text(rule_name)
    return line_original, line_derivative, title

# Create the animation
ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=len([f, sum_func, product_func, quotient_func, chain_func]), interval=2000, repeat=True
)

# Display the animation
plt.show()

