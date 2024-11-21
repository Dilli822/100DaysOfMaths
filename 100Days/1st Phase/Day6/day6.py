import numpy as np
import matplotlib.pyplot as plt

# Define various types of functions

def continuous_function(x):
    return x**2  # Example of a continuous function: y = x^2

def linear_function(x):
    return 2 * x + 1  # Example of a continuous linear function

def sine_function(x):
    return np.sin(x)  # Example of a continuous sine function

def discontinuous_function(x):
    return np.where(x < 0, 1, 2)  # Example of a discontinuous function: jump discontinuity

def removable_discontinuity(x):
    return np.sin(x) / x  # Removable discontinuity at x = 0

def infinite_discontinuity(x):
    return 1 / x  # Infinite discontinuity at x = 0

def step_function(x):
    return np.where(x >= 0, 1, 0)  # Step function

def absolute_function(x):
    return np.abs(x)  # Continuous absolute function

def piecewise_linear(x):
    return np.where(x < 0, -x, x)  # Piecewise linear, continuous

# Generate x values
x_cont = np.linspace(-10, 10, 400)
x_disc = np.linspace(-10, 10, 400)
x_removable = np.linspace(-10, 10, 400)
x_removable = x_removable[x_removable != 0]
x_inf = np.linspace(-10, 10, 400)
x_inf = x_inf[x_inf != 0]
x_step = np.linspace(-10, 10, 400)
x_abs = np.linspace(-10, 10, 400)
x_piecewise = np.linspace(-10, 10, 400)

# Plotting the functions
plt.figure(figsize=(15, 15))

# Continuous Function
plt.subplot(5, 3, 1)
plt.plot(x_cont, continuous_function(x_cont), label="y = x^2", color='blue')
plt.title("Continuous Function y = x^2")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Linear Function
plt.subplot(5, 3, 2)
plt.plot(x_cont, linear_function(x_cont), label="y = 2x + 1", color='green')
plt.title("Continuous Function y = 2x + 1")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Sine Function
plt.subplot(5, 3, 3)
plt.plot(x_cont, sine_function(x_cont), label="y = sin(x)", color='red')
plt.title("Continuous Function y = sin(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Discontinuous Function (Jump Discontinuity)
plt.subplot(5, 3, 4)
plt.plot(x_disc, discontinuous_function(x_disc), label="Jump Discontinuity", color='orange')
plt.title("Discontinuous Function (Jump)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Removable Discontinuity
plt.subplot(5, 3, 5)
plt.plot(x_removable, removable_discontinuity(x_removable), label=r"$\sin(x)/x$", color='purple')
plt.scatter([0], [1], color='yellow', label="Removable Discontinuity", zorder=5)  # Indicate the defined point at x=0
plt.title("Discontinuous Function (Removable)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Infinite Discontinuity
plt.subplot(5, 3, 6)
plt.plot(x_inf, infinite_discontinuity(x_inf), label=r"$f(x) = 1/x$", color='brown')
plt.title("Discontinuous Function (Infinite)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Step Function
plt.subplot(5, 3, 7)
plt.plot(x_step, step_function(x_step), label="Step Function", color='cyan')
plt.title("Discontinuous Function (Step)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Absolute Function
plt.subplot(5, 3, 8)
plt.plot(x_abs, absolute_function(x_abs), label="Absolute Function", color='magenta')
plt.title("Continuous Function y = |x|")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Piecewise Linear Function
plt.subplot(5, 3, 9)
plt.plot(x_piecewise, piecewise_linear(x_piecewise), label="Piecewise Linear", color='pink')
plt.title("Continuous Piecewise Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Continuous Exponential Function
plt.subplot(5, 3, 10)
plt.plot(x_cont, np.exp(x_cont / 10), label="Exponential Function", color='green')
plt.title("Continuous Function (Exponential)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Discontinuous with Oscillations
plt.subplot(5, 3, 11)
plt.plot(x_cont, np.where(x_cont % 2 == 0, 1, -1), label="Oscillating Discontinuity", color='blue')
plt.title("Discontinuous Oscillating Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Discontinuous with Jump at Multiple Points
plt.subplot(5, 3, 12)
plt.plot(x_disc, np.where(np.abs(x_disc) > 5, 1, 0), label="Multiple Jump Discontinuity", color='red')
plt.title("Multiple Jump Discontinuity")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Continuous Logarithmic Function
plt.subplot(5, 3, 13)
plt.plot(x_cont[x_cont > 0], np.log(x_cont[x_cont > 0]), label="Logarithmic Function", color='orange')
plt.title("Continuous Logarithmic Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Piecewise Defined Discontinuous Function
def piecewise_discontinuous(x):
    return np.where(x < 0, -x, np.sin(x))  # Piecewise function with discontinuities
plt.subplot(5, 3, 14)
plt.plot(x_cont, piecewise_discontinuous(x_cont), label="Piecewise Discontinuous", color='purple')
plt.title("Piecewise Discontinuous Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Continuous Quadratic Function
plt.subplot(5, 3, 15)
plt.plot(x_cont, continuous_function(x_cont), label="y = x^2", color='brown')
plt.title("Continuous Function (Quadratic)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()




