import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad

# Define the function f(x)
def f(x):
    return x**2

# Generate x values for plotting
x = np.linspace(-2, 2, 500)
y = f(x)

# 1. Indefinite Integral Visualization (2D Plot)
# Indefinite integral (antiderivative): F(x) = x^3 / 3
def F(x):
    return x**3 / 3

F_y = F(x)

# 2D plot for indefinite integral
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$f(x) = x^2$", color="blue")
plt.plot(x, F_y, label=r"$\int f(x) dx = \frac{x^3}{3} + C$", color="orange", linestyle="--")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.title("Indefinite Integral (Antiderivative)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# 2. Definite Integral Visualization (2D Plot)
# Compute definite integral between -1 and 1
x_def = np.linspace(-1, 1, 100)
y_def = f(x_def)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$f(x) = x^2$", color="blue")
plt.fill_between(x_def, y_def, color="lightblue", alpha=0.5, label="Area under the curve")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.title("Definite Integral: Area under the Curve (-1 to 1)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# 3. 3D Visualization of the Function and Integral
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Create mesh grid for 3D plotting
X = np.linspace(-2, 2, 500)
Y = np.linspace(-2, 2, 500)
X, Y = np.meshgrid(X, Y)
Z = X**2  # Function in 3D

ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
ax.set_title("3D Visualization of Function $f(x) = x^2$")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("f(x)")
plt.show()


# Generate x values
x = np.linspace(-2, 2, 500)
y = f(x)
F_y = F(x)

# Definite integral: Area under the curve from a to b
a, b = -1, 1
area, _ = quad(f, a, b)  # Using scipy's quad for numerical integration

# 1. Plot Antiderivative
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r"$f(x) = x^2$", color="blue")
plt.plot(x, F_y, label=r"Antiderivative $\int f(x) dx = \frac{x^3}{3} + C$", color="orange", linestyle="--")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.title("Antiderivative and Function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# 2. Plot Area Under the Curve
x_fill = np.linspace(a, b, 100)
y_fill = f(x_fill)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r"$f(x) = x^2$", color="blue")
plt.fill_between(x_fill, y_fill, color="lightblue", alpha=0.5, label=f"Area = {area:.2f}")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.title("Area Under the Curve")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()



# Define functions and their antiderivatives
functions = [
    {"func": lambda x: x**2, "antideriv": lambda x: x**3 / 3, "label": r"$\int x^2 dx = \frac{x^3}{3} + C$", "title": r"$f(x) = x^2$"},
    {"func": lambda x: np.sin(x), "antideriv": lambda x: -np.cos(x), "label": r"$\int \sin(x) dx = -\cos(x) + C$", "title": r"$f(x) = \sin(x)$"},
    {"func": lambda x: np.exp(x), "antideriv": lambda x: np.exp(x), "label": r"$\int e^x dx = e^x + C$", "title": r"$f(x) = e^x$"},
    {"func": lambda x: 1 / (1 + x**2), "antideriv": lambda x: np.arctan(x), "label": r"$\int \frac{1}{1+x^2} dx = \arctan(x) + C$", "title": r"$f(x) = \frac{1}{1+x^2}$"},
    {"func": lambda x: 1 / x, "antideriv": lambda x: np.log(np.abs(x)), "label": r"$\int \frac{1}{x} dx = \ln|x| + C$", "title": r"$f(x) = \frac{1}{x}$"},
]

# Plot settings
x_values = np.linspace(-2, 2, 500)

# Loop through each function and its antiderivative to plot separately
for func_data in functions:
    f = func_data["func"]
    F = func_data["antideriv"]
    label = func_data["label"]
    title = func_data["title"]
    
    # Compute function and antiderivative
    y = f(x_values)
    antiderivative_y = F(x_values)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y, label=r"$f(x)$", color="blue", linewidth=2)
    plt.plot(x_values, antiderivative_y, label=label, color="orange", linestyle="--", linewidth=2)
    plt.axhline(0, color="black", linewidth=0.8, linestyle=":")
    plt.axvline(0, color="black", linewidth=0.8, linestyle=":")
    plt.title(f"Function and Antiderivative: {title}", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(fontsize=12)
    plt.grid(alpha=0.4)
    plt.show()
