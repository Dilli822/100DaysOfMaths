import numpy as np
import matplotlib.pyplot as plt

# Visualize a continuous polynomial function
x = np.linspace(-10, 10, 100)
y = x**3 - 2*x + 1
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title("Continuous Polynomial Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

# Visualize a continuous rational function
x = np.linspace(-10, 10, 100)
y = (x + 2) / (x - 1)
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title("Continuous Rational Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

# Visualize a continuous trigonometric function
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title("Continuous Trigonometric Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()


# Example function from Image 1
def f(x):
    return np.log(x) + np.tan(1/x)

x = np.linspace(0.01, 10, 1000)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Continuous Function: f(x) = log(x) + tan(1/x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

# Visualizing the intersection point from Image 2
x = np.linspace(-2, 2, 1000)
y1 = np.tan(x)
y2 = 0.8
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label="y = tan(x)")
plt.plot(x, [y2] * len(x), label="y = 0.8")
plt.scatter(0.67, 0.88, color='r', s=100, label="Intersection Point")
plt.title("Intersection of y = tan(x) and y = 0.8")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()



# Define epsilon and delta
epsilon = 0.2
delta = epsilon**2

# Define function
x = np.linspace(0, delta + 0.1, 1000)
y = np.sqrt(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r"$f(x) = \sqrt{x}$", color="blue")
plt.axvline(delta, color="orange", linestyle="--", label=r"$\delta = \epsilon^2$")
plt.axhline(epsilon, color="green", linestyle="--", label=r"$\epsilon$")
plt.fill_between(x, 0, y, color="blue", alpha=0.1, label=r"$\delta-\epsilon$ Region")
plt.scatter([delta], [epsilon], color="red", s=100, label="Point: $(\delta, \epsilon)$")
plt.title(r"$\delta-\epsilon$ Neighborhood for $f(x) = \sqrt{x}$")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.grid()
plt.show()


# Values of interest
pi_4 = np.pi / 4  # π/4
x_vals = [0.67, 0.88]

# Plotting
plt.figure(figsize=(10, 6))
plt.axvline(x=pi_4, color='green', linestyle='--', label=r"$x = \frac{\pi}{4}$")
plt.scatter(x_vals, [0] * len(x_vals), color='red', s=100, label="Points: 0.67 and 0.88")
plt.text(0.67, -0.1, "0.67", color="red", ha="center")
plt.text(0.88, -0.1, "0.88", color="red", ha="center")
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Symmetry of Intervals Around $x = \frac{\pi}{4}$")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.grid()
plt.show()

# Define the x range and exclude discontinuities
x = np.linspace(0, 2, 1000)
x = x[np.abs(np.mod(x - np.pi / 2, np.pi)) > 0.1]  # Avoid discontinuities

# Define functions
y_tan = np.tan(x)
y_const = 0.8

# Intersection point
x_intersect = np.arctan(0.8)
y_intersect = 0.8

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y_tan, label=r"$y = \tan(x)$")
plt.axhline(y=0.8, color='orange', label=r"$y = 0.8$")
plt.scatter(x_intersect, y_intersect, color='red', s=100, label="Intersection: (0.674, 0.8)")
plt.title("Intersection of $y = \tan(x)$ and $y = 0.8$")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.grid()
plt.show()

