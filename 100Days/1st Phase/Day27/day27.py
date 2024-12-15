import numpy as np
import matplotlib.pyplot as plt

# Function y = x^(sqrt(x))
def f(x):
    return x**np.sqrt(x)

# Derivative of y = x^(sqrt(x))
def f_derivative(x):
    return f(x) * ((np.log(x) / (2 * np.sqrt(x))) + (1 / np.sqrt(x)))

# Create x values for plotting (avoid x <= 0 due to sqrt and log restrictions)
x = np.linspace(0.1, 5, 500)

y = f(x) # Function values
y_prime = f_derivative(x) # Derivative values

# Plotting the function and its derivative
plt.figure(figsize=(10, 6))

# Plot y = x^(sqrt(x))
plt.plot(x, y, label=r'$y = x^{\sqrt{x}}$', color='blue')

# Plot its derivative
plt.plot(x, y_prime, label=r"$y' = \text{derivative of } x^{\sqrt{x}}$", color='red', linestyle='--')

# Highlight critical points (where y' = 0)
critical_x = x[np.isclose(y_prime, 0, atol=1e-3)]  # Find x where derivative ~ 0
critical_y = f(critical_x)
plt.scatter(critical_x, critical_y, color='green', label='Critical Points', zorder=5)

# Titles and labels
plt.title("Visualization of $y = x^{\sqrt{x}}$ and its Derivative", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.axhline(0, color='black', linewidth=0.7, linestyle='--')
plt.axvline(0, color='black', linewidth=0.7, linestyle='--')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.show()

# Explanation of Extreme Value Theorem and Fermat's Theorem
# Extreme Value Theorem: Highlight max and min values in the domain of [0.1, 5]
max_x = x[np.argmax(y)]
min_x = x[np.argmin(y)]

print("Extreme Value Theorem:")
print(f"Maximum value of y = {f(max_x):.2f} at x = {max_x:.2f}")
print(f"Minimum value of y = {f(min_x):.2f} at x = {min_x:.2f}")

# Fermat's Theorem: Critical points where y' = 0
if critical_x.size > 0:
    print("\nFermat's Theorem:")
    for cx, cy in zip(critical_x, critical_y):
        print(f"Critical Point at x = {cx:.2f}, y = {cy:.2f}")


import matplotlib.animation as animation

# Define the function f(x) and its derivative f'(x)
def f(x):
    return x**2 - 4*x + 3  # A simple quadratic function with a minimum

def f_prime(x):
    return 2*x - 4  # Derivative of f(x)

# Generate x values
x = np.linspace(-2, 6, 400)
y = f(x)

# Set up the figure and axis
fig, ax = plt.subplots()
line, = ax.plot(x, y, label='f(x) = x^2 - 4x + 3')
tangent_line, = ax.plot([], [], 'r--', label="Tangent Line")
point, = ax.plot([], [], 'bo', label="Point of Interest")

ax.set_xlim(-2, 6)
ax.set_ylim(-2, 8)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.legend()

# Animation function
def animate(i):
    # Moving point along the function
    x_point = -1 + 0.1 * i
    y_point = f(x_point)

    # Tangent line at the point
    slope = f_prime(x_point)
    y_tangent = slope * (x - x_point) + y_point  # Equation of the tangent line
    
    point.set_data([x_point], [y_point])  # Pass lists for x and y data
    tangent_line.set_data(x, y_tangent)

    return point, tangent_line

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=50, interval=100, blit=True)

plt.show()



# Define the function f(x) and its derivative f'(x)
def f(x):
    return x**3 - 6*x**2 + 9*x

def f_prime(x):
    return 3*x**2 - 12*x + 9  # Derivative of f(x)

# Generate x values
x = np.linspace(0, 5, 400)
y = f(x)

# Calculate the critical points where the derivative equals zero
critical_x = np.roots([3, -12, 9, 0])  # Solve f'(x) = 0 for critical points
critical_x = critical_x[np.isreal(critical_x)]  # Keep only real roots
critical_x = np.real(critical_x)  # Convert complex roots to real if any

# Calculate the y values of the critical points
critical_y = f(critical_x)

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$f(x) = x^3 - 6x^2 + 9x$', color='blue')

# Highlight critical points (local maxima and minima)
plt.scatter(critical_x, critical_y, color='red', zorder=5)

# Highlight global maximum and minimum based on the interval
max_x = x[np.argmax(y)]
min_x = x[np.argmin(y)]
plt.scatter(max_x, f(max_x), color='green', label='Global Max', zorder=5)
plt.scatter(min_x, f(min_x), color='orange', label='Global Min', zorder=5)

# Titles and labels
plt.title("Function with Local Maxima and Minima", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("f(x)", fontsize=12)
plt.axhline(0, color='black', linewidth=0.7, linestyle='--')
plt.axvline(0, color='black', linewidth=0.7, linestyle='--')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.show()

# Print information about critical points
print("Critical Points (Fermat's Theorem):")
for cx, cy in zip(critical_x, critical_y):
    print(f"Critical Point at x = {cx:.2f}, y = {cy:.2f}")

print("\nExtreme Value Theorem:")
print(f"Maximum value of y = {f(max_x):.2f} at x = {max_x:.2f}")
print(f"Minimum value of y = {f(min_x):.2f} at x = {min_x:.2f}")
