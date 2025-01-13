import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define acceleration function
def acceleration(t):
    return 2 * t  # a(t) = 2t

# Define velocity function (antiderivative of acceleration)
def velocity(t):
    return t**2 + C1  # v(t) = t^2 + C1

# Define position function (antiderivative of velocity)
def position(t):
    return (t**3) / 3 + C1 * t + C2  # s(t) = t^3/3 + C1 * t + C2

# Constants (Initial conditions)
C1 = 0  # Initial velocity
C2 = 0  # Initial position

# Time range
t = np.linspace(0, 5, 500)  # From t=0 to t=5

# Calculate velocity and position
v = velocity(t)
s = position(t)

# Definite integral for displacement (area under velocity curve)
displacement, _ = quad(velocity, 0, 5)

# Plot 2D graphs
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, acceleration(t), label="Acceleration (a(t))", color='r')
plt.plot(t, v, label="Velocity (v(t))", color='g')
plt.legend()
plt.title("Acceleration and Velocity vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, s, label="Position (s(t))", color='b')
plt.legend()
plt.title("Position vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.grid()

plt.tight_layout()
plt.show()

# 3D plot of acceleration, velocity, and position
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(t, acceleration(t), np.zeros_like(t), label="Acceleration (a(t))", color='r')
ax.plot(t, v, np.ones_like(t) * 5, label="Velocity (v(t))", color='g')
ax.plot(t, s, np.ones_like(t) * 10, label="Position (s(t))", color='b')

ax.set_xlabel("Time (s)")
ax.set_ylabel("Value")
ax.set_zlabel("Curve")
ax.legend()
ax.set_title("3D Plot: Acceleration, Velocity, and Position")
plt.show()

print(f"Total displacement from t=0 to t=5 is {displacement:.2f} meters.")


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define the function to integrate
def f(x):
    return x**2  # Example: f(x) = x^2

# Define the bounds of the integral
a, b = 0, 2  # Integrate from x=0 to x=2

# Compute the definite integral using scipy
result, _ = quad(f, a, b)

# Generate x values for plotting
x = np.linspace(a - 1, b + 1, 500)
y = f(x)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the function
plt.plot(x, y, label="f(x) = x^2", color='blue')

# Highlight the area under the curve (definite integral)
x_fill = np.linspace(a, b, 500)
y_fill = f(x_fill)
plt.fill_between(x_fill, y_fill, color='skyblue', alpha=0.4, label=f"Area = {result:.2f}")

# Annotate and style the plot
plt.title("Definite Integral: Area Under the Curve", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("f(x)", fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend(fontsize=12)
plt.grid(alpha=0.4)

# Display the plot
plt.show()

print(f"The definite integral of f(x) = x^2 from x={a} to x={b} is {result:.2f}.")



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the function y = x^2
def f(x):
    return x**2

# Interval [a, b] and number of subintervals n
a, b = 0, 2
n = 4

# Calculate width of each subinterval
dx = (b - a) / n

# Generate x values for the subintervals
x_left = np.linspace(a, b - dx, n)  # Left endpoints
x_right = np.linspace(a + dx, b, n)  # Right endpoints

# Calculate the heights of rectangles
heights_left = f(x_left)
heights_right = f(x_right)

# Compute L4 and R4 (areas of rectangles)
L4 = np.sum(heights_left * dx)
R4 = np.sum(heights_right * dx)

# Visualization
x = np.linspace(a, b, 500)
y = f(x)

plt.figure(figsize=(10, 6))

# Plot the curve
plt.plot(x, y, label="$y = x^2$", color="blue")

# Plot left Riemann sum rectangles
for i in range(n):
    plt.bar(x_left[i], heights_left[i], width=dx, align='edge', color='red', alpha=0.4, edgecolor="black", label="Left Riemann Rectangles" if i == 0 else "")

# Plot right Riemann sum rectangles
for i in range(n):
    plt.bar(x_right[i] - dx, heights_right[i], width=dx, align='edge', color='green', alpha=0.4, edgecolor="black", label="Right Riemann Rectangles" if i == 0 else "")

# Labels and legend
plt.title("Visualization of Left and Right Riemann Sums")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()

plt.show()

print(f"L4 (Left Riemann Sum): {L4}")
print(f"R4 (Right Riemann Sum): {R4}")



import sympy as sp

# Define the variables and function
x = sp.symbols('x')
f = x  # f(x) = x
g = sp.exp(-x)  # g(x) = e^(-x)

# Integration by parts formula: ∫f(x)g'(x)dx = f(x)g(x) - ∫f'(x)g(x)dx
f_prime = sp.diff(f, x)  # Derivative of f
g_integral = sp.integrate(g, x)  # Integral of g

# Apply the formula
result = f * g_integral - sp.integrate(f_prime * g_integral, x)

# Simplified result
print("Integration by parts result:")
sp.pprint(result)

# Visualize the function and its integral
import numpy as np
import matplotlib.pyplot as plt

# Numerical integration example
x_vals = np.linspace(0, 5, 100)
y_vals = x_vals * np.exp(-x_vals)
integral_vals = -np.exp(-x_vals) * (x_vals + 1)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="Function: $x e^{-x}$", color="blue")
plt.plot(x_vals, integral_vals, label="Integral of $x e^{-x}$", color="green")
plt.title("Integration by Parts Visualization")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

# Set Seaborn style
sns.set(style="whitegrid")

# Generate example data (you can replace this with your actual data)
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], 'b-', lw=2, label="Sine Wave")

# Customize plot
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Animated Sine Wave", fontsize=16)
ax.set_xlabel("X-axis", fontsize=14)
ax.set_ylabel("Y-axis", fontsize=14)
ax.legend(loc="upper right")

# Initialize the animation
def init():
    line.set_data([], [])
    return line,

# Update function for the animation
def update(frame):
    line.set_data(x[:frame], y[:frame])
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval=50)

# Save or display the animation
# Uncomment the following line to save the animation as a file (e.g., MP4 or GIF)
# ani.save("sine_wave_animation.mp4", writer="ffmpeg")

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the functions u(x) and v(x)
def u(x):
    return np.exp(-x**2)

def v(x):
    return x

# Derivative of u(x) and v(x)
def du(x):
    return -2*x * np.exp(-x**2)

def dv(x):
    return 1

# Compute the product of u(x) and v(x)
def uv(x):
    return u(x) * v(x)

# Define the integral of v(x) * du(x)
def integral_vdu(x):
    dx = x[1] - x[0]  # Compute the spacing
    return np.cumsum(v(x) * du(x)) * dx  # Numerical integration (trapezoidal)

# Set up the figure and axis
x = np.linspace(-3, 3, 500)
X, Y = np.meshgrid(x, x)

# Compute the necessary values
Z_uv = uv(X)
Z_integral_vdu = integral_vdu(X)
Z_result = Z_uv - Z_integral_vdu

# Create 3D plot
fig = plt.figure(figsize=(14, 10))

# Plot uv
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(X, Y, Z_uv, cmap='viridis')
ax.set_title(r'$uv$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Plot integral_vdu
ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(X, Y, Z_integral_vdu, cmap='viridis')
ax.set_title(r'$\int v du$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Plot result
ax = fig.add_subplot(133, projection='3d')
ax.plot_surface(X, Y, Z_result, cmap='viridis')
ax.set_title(r'$uv - \int v du$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the functions u(x) and v(x)
def u(x):
    return np.exp(-x**2)

def v(x):
    return x

# Derivative of u(x) and v(x)
def du(x):
    return -2*x * np.exp(-x**2)

def dv(x):
    return 1

# Compute the product of u(x) and v(x)
def uv(x):
    return u(x) * v(x)

# Define the integral of v(x) * du(x)
def integral_vdu(x):
    return np.cumsum(v(x) * du(x)) * (x[1] - x[0])  # Numerical integration (trapezoidal)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(-3, 3, 500)
ax.set_xlim(-3, 3)
ax.set_ylim(-1, 1)

line_uv, = ax.plot([], [], label=r'$uv$', color='red')
line_integral_vdu, = ax.plot([], [], label=r'$\int v du$', color='blue')
line_result, = ax.plot([], [], label=r'$uv - \int v du$', color='green')

# Adding labels and title
ax.set_title("Animation of Integration by Parts")
ax.legend(loc="upper right")

# Initialize the plot
def init():
    line_uv.set_data([], [])
    line_integral_vdu.set_data([], [])
    line_result.set_data([], [])
    return line_uv, line_integral_vdu, line_result

# Update function for the animation
def update(frame):
    # Limit the x values for the current frame
    x_frame = x[:frame]
    
    # Update the data for u(x), v(x), and the integrals
    line_uv.set_data(x_frame, uv(x_frame))
    line_integral_vdu.set_data(x_frame, integral_vdu(x_frame))
    
    # Combine uv and integral_vdu to get the result
    result = uv(x_frame) - integral_vdu(x_frame)
    line_result.set_data(x_frame, result)
    
    return line_uv, line_integral_vdu, line_result


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=range(1, len(x)), init_func=init, blit=True, interval=30)

# Save the animation
# ani.save('integration_by_parts_animation.mp4', writer='ffmpeg', fps=30)  # Save as .mp4 file
# Alternatively, to save as a GIF
# ani.save('integration_by_parts_animation.gif', writer='imagemagick', fps=30)
plt.show()

