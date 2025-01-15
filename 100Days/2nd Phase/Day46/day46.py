import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# Define the function
def f(x):
    return x**2

# Interval [0, 2] with 4 subintervals
a, b = 0, 2
n = 4
x = np.linspace(a, b, n+1)
y = f(x)

# Trapezoidal approximation
x_full = np.linspace(a, b, 500)
y_full = f(x_full)
trapezoid_x = np.array([[x[i], x[i+1]] for i in range(len(x)-1)]).flatten()
trapezoid_y = np.array([[y[i], y[i+1]] for i in range(len(y)-1)]).flatten()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_full, y_full, label='y = x^2', color='blue')
plt.fill_between(x, y, alpha=0.3, label='Trapezoidal Area', color='orange')
for i in range(n):
    plt.plot(trapezoid_x[2*i:2*i+2], trapezoid_y[2*i:2*i+2], 'r--')
plt.title('2D Plot: Trapezoidal Rule Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function and parameters
def f(x):
    return x ** 2

a, b = 0, 2  # Integration limits
n = 10  # Number of trapezoids

# Create animation frames
x_vals = np.linspace(a, b, 100)
y_vals = f(x_vals)

fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], 'b', label="y = x^2")
text = ax.text(1.5, 2.5, "", fontsize=12)

# Setting up the axis limits
ax.set_xlim(a, b)
ax.set_ylim(0, max(f(x_vals)) + 1)
ax.set_title("Animated Trapezoidal Approximation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

# Function to update frames
def update(frame):
    current_x = np.linspace(a, a + frame * (b - a) / n, frame + 1)
    current_y = f(current_x)
    line.set_data(current_x, current_y)
    
    # Remove existing collections (clear previous fills)
    for coll in ax.collections:
        coll.remove()
    
    # Re-fill the area for animation
    ax.fill_between(current_x, current_y, alpha=0.3, color='orange', label="Trapezoidal Area")
    text.set_text(f"Step: {frame}")
    return line, text

ani = FuncAnimation(fig, update, frames=n + 1, interval=500, blit=False)
plt.show()



# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the function and parameters
def f(x):
    return x ** 2

a, b = 0, 2  # Integration limits
n = 10  # Number of trapezoids

# Create animation frames
x_vals = np.linspace(a, b, 100)
y_vals = f(x_vals)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Trapezoidal Approximation of y = x^2')

# Set the view angle
ax.view_init(elev=30, azim=60)

# Function to update frames for animation
def update(frame):
    ax.cla()  # Clear the previous plot

    # Redefine axis and limits
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Trapezoidal Approximation of y = x^2')

    # Redefine the trapezoidal approximation for the current frame
    x = np.linspace(a, a + frame * (b - a) / n, frame + 1)
    y = f(x)
    z = np.zeros_like(x)  # The Z-coordinate for the flat surface

    # Plot the function as a 3D curve
    ax.plot(x, y, z, label="Curve y = x^2", color='blue')

    # Plot trapezoids in 3D
    for i in range(len(x) - 1):
        x_trap = [x[i], x[i + 1], x[i + 1], x[i]]
        y_trap = [y[i], y[i + 1], 0, 0]  # Corresponding y-values for trapezoidal sides
        z_trap = [0, 0, 0, 0]  # Z = 0 for the bottom surface of trapezoid
        ax.plot_trisurf(x_trap, y_trap, z_trap, color='orange', alpha=0.5)

    ax.view_init(elev=30, azim=60)  # Keep the same view angle
    return ax,

ani = FuncAnimation(fig, update, frames=n + 1, interval=500, blit=False)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Function and its derivative
def f(x):
    return (4 * np.sqrt(2) / 3) * x**(3/2) - 1

def f_prime(x):
    return 2 * np.sqrt(2) * x**(1/2)

# Arc length formula: L = integral from a to b of sqrt(1 + (f'(x))^2) dx
def arc_length_integral(a, b):
    integrand = lambda x: np.sqrt(1 + (f_prime(x))**2)
    length, error = quad(integrand, a, b)
    return length

# 2D plot of the function y = f(x)
x = np.linspace(0, 1, 1000)
y = f(x)

fig, ax = plt.subplots()
ax.plot(x, y, label="y = (4√2 / 3) * x^(3/2) - 1", color="b")
ax.set_title("2D Plot of y = (4√2 / 3) * x^(3/2) - 1")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.grid(True)
plt.show()

# 3D plot (Visualizing the curve as a 3D function)
x = np.linspace(0, 1, 100)
y = f(x)
z = np.zeros_like(x)  # z-axis set to 0 to show the 2D curve on the XY plane

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label="y = (4√2 / 3) * x^(3/2) - 1")
ax.set_title("3D Plot of the Curve")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()

# Animation of arc length (accumulating arc length over time)
fig, ax = plt.subplots()
line, = ax.plot([], [], color='b')
arc_length_text = ax.text(0.5, 0.9, '', transform=ax.transAxes)

def init():
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 2)
    arc_length_text.set_text('')
    return line, arc_length_text

def update(frame):
    x_vals = np.linspace(0, frame / 100, 100)
    y_vals = f(x_vals)
    line.set_data(x_vals, y_vals)
    
    # Calculate the arc length from 0 to the current x value
    arc_length = arc_length_integral(0, frame / 100)
    arc_length_text.set_text(f"Arc Length: {arc_length:.4f}")
    
    return line, arc_length_text

ani = animation.FuncAnimation(fig, update, frames=range(1, 101), init_func=init, blit=True)
plt.show()

# Compute the total arc length for the given interval [0, 1]
total_length = arc_length_integral(0, 1)
print(f"Total Arc Length: {total_length:.4f}")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Define the function y = x^2
def f(x):
    return x**2

# Step 2: Generate data for the plot
x = np.linspace(0, 2, 100)  # x values
y = f(x)  # y values

# Step 3: Create the 2D plot of y = x^2
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x, y, label=r"$y = x^2$", color='b')
ax.fill_between(x, 0, y, alpha=0.3, color='orange')  # Highlight the area
ax.set_title("2D Plot of y = x^2")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.legend()
plt.show()

# Step 4: Create a 3D surface plot by rotating y = x^2 around the y-axis
theta = np.linspace(0, 2 * np.pi, 100)  # Angle for rotation
X, T = np.meshgrid(x, theta)  # Create meshgrid for 3D plot

# Parametric equations for rotation around the y-axis
X_rot = X * np.cos(T)  # X-coordinate after rotation
Z_rot = X * np.sin(T)  # Z-coordinate after rotation
Y_rot = f(X)  # Y remains unchanged

# Step 5: Plot the 3D surface of the volume of revolution
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_rot, Y_rot, Z_rot, cmap='viridis', edgecolor='k', alpha=0.5)
ax.set_title("3D Visualization of Volume of Revolution")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# Step 6: Animate cylindrical shells
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 2)
ax.set_ylim(0, 5)
ax.set_title("Animation of Cylindrical Shells")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Function for drawing cylindrical shells
def update(frame):
    ax.clear()  # Clear previous plots
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 5)
    ax.set_title("Animation of Cylindrical Shells")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Plot the function y = x^2
    ax.plot(x, f(x), label=r"$y = x^2$", color='b')

    # Draw a cylindrical shell at the current x value
    ax.fill_between(x[:frame], 0, f(x[:frame]), alpha=0.3, color='orange')
    
    return ax,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(x), interval=50, blit=False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Define the cardioid and circle equations
def cardioid(theta):
    return 1 - np.cos(theta)

def circle(theta):
    return np.ones_like(theta)  # Ensure this is an array of 1s with the same length as theta

# Area function for integration
def area_function(theta):
    return 0.5 * (circle(theta)**2 - cardioid(theta)**2)

# Compute the area using numerical integration
area, _ = quad(area_function, 0, 2 * np.pi)
print(f"Area of the region: {area:.4f}")

# Step 1: 2D plot of the cardioid and the circle
theta = np.linspace(0, 2 * np.pi, 1000)
r_cardioid = cardioid(theta)
r_circle = circle(theta)

plt.figure(figsize=(6, 6))
plt.subplot(111, projection='polar')
plt.plot(theta, r_cardioid, label="Cardioid: $r = 1 - \cos(\\theta)$", color='r')
plt.plot(theta, r_circle, label="Circle: $r = 1$", color='b')
plt.fill(theta, r_cardioid, alpha=0.3, color='red')
plt.fill(theta, r_circle, alpha=0.1, color='blue')
plt.legend()
plt.title("2D Plot: Circle and Cardioid in Polar Coordinates")
plt.show()

# Step 2: 3D plot of the region between the cardioid and the circle
theta = np.linspace(0, 2 * np.pi, 100)
r = np.linspace(0, 1, 100)

Theta, R = np.meshgrid(theta, r)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
Z = np.zeros_like(X)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=5, cstride=5, color='b', alpha=0.3)
ax.set_title("3D Plot: Volume Between Circle and Cardioid")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# Step 3: Animation of the area between the cardioid and the circle
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_title("Animation of Area Between Cardioid and Circle")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame):
    ax.clear()  # Clear previous plot
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("Animation of Area Between Cardioid and Circle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Plot the circle and cardioid
    theta = np.linspace(0, 2 * np.pi, 1000)
    r_cardioid = cardioid(theta)
    r_circle = circle(theta)

    # Plot the cardioid and circle
    ax.plot(r_cardioid * np.cos(theta), r_cardioid * np.sin(theta), label="Cardioid: $r = 1 - \cos(\\theta)$", color='r')
    ax.plot(r_circle * np.cos(theta), r_circle * np.sin(theta), label="Circle: $r = 1$", color='b')

    # Fill the area between them incrementally
    ax.fill_between(theta[:frame], r_cardioid[:frame], r_circle[:frame], color='purple', alpha=0.3)

    return ax,

ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=False)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Define the parabola and the line
def parabola(x):
    return 2 - x**2

def line(x):
    return -x

# Set up the area function for integration
def area_function(x):
    return parabola(x) - line(x)

# Compute the area using numerical integration
area, _ = quad(area_function, -1, 2)
print(f"Area of the region: {area:.4f}")

# Step 1: 2D plot of the parabola and the line
x = np.linspace(-1.5, 2.5, 1000)
y_parabola = parabola(x)
y_line = line(x)

plt.figure(figsize=(6, 6))
plt.plot(x, y_parabola, label="Parabola: $y = 2 - x^2$", color='r')
plt.plot(x, y_line, label="Line: $y = -x$", color='b')
plt.fill_between(x, y_parabola, y_line, where=(y_parabola > y_line), color='purple', alpha=0.3)
plt.legend()
plt.title("2D Plot: Parabola and Line")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Step 2: 3D plot (for visualization of the enclosed area in a surface)
x = np.linspace(-2, 2, 1000)  # Use 1000 points for better resolution

X, Y = np.meshgrid(x, np.linspace(-1.5, 2.5, 100))
Z_parabola = 2 - X**2
Z_line = -X

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z_parabola, rstride=5, cstride=5, color='r', alpha=0.5)
ax.plot_surface(X, Y, Z_line, rstride=5, cstride=5, color='b', alpha=0.5)

ax.set_title("3D Plot: Parabola and Line")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# Step 3: Animation of the area between the parabola and the line
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_title("Animation of Area Between Parabola and Line")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame):
    ax.clear()  # Clear previous plot
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title("Animation of Area Between Parabola and Line")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Plot the parabola and the line
    ax.plot(x, y_parabola, label="Parabola: $y = 2 - x^2$", color='r')
    ax.plot(x, y_line, label="Line: $y = -x$", color='b')

    # Fill the area between them incrementally
    ax.fill_between(x[:frame], y_parabola[:frame], y_line[:frame], where=(y_parabola[:frame] > y_line[:frame]), color='purple', alpha=0.3)

    return ax,

ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def parabola(x):
    return 2 - x**2

def line(x):
    return -x

# Generate x values
x = np.linspace(-2, 2, 400)
x_fill = np.linspace(-1, 2, 400)  # For shading between the curves

# Calculate y values for the functions
y_parabola = parabola(x)
y_line = line(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y_parabola, label="Parabola: $y = 2 - x^2$", color='r')
plt.plot(x, y_line, label="Line: $y = -x$", color='b')

# Fill the area between the curves
plt.fill_between(x_fill, parabola(x_fill), line(x_fill), color='gray', alpha=0.5)

# Adding labels and title
plt.title('Area between the Parabola and the Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Define the meshgrid for 3D plotting
X = np.linspace(-2, 2, 400)
Y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(X, Y)

# Create the z values (using the parabola and line equations)
Z_parabola = 2 - X**2
Z_line = -X

# Create the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surfaces for the parabola and the line
ax.plot_surface(X, Y, Z_parabola, alpha=0.5, cmap='Reds', label="Parabola")
ax.plot_surface(X, Y, Z_line, alpha=0.5, cmap='Blues', label="Line")

# Set labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

ax.set_title("3D Visualization of Parabola and Line")

plt.show()

import matplotlib.animation as animation

# Define the figure and axis for the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Initial plot for the curves
line_plot, = ax.plot([], [], label="Line: $y = -x$", color='b')
parabola_plot, = ax.plot([], [], label="Parabola: $y = 2 - x^2$", color='r')
fill_area = ax.fill_between([], [], [], color='gray', alpha=0.5)

# Set limits and labels
ax.set_xlim(-1.5, 2)
ax.set_ylim(-3, 3)
ax.set_title("Shading Area between Parabola and Line")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

# Update function for the animation
def update(frame):
    x_frame = np.linspace(-1, frame, 400)
    y_parabola_frame = parabola(x_frame)
    y_line_frame = line(x_frame)

    # Update the line plot and fill the area
    line_plot.set_data(x_frame, y_line_frame)
    parabola_plot.set_data(x_frame, y_parabola_frame)
    fill_area = ax.fill_between(x_frame, y_parabola_frame, y_line_frame, color='gray', alpha=0.5)

    return line_plot, parabola_plot, fill_area

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=np.linspace(-1, 2, 200), interval=50, blit=False)

# Show the animation
plt.show()
