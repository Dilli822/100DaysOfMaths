import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the function we want to observe, e.g., f(x) = (x^2 - 4) / (x - 2)
def f(x):
    # Handling division by zero error when x == 2
    return np.where(x != 2, (x**2 - 4) / (x - 2), np.inf)

# Set up the figure, axis, and plot element
fig, ax = plt.subplots()
ax.set_xlim(1.98, 2.02)
ax.set_ylim(-10, 10)  # Increased the range to observe behavior near infinity
ax.axvline(x=2, color='red', linestyle='--', label='x = 2')
ax.set_title("x Approaching 2 from Left (Blue) and Right (Green)")
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# Initial points on the curve
left_point, = ax.plot([], [], 'bo', label='Approaching from Left')  # Blue dot for left approach
right_point, = ax.plot([], [], 'go', label='Approaching from Right')  # Green dot for right approach
text_left = ax.text(1.98, 7, '', fontsize=10, color='blue')
text_right = ax.text(1.98, 6, '', fontsize=10, color='green')

# Initialize the plot elements
def init():
    left_point.set_data([], [])
    right_point.set_data([], [])
    text_left.set_text('')
    text_right.set_text('')
    return left_point, right_point, text_left, text_right

# Animation update function
def update(frame):
    # Left side approach (Blue dot)
    x_left = 2 - 10**(-frame / 10)  # e.g., 1.9, 1.99, 1.999...
    y_left = f(x_left)

    # Right side approach (Green dot)
    x_right = 2 + 10**(-frame / 10)  # e.g., 2.1, 2.01, 2.001...
    y_right = f(x_right)

    # Update the points and text
    left_point.set_data([x_left], [y_left])
    right_point.set_data([x_right], [y_right])
    text_left.set_text(f"x (left) = {x_left:.10f}, f(x) = {y_left:.3f}")
    text_right.set_text(f"x (right) = {x_right:.10f}, f(x) = {y_right:.3f}")

    return left_point, right_point, text_left, text_right

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 100), init_func=init,
                              blit=True, interval=100)

# Add legend
ax.legend()

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the radius of the circle
r = 1  # For simplicity, set the radius of the circle to 1

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal', 'box')
ax.set_title('Polygon Approaching Circle Area')

# Circle plot
circle = plt.Circle((0, 0), r, color='blue', fill=False, linestyle='-', linewidth=2, label="Circle")
ax.add_artist(circle)

# Polygon plot (this will be updated in the animation)
polygon, = ax.plot([], [], 'go-', label="Inscribed Polygon")

# Text for area
text_area = ax.text(0.1, 1.1, '', fontsize=12)

# Function to calculate the area of the inscribed polygon
def polygon_area(n):
    return (n / 2) * r**2 * np.sin(2 * np.pi / n)

# Initialize the plot elements
def init():
    polygon.set_data([], [])
    text_area.set_text('')
    return polygon, text_area

# Animation update function
def update(frame):
    # Number of sides of the polygon (up to 20)
    n = frame + 3  # Starting with a triangle (n=3)
    
    # Calculate the polygon vertices
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    # Update the polygon
    polygon.set_data(np.append(x, x[0]), np.append(y, y[0]))  # Close the polygon
    
    # Calculate and display the area of the polygon
    poly_area = polygon_area(n)
    text_area.set_text(f"Area of Polygon (n={n}): {poly_area:.4f}\nArea of Circle: {np.pi * r**2:.4f}")
    
    return polygon, text_area

# Create the animation (limit to 20 sides)
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 18), init_func=init,
                              blit=True, interval=1000)  # 18 + 3 = 21 sides max

# Add legend
ax.legend()

plt.show()
