import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)  # Focusing on positive x-axis
ax.set_ylim(0, 10)  # Limiting the y-axis for better visualization
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# Line for animation
line, = ax.plot([], [], color='red', label='y = 1/x^2')
text_value = ax.text(1, 8, '', fontsize=12, color='blue')

# Variables
x = np.linspace(0.1, 10, 500)  # Starting slightly above 0 to avoid division by zero

# Initialize animation
def init():
    line.set_data([], [])
    text_value.set_text('')
    return line, text_value

# Update function for y = 1/x^2
def update(frame):
    ax.set_title('Animation of y = 1/x^2 as x increases')
    
    # Incrementally show more of x
    current_x = x[:frame]
    y = 1 / current_x**2
    
    # Update line data
    line.set_data(current_x, y)
    
    # Get the current x and y values to display
    current_x_value = current_x[-1]
    current_y_value = y[-1]
    
    # Set the text value on the plot
    text_value.set_text(f'x = {current_x_value:.2f}, y = {current_y_value:.2f}')
    text_value.set_position((current_x_value, current_y_value))
    
    return line, text_value

# Animation setup
ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, repeat=False)

# Legend and labels
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0.01, 1)  # Setting x-axis range close to zero but positive
ax.set_ylim(0, 100)   # Setting y-axis range to see the infinity trend
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# Line for animation
line, = ax.plot([], [], color='blue', label='y = 1/x')
text_value = ax.text(0.2, 80, '', fontsize=12, color='red')

# Variables
x = np.linspace(0.01, 1, 500)  # x values getting closer to zero from positive side

# Initialize animation
def init():
    line.set_data([], [])
    text_value.set_text('')
    return line, text_value

# Update function for y = 1/x as x -> 0+
def update(frame):
    ax.set_title('Animation of y = 1/x as x approaches 0 from positive side')
    
    # Subset of x for current frame to show increasing portions gradually
    current_x = x[:frame]
    y = 1 / current_x
    
    # Plot data update
    line.set_data(current_x, y)
    
    # Display current value of y at x
    display_x = current_x[-1]
    display_y = y[-1]

    # Set text with current values
    text_value.set_text(f'At x={display_x:.3f}, y={display_y:.2f}')
    text_value.set_position((display_x, min(display_y, 90)))  # Limit text position to stay within plot
    
    return line, text_value

# Animation setup with faster speed
ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, repeat=False, interval=500)

# Legend and labels
plt.legend()
plt.xlabel('x (approaching 0)')
plt.ylabel('y = 1/x')
plt.grid()
plt.show()
