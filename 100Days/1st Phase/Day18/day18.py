import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(-10, 10, 500)

# Define odd and even functions
def odd_function(x):
    return x**3

def even_function(x):
    return x**2

# Piecewise function example
def piecewise_function(x):
    return np.where(x < 0, -x, x**2)

# Calculate y values for each function
y_odd = odd_function(x)
y_even = even_function(x)
y_piecewise = piecewise_function(x)

# Plotting
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Odd function
ax[0].plot(x, y_odd, label="Odd Function: $f(x) = x^3$", color="blue")
ax[0].plot(x, -y_odd, '--', color="red", label="Reflection: $f(-x) = -f(x)$")
ax[0].set_title("Odd Symmetry Function")
ax[0].axhline(0, color='black', linewidth=0.5)
ax[0].axvline(0, color='black', linewidth=0.5)
ax[0].legend()
ax[0].grid()

# Even function
ax[1].plot(x, y_even, label="Even Function: $f(x) = x^2$", color="green")
ax[1].plot(-x, y_even, '--', color="purple", label="Reflection: $f(-x) = f(x)$")
ax[1].set_title("Even Symmetry Function")
ax[1].axhline(0, color='black', linewidth=0.5)
ax[1].axvline(0, color='black', linewidth=0.5)
ax[1].legend()
ax[1].grid()

# Piecewise function
ax[2].plot(x, y_piecewise, label="Piecewise Function:\n$f(x) = -x$ if $x<0$\n$f(x) = x^2$ if $x \geq 0$", color="orange")
ax[2].set_title("Piecewise Function Example")
ax[2].axhline(0, color='black', linewidth=0.5)
ax[2].axvline(0, color='black', linewidth=0.5)
ax[2].legend()
ax[2].grid()

plt.tight_layout()
plt.show()


# Block Diagram: Linear Mathematical Model
fig, ax1 = plt.subplots(figsize=(8, 6))

# Block diagram components
ax1.text(0.2, 0.5, 'Input: Height (h)', fontsize=12, ha='center', bbox=dict(boxstyle="round", fc="lightblue", ec="black"))
ax1.arrow(0.35, 0.5, 0.1, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
ax1.text(0.5, 0.5, 'Linear Model: $T = m \cdot h + b$', fontsize=12, ha='center', bbox=dict(boxstyle="round", fc="lightgreen", ec="black"))
ax1.arrow(0.65, 0.5, 0.1, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
ax1.text(0.8, 0.5, 'Output: Temperature (T)', fontsize=12, ha='center', bbox=dict(boxstyle="round", fc="lightblue", ec="black"))

ax1.axis('off')
ax1.set_title("Linear Mathematical Model Block Diagram", fontsize=16)

# Define the real-world mathematical model
# T = m * h + b
def temperature_at_height(h, m=-0.0065, b=15):
    return m * h + b

# Input (height values in meters)
heights = np.linspace(0, 10000, 500)  # From ground level to 10,000 m

# Process: Calculate temperature using the model
temperatures = temperature_at_height(heights)

# Real-world prediction for specific heights
specific_heights = [0, 2000, 5000, 10000]  # Ground level, 2 km, 5 km, 10 km
predicted_temperatures = [temperature_at_height(h) for h in specific_heights]

# Output: Visualize the relationship and conclusions
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(heights, temperatures, label="Temperature vs Height ($T = -0.0065h + 15$)", color="blue")

# Annotating specific predictions
for h, t in zip(specific_heights, predicted_temperatures):
    ax.scatter(h, t, color="red")
    ax.text(h, t + 1, f"({h} m, {t:.1f}°C)", fontsize=10, ha='center')

# Graph labels and title
ax.set_title("Real-World Prediction: Temperature vs Height", fontsize=14)
ax.set_xlabel("Height (h) [m]", fontsize=12)
ax.set_ylabel("Temperature (T) [°C]", fontsize=12)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
ax.legend()
ax.grid()

plt.show()

# Print predictions for specific heights
for h, t in zip(specific_heights, predicted_temperatures):
    print(f"At height {h} m, predicted temperature: {t:.1f}°C")