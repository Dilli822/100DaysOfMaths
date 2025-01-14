import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function and its derivative
def f(x):
    return x**2

def df(x):
    return 2*x

x = np.linspace(-10, 10, 1000)
y = f(x)

# Plot the function
fig, ax = plt.subplots()
ax.plot(x, y, label='$f(x) = x^2$', color='b')
tangent_line, = ax.plot([], [], label='Tangent Line', color='r')

# Animation function
def update(frame):
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 100)
    
    # Calculate tangent at point frame
    x_tangent = frame
    y_tangent = f(x_tangent)
    slope = df(x_tangent)
    
    # Tangent line equation: y = mx + c
    tangent = slope * (x - x_tangent) + y_tangent
    tangent_line.set_data(x, tangent)
    return tangent_line,

# Create the animation
ani = FuncAnimation(fig, update, frames=np.linspace(-10, 10, 100), interval=100, blit=True)
plt.legend()
plt.title("1. Geometrical Interpretation of Derivative of a Function")
plt.show()


# Function for f(x) = x^3 - 3x
def f2(x):
    return x**3 - 3*x

def df2(x):
    return 3*x**2 - 3

x = np.linspace(-3, 3, 1000)
y = f2(x)
dy = df2(x)

fig, ax = plt.subplots()
ax.plot(x, y, label='$f(x) = x^3 - 3x$', color='b')

# Animation function
def update2(frame):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-10, 10)
    
    # Highlight intervals where derivative is positive or negative
    ax.fill_between(x, y, where=(dy >= 0), color='g', alpha=0.3, label='Increasing (dy > 0)')
    ax.fill_between(x, y, where=(dy < 0), color='r', alpha=0.3, label='Decreasing (dy < 0)')
    return []

ani2 = FuncAnimation(fig, update2, frames=np.linspace(-3, 3, 100), interval=100, blit=True)
plt.legend()
plt.title("2. Increasing and Decreasing Functions")
plt.show()


# Function for f(x) = sin(x) * x
def f3(x):
    return np.sin(x) * x

def df3(x):
    return np.cos(x) * x + np.sin(x)

x = np.linspace(-10, 10, 1000)
y = f3(x)
dy = df3(x)

fig, ax = plt.subplots()
ax.plot(x, y, label='$f(x) = sin(x) * x$', color='b')

# Animation function
def update3(frame):
    ax.set_xlim(-10, 10)
    ax.set_ylim(-15, 15)
    
    # Find and highlight maxima and minima
    critical_points = x[np.isclose(dy, 0, atol=0.1)]  # where derivative is near 0
    ax.scatter(critical_points, f3(critical_points), color='r', label='Critical Points')
    return []

ani3 = FuncAnimation(fig, update3, frames=np.linspace(-10, 10, 100), interval=100, blit=True)
plt.legend()
plt.title("3. Maxima and Minima")
plt.show()


# Function for f(x) = x^4 - 6x^2
def f4(x):
    return x**4 - 6*x**2

def df4(x):
    return 4*x**3 - 12*x

x = np.linspace(-3, 3, 1000)
y = f4(x)

fig, ax = plt.subplots()
ax.plot(x, y, label='$f(x) = x^4 - 6x^2$', color='b')

# Find the critical points and endpoints
critical_points = x[np.isclose(df4(x), 0, atol=0.1)]
endpoints = np.array([x[0], x[-1]])

# Animation function
def update4(frame):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-10, 10)
    
    # Clear the previous frame to avoid overlapping points
    ax.clear()
    ax.plot(x, y, label='$f(x) = x^4 - 6x^2$', color='b')

    # Highlight absolute maxima and minima
    ax.scatter(critical_points, f4(critical_points), color='r', label='Critical Points')
    ax.scatter(endpoints, f4(endpoints), color='g', label='Endpoints')
    
    return []

ani4 = FuncAnimation(fig, update4, frames=np.linspace(-3, 3, 100), interval=100, blit=False)
plt.legend()
plt.title("4. Absolute Maxima and Minima")
plt.show()


# Function for f(x) = log(x)
def f5(x):
    return np.log(x)

def df5(x):
    return 1/x

x = np.linspace(0.1, 10, 1000)
y = f5(x)
dy = df5(x)

fig, ax = plt.subplots()
ax.plot(x, y, label='$f(x) = log(x)$', color='b')

# Find stationary points
stationary_points = x[np.isclose(dy, 0, atol=0.1)]

# Animation function
def update5(frame):
    ax.set_xlim(0.1, 10)
    ax.set_ylim(-3, 3)
    
    # Highlight stationary points
    ax.scatter(stationary_points, f5(stationary_points), color='r', label='Stationary Points')
    return []

ani5 = FuncAnimation(fig, update5, frames=np.linspace(0.1, 10, 100), interval=100, blit=True)
plt.legend()
plt.title("5. Stationary Point")
plt.show()


# Function for f(x) = sin(x)
def f6(x):
    return np.sin(x)

def d2f6(x):
    return -np.sin(x)

x = np.linspace(-10, 10, 1000)
y = f6(x)
d2y = d2f6(x)

fig, ax = plt.subplots()
ax.plot(x, y, label='$f(x) = sin(x)$', color='b')

# Animation function
def update6(frame):
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 2)
    
    # Highlight concave and convex regions
    ax.fill_between(x, y, where=(d2y >= 0), color='g', alpha=0.3, label='Concave Up (d2f > 0)')
    ax.fill_between(x, y, where=(d2y < 0), color='r', alpha=0.3, label='Concave Down (d2f < 0)')
    return []

ani6 = FuncAnimation(fig, update6, frames=np.linspace(-10, 10, 100), interval=100, blit=True)
plt.legend()
plt.title("6. Concavity and Convexity of Curves")
plt.show()


# Function for f(x) = sin(x) * x
def f7(x):
    return np.sin(x) * x

def df7(x):
    return np.cos(x) * x + np.sin(x)

def d2f7(x):
    return -np.sin(x) * x + 2 * np.cos(x)

x = np.linspace(-10, 10, 1000)
y = f7(x)
dy = df7(x)
d2y = d2f7(x)

fig, ax = plt.subplots()
ax.plot(x, y, label='$f(x) = sin(x) * x$', color='b')

# Find inflection points (where d2f(x) = 0)
inflection_points = x[np.isclose(d2y, 0, atol=0.1)]

# Animation function
def update7(frame):
    ax.set_xlim(-10, 10)
    ax.set_ylim(-15, 15)
    
    # Highlight points of inflection
    ax.scatter(inflection_points, f7(inflection_points), color='r', label='Inflection Points')
    return []

ani7 = FuncAnimation(fig, update7, frames=np.linspace(-10, 10, 100), interval=100, blit=True)
plt.legend()
plt.title("7. Point of Inflection, Critical Points, Rate of Measure")
plt.show()
