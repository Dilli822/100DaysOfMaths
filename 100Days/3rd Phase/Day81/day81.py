import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the functions and their derivatives
def f_diff(x):
    return x**2  # Differentiable function

def df_diff(x):
    return 2*x  # Derivative of x^2

def f_nondiff(x):
    return np.abs(x)  # Non-differentiable function

def df_nondiff(x):
    return np.sign(x)  # This derivative is undefined at x=0

# Create figure
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

x = np.linspace(-2, 2, 400)
y_diff = f_diff(x)
y_nondiff = f_nondiff(x)

# Plot both functions
ax[0].plot(x, y_diff, label=r"$f(x) = x^2$")
ax[1].plot(x, y_nondiff, label=r"$f(x) = |x|$", color="orange")

# Initialize lines for tangents
tangent_diff, = ax[0].plot([], [], 'r-', lw=2, label="Tangent Line")
tangent_nondiff, = ax[1].plot([], [], 'r-', lw=2, label="Tangent Line")

# Titles and Legends
ax[0].set_title("Differentiable Function")
ax[1].set_title("Non-Differentiable Function")
ax[0].legend()
ax[1].legend()

def init():
    tangent_diff.set_data([], [])
    tangent_nondiff.set_data([], [])
    return tangent_diff, tangent_nondiff

def animate(i):
    x0 = -2 + i * 0.1  # Moving point

    # Differentiable case
    y0_diff = f_diff(x0)
    slope_diff = df_diff(x0)
    x_line = np.linspace(x0 - 0.5, x0 + 0.5, 10)
    y_line_diff = y0_diff + slope_diff * (x_line - x0)
    tangent_diff.set_data(x_line, y_line_diff)

    # Non-Differentiable case
    y0_nondiff = f_nondiff(x0)
    slope_nondiff = df_nondiff(x0)
    y_line_nondiff = y0_nondiff + slope_nondiff * (x_line - x0)
    tangent_nondiff.set_data(x_line, y_line_nondiff)

    return tangent_diff, tangent_nondiff

# Animate
ani = animation.FuncAnimation(fig, animate, frames=40, init_func=init, blit=True, interval=100)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Functions
def exp_growth(x):
    return np.exp(x)

def exp_decay(x):
    return np.exp(-x)

# Generate x values
x = np.linspace(-2, 2, 100)

# Initialize plot
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(0, 8)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Exponential Growth & Decay")

# Plot initial function
growth_line, = ax.plot([], [], 'g-', label="e^x (Growth)")
decay_line, = ax.plot([], [], 'r-', label="e^(-x) (Decay)")

# Animation function
def update(frame):
    growth_line.set_data(x[:frame], exp_growth(x[:frame]))
    decay_line.set_data(x[:frame], exp_decay(x[:frame]))
    return growth_line, decay_line

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(x), interval=50, blit=True)

plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function definitions
def log_func(x):
    return np.log(x)

def derivative_log(x):
    return 1 / x

# Generate x values (avoiding zero)
x = np.linspace(0.1, 5, 100)

# Initialize plot
fig, ax = plt.subplots()
ax.set_xlim(0, 5)
ax.set_ylim(-1, 2)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Logarithm Function and Its Derivative")

# Plot initial functions
log_line, = ax.plot([], [], 'b-', label="ln(x)")
deriv_line, = ax.plot([], [], 'r--', label="d/dx ln(x) = 1/x")

# Animation function
def update(frame):
    log_line.set_data(x[:frame], log_func(x[:frame]))
    deriv_line.set_data(x[:frame], derivative_log(x[:frame]))
    return log_line, deriv_line

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(x), interval=50, blit=True)

plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define functions
f = lambda x: np.sin(x)
g = lambda x: np.cos(x)
fp = lambda x: np.cos(x)  # Derivative of f
gp = lambda x: -np.sin(x) # Derivative of g

def animate_sum_rule():
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y1, y2 = f(x), g(x)
    y_sum = y1 + y2
    y_sum_derivative = fp(x) + gp(x)
    
    fig, ax = plt.subplots()
    ax.set_xlim(-2*np.pi, 2*np.pi)
    ax.set_ylim(-2, 2)
    
    line1, = ax.plot([], [], 'r', label='f(x) + g(x)')
    line2, = ax.plot([], [], 'b--', label="(f' + g')(x)")
    
    def update(frame):
        line1.set_data(x[:frame], y_sum[:frame])
        line2.set_data(x[:frame], y_sum_derivative[:frame])
        return line1, line2
    
    ani = animation.FuncAnimation(fig, update, frames=len(x), interval=50, blit=True)
    plt.title("Sum Rule: (f(x) + g(x))'")
    plt.legend()
    plt.show()

def animate_product_rule():
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y_product = f(x) * g(x)
    y_product_derivative = fp(x) * g(x) + f(x) * gp(x)
    
    fig, ax = plt.subplots()
    ax.set_xlim(-2*np.pi, 2*np.pi)
    ax.set_ylim(-2, 2)
    
    line1, = ax.plot([], [], 'r', label='f(x) * g(x)')
    line2, = ax.plot([], [], 'b--', label="(f'g + fg')(x)")
    
    def update(frame):
        line1.set_data(x[:frame], y_product[:frame])
        line2.set_data(x[:frame], y_product_derivative[:frame])
        return line1, line2
    
    ani = animation.FuncAnimation(fig, update, frames=len(x), interval=50, blit=True)
    plt.title("Product Rule: (f(x) * g(x))'")
    plt.legend()
    plt.show()

def animate_chain_rule():
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    outer_func = lambda u: np.exp(u)
    outer_derivative = lambda u: np.exp(u)
    y_chain = outer_func(f(x))
    y_chain_derivative = outer_derivative(f(x)) * fp(x)
    
    fig, ax = plt.subplots()
    ax.set_xlim(-2*np.pi, 2*np.pi)
    ax.set_ylim(0, np.exp(1))
    
    line1, = ax.plot([], [], 'r', label='e^(f(x))')
    line2, = ax.plot([], [], 'b--', label="(e^(f(x)) * f'(x))")
    
    def update(frame):
        line1.set_data(x[:frame], y_chain[:frame])
        line2.set_data(x[:frame], y_chain_derivative[:frame])
        return line1, line2
    
    ani = animation.FuncAnimation(fig, update, frames=len(x), interval=50, blit=True)
    plt.title("Chain Rule: (e^(f(x)))'")
    plt.legend()
    plt.show()

# Run animations separately
animate_sum_rule()
animate_product_rule()
animate_chain_rule()
