import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the first-order ODE
def ode1(y, t):
    return 4 - 2*y  # dy/dt = 4 - 2y

# Time points where solution is computed
t = np.linspace(0, 10, 100)

# Initial condition
y0 = 0

# Solve ODE
y = odeint(ode1, y0, t)

# Plot the solution
plt.plot(t, y, label="y(t)")
plt.title("Solution of First-Order Linear ODE: dy/dt + 2y = 4")
plt.xlabel("Time (t)")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.show()



# Define the second-order ODE as a system of first-order ODEs
def ode2(y, t):
    dy1 = y[1]  # dy1/dt = y2
    dy2 = -3*y[1] - 2*y[0]  # dy2/dt = -3y2 - 2y1
    return [dy1, dy2]

# Time points
t = np.linspace(0, 10, 100)

# Initial conditions
y0 = [0, 1]  # y(0) = 0, y'(0) = 1

# Solve the system
sol = odeint(ode2, y0, t)

# Plot the solution
plt.plot(t, sol[:, 0], label="y(t)")
plt.title("Solution of Second-Order Linear ODE: d²y/dt² + 3dy/dt + 2y = 0")
plt.xlabel("Time (t)")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.show()


# Define the variable separable ODE
def ode_separable(y, x):
    return y/x  # dy/dx = y/x

# Time points
x = np.linspace(1, 10, 100)

# Initial condition
y0 = 1

# Solve the ODE
y_separable = odeint(ode_separable, y0, x)

# Plot the solution
plt.plot(x, y_separable, label="y(x)")
plt.title("Solution of Variable Separable ODE: dy/dx = y/x")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True)
plt.show()


# Define the exact ODE
def ode_exact(y, x):
    return x/y  # dy/dx = x/y

# Time points
x = np.linspace(1, 10, 100)

# Initial condition
y0 = 1

# Solve the ODE
y_exact = odeint(ode_exact, y0, x)

# Plot the solution
plt.plot(x, y_exact, label="y(x)")
plt.title("Solution of Exact Differential Equation: dy/dx = x/y")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True)
plt.show()



# Infinite Sequence Examples

# Bounded below (example: y_n = 1/n, bounded below by 0)
n = np.arange(1, 100)
y_bounded_below = 1 / n

# Unbounded sequence (example: y_n = n)
y_unbounded = n

# Bounded above (example: y_n = sin(n), bounded above by 1)
y_bounded_above = np.sin(n)

# Constant sequence (example: y_n = 5)
y_constant = np.full_like(n, 5)

# Plotting all sequences
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(n, y_bounded_below, label="Bounded Below (1/n)")
plt.title("Bounded Below")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(n, y_unbounded, label="Unbounded (n)", color='red')
plt.title("Unbounded")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(n, y_bounded_above, label="Bounded Above (sin(n))", color='green')
plt.title("Bounded Above")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(n, y_constant, label="Constant (5)", color='purple')
plt.title("Constant Sequence")
plt.legend()

plt.tight_layout()
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 1. First-Order Linear ODE: dy/dt = 4 - 2y

def ode1(y, t):
    return 4 - 2*y  # dy/dt = 4 - 2y

t = np.linspace(0, 10, 100)
y0 = 0
y = odeint(ode1, y0, t)

# 3D Plot: First-Order Linear ODE
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(t, y, zs=0, zdir='z', label='First-Order Linear ODE: dy/dt + 2y = 4')
ax.set_xlabel('Time (t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z')
plt.title("3D Plot of First-Order ODE Solution")
plt.show()

# 2. Second-Order ODE: d²y/dt² + 3dy/dt + 2y = 0

def ode2(y, t):
    dy1 = y[1]  # dy1/dt = y2
    dy2 = -3*y[1] - 2*y[0]  # dy2/dt = -3y2 - 2y1
    return [dy1, dy2]

t = np.linspace(0, 10, 100)
y0 = [0, 1]  # y(0) = 0, y'(0) = 1
sol = odeint(ode2, y0, t)

# 3D Plot: Second-Order Linear ODE
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(t, sol[:, 0], zs=0, zdir='z', label='Second-Order Linear ODE: d²y/dt² + 3dy/dt + 2y = 0')
ax.set_xlabel('Time (t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z')
plt.title("3D Plot of Second-Order ODE Solution")
plt.show()

# 3. Variable Separable ODE: dy/dx = y/x

def ode_separable(y, x):
    return y/x  # dy/dx = y/x

x = np.linspace(1, 10, 100)
y0 = 1
y_separable = odeint(ode_separable, y0, x)

# 3D Plot: Variable Separable ODE
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y_separable, zs=0, zdir='z', label='Variable Separable ODE: dy/dx = y/x')
ax.set_xlabel('x')
ax.set_ylabel('y(x)')
ax.set_zlabel('z')
plt.title("3D Plot of Variable Separable ODE Solution")
plt.show()

# Animation for Variable Separable ODE

fig, ax = plt.subplots()
line, = ax.plot([], [], label='dy/dx = y/x')

def init():
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y(x)')
    ax.legend()
    return line,

def animate(i):
    x_vals = x[:i]
    y_vals = y_separable[:i]
    line.set_data(x_vals, y_vals)
    return line,

ani = FuncAnimation(fig, animate, frames=len(x), init_func=init, blit=True)
plt.title("Animation for Variable Separable ODE")
plt.show()

# Infinite Sequence Examples (Animation)

# Bounded below (example: y_n = 1/n, bounded below by 0)
n = np.arange(1, 100)
y_bounded_below = 1 / n
y_unbounded = n
y_bounded_above = np.sin(n)
y_constant = np.full_like(n, 5)

# Animation for Infinite Sequence
fig, ax = plt.subplots()
line, = ax.plot([], [], label="Bounded Below (1/n)")

def init_sequence():
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('n')
    ax.set_ylabel('y_n')
    ax.legend()
    return line,

def animate_sequence(i):
    n_vals = n[:i]
    y_vals = y_bounded_below[:i]
    line.set_data(n_vals, y_vals)
    return line,

ani_seq = FuncAnimation(fig, animate_sequence, frames=len(n), init_func=init_sequence, blit=True)
plt.title("Animation for Bounded Below Sequence (1/n)")
plt.show()
