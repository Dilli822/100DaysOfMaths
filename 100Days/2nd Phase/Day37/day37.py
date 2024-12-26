
from sympy import symbols, diff

# Define variables
x, y = symbols('x y')

# Define a function
f = x**2 * y**3

# First-order partial derivatives
f_x = diff(f, x)
f_y = diff(f, y)

# Second-order partial derivatives
f_xx = diff(f_x, x)
f_yy = diff(f_y, y)
f_xy = diff(f_x, y)
f_yx = diff(f_y, x)

print("f_xx:", f_xx)
print("f_yy:", f_yy)
print("f_xy:", f_xy)
print("f_yx:", f_yx)


from sympy import Eq, simplify

# Define a homogeneous function
f_h = x**2 + y**2

# Verify Euler's Theorem: x * f_x + y * f_y = n * f
lhs = x * diff(f_h, x) + y * diff(f_h, y)
rhs = 2 * f_h  # Degree is 2

print("Euler's Theorem:", simplify(Eq(lhs, rhs)))


# Total differential: df = f_x * dx + f_y * dy
df = f_x * symbols('dx') + f_y * symbols('dy')

print("Total Differential df:", df)


u, v = symbols('u v')

# Define function g(u, v)
g = u**2 + v**3

# u and v as functions of x, y
u_expr = x**2 + y
v_expr = x + y**2

# Chain rule
g_x = diff(g.subs({u: u_expr, v: v_expr}), x)
g_y = diff(g.subs({u: u_expr, v: v_expr}), y)

print("g_x:", g_x)
print("g_y:", g_y)


z, t = symbols('z t')

# Define a function h(x, y, z)
h = x*y + z*t

# z and t are functions of x, y
z_expr = x**2 + y
t_expr = y**2 - x

# Chain rule for h
h_x = diff(h.subs({z: z_expr, t: t_expr}), x)
h_y = diff(h.subs({z: z_expr, t: t_expr}), y)

print("h_x:", h_x)
print("h_y:", h_y)


# For Two Variables
g_chain = diff(g, u) * diff(u_expr, x) + diff(g, v) * diff(v_expr, x)
print("Chain Rule (2 variables):", g_chain)

# For Three Variables
z_expr_3 = x + y
t_expr_3 = x*y
h_chain = diff(h, x) + diff(h, y) + diff(h, z)
print("Chain Rule (3 variables):", h_chain)


import matplotlib.pyplot as plt
import numpy as np

# Define function
def f(x, y):
    return x**2 * y**3

# Create mesh grid
X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
Z = f(X, Y)

# Plot
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(cp)
plt.title('Second-Order Partial Derivative Visualization')
plt.xlabel('x')
plt.ylabel('y')
plt.show()





import matplotlib.pyplot as plt
import numpy as np

# Define the function
def f(x, y):
    return x**2 * y**3

# Define partial derivatives
def f_xx(x, y):
    return 2 * y**3

def f_yy(x, y):
    return 6 * x**2 * y

def f_xy(x, y):
    return 6 * x * y**2

# Create mesh grid
X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))

# Compute function and derivatives
Z = f(X, Y)
Z_xx = f_xx(X, Y)
Z_yy = f_yy(X, Y)
Z_xy = f_xy(X, Y)

# 2D Contour Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

titles = ["Function f(x, y)", "Second Partial f_xx", "Second Partial f_yy", "Mixed Partial f_xy"]
data = [Z, Z_xx, Z_yy, Z_xy]

for ax, title, Z_data in zip(axes, titles, data):
    cp = ax.contourf(X, Y, Z_data, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(cp, ax=ax)

plt.tight_layout()
plt.show()

# 3D Surface Plots
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(18, 12))

# Plotting the surfaces
for i, (title, Z_data) in enumerate(zip(titles, data)):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    surf = ax.plot_surface(X, Y, Z_data, cmap='viridis', edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Value')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
