import numpy as np
import matplotlib.pyplot as plt

# Define a function
f = lambda x: x**2 - 4*x + 3  # A simple quadratic function

# Define its derivative (velocity)
df = lambda x: 2*x - 4

# Define the tangent line at a given point x=a
def tangent_line(x, a):
    return df(a) * (x - a) + f(a)

# Define a function for the hyperbola
hyperbola = lambda x: 1 / x

# Values for plotting
x = np.linspace(-2, 6, 500)
x_hyperbola = np.linspace(0.1, 2, 500)

# 1. Plot of function, derivative (velocity), and tangent line
fig, ax = plt.subplots(3, 3, figsize=(15, 15))

# Plot f(x) and its tangent at x=2
ax[0, 0].plot(x, f(x), label='f(x) = x^2 - 4x + 3')
ax[0, 0].plot(x, tangent_line(x, 2), '--', label='Tangent at x=2')
ax[0, 0].scatter(2, f(2), color='red', label='Point of Tangency (2, f(2))')
ax[0, 0].set_title('Function and Tangent Line')
ax[0, 0].legend()
ax[0, 0].grid()

# 2. Derivative and Instantaneous Velocity
ax[0, 1].plot(x, df(x), label="f'(x) = 2x - 4", color='orange')
ax[0, 1].scatter(2, df(2), color='red', label='Instantaneous Velocity at x=2')
ax[0, 1].set_title('Derivative and Instantaneous Velocity')
ax[0, 1].legend()
ax[0, 1].grid()

# 3. Rate of Change
rate_x = [1, 4]
rate_y = [f(1), f(4)]
ax[0, 2].plot(x, f(x), label='f(x) = x^2 - 4x + 3')
ax[0, 2].plot(rate_x, rate_y, '--', label='Rate of Change (Secant Line)')
ax[0, 2].scatter(rate_x, rate_y, color='red')
ax[0, 2].set_title('Rate of Change (Secant Line)')
ax[0, 2].legend()
ax[0, 2].grid()

# 4. Rolle's Theorem Example
rolle_f = lambda x: x**3 - 3*x**2 + 2*x
rolle_df = lambda x: 3*x**2 - 6*x + 2
x_rolle = np.linspace(0, 2, 500)
ax[1, 0].plot(x_rolle, rolle_f(x_rolle), label="f(x) = x^3 - 3x^2 + 2x")
ax[1, 0].scatter(1, rolle_f(1), color='red', label='c=1 where f\'=0')
ax[1, 0].set_title("Rolle's Theorem")
ax[1, 0].legend()
ax[1, 0].grid()

# 5. Mean Value Theorem (MVT)
mvt_x = [0, 2]
mvt_y = [rolle_f(0), rolle_f(2)]
secant_slope = (rolle_f(2) - rolle_f(0)) / (2 - 0)
secant_line = lambda x: secant_slope * (x - 0) + rolle_f(0)
ax[1, 1].plot(x_rolle, rolle_f(x_rolle), label="f(x) = x^3 - 3x^2 + 2x")
ax[1, 1].plot(x_rolle, secant_line(x_rolle), '--', label='Secant Line (MVT)')
ax[1, 1].scatter(0, rolle_f(0), color='red')
ax[1, 1].scatter(2, rolle_f(2), color='red')
ax[1, 1].set_title('Mean Value Theorem')
ax[1, 1].legend()
ax[1, 1].grid()

# 6. Extreme Value Theorem (EVT)
x_evt = np.linspace(-1, 2, 500)
ax[1, 2].plot(x_evt, rolle_f(x_evt), label="f(x) = x^3 - 3x^2 + 2x")
ax[1, 2].scatter([0, 2], [rolle_f(0), rolle_f(2)], color='red', label='Extreme Values')
ax[1, 2].set_title('Extreme Value Theorem')
ax[1, 2].legend()
ax[1, 2].grid()

# 7. Hyperbola Plot
ax[2, 0].plot(x_hyperbola, hyperbola(x_hyperbola), label='f(x) = 1/x')
ax[2, 0].set_title('Hyperbola')
ax[2, 0].legend()
ax[2, 0].grid()

# 8. Differentiation Visualization
x_diff = np.linspace(-2, 2, 500)
diff_func = lambda x: np.sin(x)
diff_func_deriv = lambda x: np.cos(x)
ax[2, 1].plot(x_diff, diff_func(x_diff), label='f(x) = sin(x)')
ax[2, 1].plot(x_diff, diff_func_deriv(x_diff), label="f'(x) = cos(x)")
ax[2, 1].set_title('Differentiation Example')
ax[2, 1].legend()
ax[2, 1].grid()

# 9. Instantaneous Rate of Change
instant_x = np.linspace(-1, 1, 500)
instant_f = lambda x: x**3
instant_df = lambda x: 3*x**2
ax[2, 2].plot(instant_x, instant_f(instant_x), label='f(x) = x^3')
ax[2, 2].scatter(1, instant_df(1), color='red', label='Instantaneous Rate at x=1')
ax[2, 2].set_title('Instantaneous Rate of Change')
ax[2, 2].legend()
ax[2, 2].grid()

plt.tight_layout()
plt.show()


# Define helper functions for calculus concepts
# 1. Gradient Descent Visualization
f = lambda x: x**2  # Quadratic function
f_grad = lambda x: 2*x  # Gradient
x_vals = np.linspace(-10, 10, 500)
x_points = [8, 6, 4, 2, 1, 0.5, 0.1]  # Simulated gradient descent steps
y_points = [f(x) for x in x_points]

# 2. Partial Derivatives Visualization
partial_f = lambda x, y: x**2 + y**2
x_partial, y_partial = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
z_partial = partial_f(x_partial, y_partial)

# 3. Taylor Series Approximation
f_taylor = lambda x: np.sin(x)
f_taylor_approx = lambda x: x - x**3 / 6 + x**5 / 120  # 5th order approximation
x_taylor = np.linspace(-2*np.pi, 2*np.pi, 500)

# 4. Chain Rule Visualization
chain_f = lambda u: u**2
chain_g = lambda x: np.sin(x)
chain_h = lambda x: chain_f(chain_g(x))
chain_h_deriv = lambda x: 2 * chain_g(x) * np.cos(x)  # Derivative using chain rule
x_chain = np.linspace(-2*np.pi, 2*np.pi, 500)

# 5. Divergence and Curl (Vector Calculus)
x_div, y_div = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
u = x_div
v = y_div
curl = np.gradient(v, axis=0) - np.gradient(u, axis=1)

# 6. Hessian Matrix Visualization
hessian_f = lambda x, y: x**2 + y**2
x_hessian, y_hessian = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
hessian_z = hessian_f(x_hessian, y_hessian)

# Create visualizations
fig, ax = plt.subplots(3, 3, figsize=(15, 15))

# 1. Gradient Descent
ax[0, 0].plot(x_vals, f(x_vals), label='f(x) = x^2')
ax[0, 0].scatter(x_points, y_points, color='red', label='Gradient Descent Steps')
ax[0, 0].set_title('Gradient Descent')
ax[0, 0].legend()
ax[0, 0].grid()

# 2. Partial Derivatives
ax[0, 1].contourf(x_partial, y_partial, z_partial, levels=50, cmap='viridis')
ax[0, 1].set_title('Partial Derivatives (f(x, y) = x^2 + y^2)')
ax[0, 1].grid()

# 3. Taylor Series
ax[0, 2].plot(x_taylor, f_taylor(x_taylor), label='f(x) = sin(x)')
ax[0, 2].plot(x_taylor, f_taylor_approx(x_taylor), '--', label='Taylor Approximation')
ax[0, 2].set_title('Taylor Series Approximation')
ax[0, 2].legend()
ax[0, 2].grid()

# 4. Chain Rule
ax[1, 0].plot(x_chain, chain_h(x_chain), label='h(x) = (sin(x))^2')
ax[1, 0].plot(x_chain, chain_h_deriv(x_chain), '--', label="h'(x)")
ax[1, 0].set_title('Chain Rule')
ax[1, 0].legend()
ax[1, 0].grid()

# 5. Divergence and Curl
ax[1, 1].quiver(x_div, y_div, u, v)
ax[1, 1].set_title('Vector Field (Divergence and Curl)')
ax[1, 1].grid()

# 6. Hessian Matrix
ax[1, 2].contourf(x_hessian, y_hessian, hessian_z, levels=50, cmap='coolwarm')
ax[1, 2].set_title('Hessian Matrix Visualization')
ax[1, 2].grid()

# 7. Rate of Change in Multi-Dimensions
rate_x = np.linspace(-1, 1, 500)
rate_y = rate_x**2
ax[2, 0].plot(rate_x, rate_y, label='f(x) = x^2')
ax[2, 0].set_title('Rate of Change')
ax[2, 0].legend()
ax[2, 0].grid()

# 8. Instantaneous Change (Differentiation)
instant_x = np.linspace(-2, 2, 500)
instant_f = lambda x: x**3
instant_df = lambda x: 3*x**2
ax[2, 1].plot(instant_x, instant_f(instant_x), label='f(x) = x^3')
ax[2, 1].plot(instant_x, instant_df(instant_x), label="f'(x) = 3x^2")
ax[2, 1].set_title('Instantaneous Rate of Change')
ax[2, 1].legend()
ax[2, 1].grid()

# 9. Optimization (Critical Points)
optim_x = np.linspace(-2, 2, 500)
optim_f = lambda x: x**4 - 2*x**2
optim_df = lambda x: 4*x**3 - 4*x
ax[2, 2].plot(optim_x, optim_f(optim_x), label='f(x) = x^4 - 2x^2')
ax[2, 2].plot(optim_x, optim_df(optim_x), label="f'(x)")
ax[2, 2].scatter([-1, 0, 1], [optim_f(-1), optim_f(0), optim_f(1)], color='red', label='Critical Points')
ax[2, 2].set_title('Optimization and Critical Points')
ax[2, 2].legend()
ax[2, 2].grid()

plt.tight_layout()
plt.show()
