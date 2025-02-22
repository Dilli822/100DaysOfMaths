import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_1d():
    # Define function f(x) = x^2 - 4x + 4
    x = np.linspace(-2, 6, 100)
    y = x**2 - 4*x + 4
    
    # Gradient Descent Steps
    learning_rate = 0.1
    x_init = 5  # Initial point
    x_vals = [x_init]
    for _ in range(10):
        grad = 2 * x_vals[-1] - 4  # df/dx = 2x - 4
        x_new = x_vals[-1] - learning_rate * grad
        x_vals.append(x_new)
    
    y_vals = [x**2 - 4*x + 4 for x in x_vals]
    
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label='f(x) = x² - 4x + 4', color='blue')
    plt.scatter(x_vals, y_vals, color='red', label='Gradient Descent Steps')
    plt.title('Gradient Descent in One Variable')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.show()

def gradient_descent_2d():
    # Define function f(x, y) = x² + y² - 6x
    X, Y = np.meshgrid(np.linspace(-2, 6, 30), np.linspace(-4, 4, 30))
    Z = X**2 + Y**2 - 6*X
    
    # Gradient Descent Steps
    learning_rate = 0.1
    xy_init = np.array([5, 3])
    xy_vals = [xy_init]
    for _ in range(10):
        grad_x = 2 * xy_vals[-1][0] - 6
        grad_y = 2 * xy_vals[-1][1]
        xy_new = xy_vals[-1] - learning_rate * np.array([grad_x, grad_y])
        xy_vals.append(xy_new)
    
    xy_vals = np.array(xy_vals)
    
    plt.figure(figsize=(6, 5))
    plt.contour(X, Y, Z, levels=20, cmap='coolwarm')
    plt.plot(xy_vals[:, 0], xy_vals[:, 1], 'ro-', label='Gradient Descent Path')
    plt.scatter(xy_vals[0, 0], xy_vals[0, 1], color='green', label='Start')
    plt.scatter(xy_vals[-1, 0], xy_vals[-1, 1], color='black', label='End')
    plt.title('Gradient Descent in Two Variables')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

def least_squares():
    # Generate sample data
    x = np.linspace(0, 10, 10)
    y = 3*x + 2 + np.random.normal(0, 2, 10)  # y = 3x + 2 + noise
    
    # Fit line using Least Squares
    coef = np.polyfit(x, y, 1)
    y_fit = coef[0] * x + coef[1]
    
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='red', label='Data Points')
    plt.plot(x, y_fit, color='blue', label=f'Best Fit: y={coef[0]:.2f}x + {coef[1]:.2f}')
    plt.title('Least Squares Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

def critical_points():
    # Define function f(x) = x³ - 6x² + 9x + 1
    x = np.linspace(-1, 5, 100)
    y = x**3 - 6*x**2 + 9*x + 1
    
    # Compute derivative and find critical points
    dy_dx = 3*x**2 - 12*x + 9
    critical_x = np.roots([3, -12, 9])
    critical_y = [x**3 - 6*x**2 + 9*x + 1 for x in critical_x]
    
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label='f(x) = x³ - 6x² + 9x + 1', color='blue')
    plt.scatter(critical_x, critical_y, color='red', label='Critical Points', zorder=3)
    plt.title('Finding Critical Points')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.show()

# Run the functions to generate plots
gradient_descent_1d()
gradient_descent_2d()
least_squares()
critical_points()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gradient_descent_3d():
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = X**2 + Y**2 - 6*X
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax.scatter([0], [1], [0**2 + 1**2 - 6*0], color='red', s=100, label='Starting Point')
    ax.set_title('Gradient Descent in Two Variables')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Function Value')
    plt.legend()
    plt.show()

def least_squares_3d():
    np.random.seed(0)
    X = np.linspace(-5, 5, 100)
    Y = 3 * X + 5 + np.random.normal(0, 2, 100)
    Z = X**2 + Y**2
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, color='red', label='Data Points')
    ax.plot_trisurf(X, Y, Z, cmap='coolwarm', alpha=0.6)
    ax.set_title('Least Squares Regression in 3D')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Squared Error')
    plt.legend()
    plt.show()

def critical_points_3d():
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = X**3 - 6*X**2 + 9*X + 1 + Y**2
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7)
    ax.scatter([2], [0], [2**3 - 6*2**2 + 9*2 + 1 + 0**2], color='blue', s=100, label='Critical Point')
    ax.set_title('Finding Critical Points')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Function Value')
    plt.legend()
    plt.show()

# Run the plots
gradient_descent_3d()
least_squares_3d()
critical_points_3d()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def function(X, Y):
    return X**2 + Y**2 - 6*X

def gradient(X, Y):
    dfdx = 2*X - 6
    dfdy = 2*Y
    return dfdx, dfdy

def gradient_descent_animation():
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = function(X, Y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax.set_title('Gradient Descent Animation')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Function Value')
    
    x_path, y_path, z_path = [0], [1], [function(0, 1)]
    learning_rate = 0.1
    x, y = 0, 1
    
    for _ in range(15):
        dfdx, dfdy = gradient(x, y)
        x -= learning_rate * dfdx
        y -= learning_rate * dfdy
        x_path.append(x)
        y_path.append(y)
        z_path.append(function(x, y))
    
    scatter = ax.scatter([], [], [], color='red', s=100, label='Path')
    
    def update(frame):
        scatter._offsets3d = (x_path[:frame], y_path[:frame], z_path[:frame])
        return scatter,
    
    ani = animation.FuncAnimation(fig, update, frames=len(x_path), interval=500, blit=False)
    plt.legend()
    plt.show()

gradient_descent_animation()
