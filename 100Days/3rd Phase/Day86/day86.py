import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

# 1. Classification with a Neural Network: Motivation
def plot_moons():
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title("Classification with a Neural Network: Motivation")
    plt.show()

# 2. Classification with a Neural Network: Minimizing Log-Loss
def train_nn_log_loss():
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=500)
    model.fit(X_train, y_train)
    
    y_pred_prob = model.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_prob)
    print(f"Log-Loss: {loss:.4f}")
    
    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 100),
                         np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='bwr')
    plt.title("Minimizing Log-Loss in Classification")
    plt.show()

# 3. Gradient Descent and Backpropagation Visualization
def plot_gradient_descent():
    x = np.linspace(-2, 2, 100)
    y = x**2
    grad = 2*x  # Derivative of x^2
    
    plt.plot(x, y, label="Loss Function")
    plt.quiver(x[::10], y[::10], -grad[::10], np.zeros_like(grad[::10]), angles='xy', scale_units='xy', scale=10, color='r')
    plt.title("Gradient Descent - Direction of Optimization")
    plt.xlabel("Weight")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# 4. Newton’s Method Visualization
def newtons_method():
    def f(x): return x**3 - 2*x + 2
    def df(x): return 3*x**2 - 2  # Derivative
    
    x_vals = np.linspace(-2, 2, 400)
    y_vals = f(x_vals)
    
    x = -1.5  # Initial guess
    for _ in range(5):
        plt.plot(x_vals, y_vals, label='f(x) = x^3 - 2x + 2')
        plt.scatter(x, f(x), color='red', zorder=5)
        x = x - f(x)/df(x)  # Newton’s update
    
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.title("Newton's Method - Root Finding")
    plt.show()

# Run visualizations
plot_moons()
train_nn_log_loss()
plot_gradient_descent()
newtons_method()


import networkx as nx
import matplotlib.pyplot as plt

def draw_mlp():
    G = nx.DiGraph()

    # Input layer (n=3 for visualization)
    inputs = [r'$x_1$', r'$x_2$', r'$x_3$']
    hidden = [r'$h_1$', r'$h_2$', r'$h_3$']
    output = [r'$\hat{y}$']

    # Adding nodes
    for i in range(len(inputs)):
        G.add_node(inputs[i], layer=0)
    for h in hidden:
        G.add_node(h, layer=1)
    for o in output:
        G.add_node(o, layer=2)

    # Adding edges (properly formatted LaTeX)
    edges = []
    for i in range(len(inputs)):
        for j in range(len(hidden)):
            weight_label = rf'$w_{{{j+1},{i+1}}}^{{(1)}}$'
            edges.append((inputs[i], hidden[j], weight_label))  # Input to Hidden
    for j in range(len(hidden)):
        for o in output:
            weight_label = rf'$w_{{o,{j+1}}}^{{(2)}}$'
            edges.append((hidden[j], o, weight_label))  # Hidden to Output

    G.add_edges_from([(e[0], e[1]) for e in edges])

    # Positioning nodes
    pos = {
        r'$x_1$': (-1, 2),
        r'$x_2$': (-1, 1),
        r'$x_3$': (-1, 0),
        r'$h_1$': (0, 2),
        r'$h_2$': (0, 1),
        r'$h_3$': (0, 0),
        r'$\hat{y}$': (1, 1),
    }

    # Draw nodes
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=14, font_weight='bold')

    # Draw edges with labels
    edge_labels = {(e[0], e[1]): e[2] for e in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color='red')

    plt.title("MLP Architecture with Superscripts/Subscripts")
    plt.show()

draw_mlp()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

def plot_moons_3d():
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='bwr', edgecolors='k')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Class")
    ax.set_title("3D Classification with Neural Network")
    plt.show()

def train_nn_log_loss_3d():
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=500)
    model.fit(X_train, y_train)
    
    y_pred_prob = model.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_prob)
    print(f"Log-Loss: {loss:.4f}")
    
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 100),
                         np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, Z, alpha=0.3, cmap='bwr')
    ax.scatter(X[:, 0], X[:, 1], y, edgecolors='k', cmap='bwr')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Prediction")
    ax.set_title("3D Decision Boundary - Neural Network")
    plt.show()

def plot_gradient_descent_3d():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Loss function: simple quadratic bowl
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    
    # Gradient descent path
    w = np.array([-1.5, 1.5])
    for _ in range(10):
        grad = 2 * w  # Derivative
        w = w - 0.2 * grad  # Update step
        ax.scatter(w[0], w[1], w[0]**2 + w[1]**2, color='red', s=50)
    
    ax.set_xlabel("Weight 1")
    ax.set_ylabel("Weight 2")
    ax.set_zlabel("Loss")
    ax.set_title("3D Gradient Descent Visualization")
    plt.show()

def newtons_method_3d():
    def f(x): return x**3 - 2*x + 2
    def df(x): return 3*x**2 - 2  # Derivative
    
    x_vals = np.linspace(-2, 2, 400)
    y_vals = f(x_vals)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_vals, np.linspace(-5, 5, 400))
    Z = f(X)
    ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.7)
    
    x = -1.5  # Initial guess
    for _ in range(5):
        ax.scatter(x, 0, f(x), color='red', s=50)
        x = x - f(x)/df(x)  # Newton's update
    
    ax.set_xlabel("x")
    ax.set_ylabel("Iteration")
    ax.set_zlabel("f(x)")
    ax.set_title("3D Newton's Method Visualization")
    plt.show()

plot_moons_3d()
train_nn_log_loss_3d()
plot_gradient_descent_3d()
newtons_method_3d()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the function and its first and second derivatives
def f(x):
    return np.exp(x) - np.log(x)  # Function to minimize

def df(x):
    return np.exp(x) - 1/x  # First derivative

def ddf(x):
    return np.exp(x) + 1/(x**2)  # Second derivative (used in Newton’s update)

# Generate x values for function plot
x_vals = np.linspace(0.1, 2, 400)
y_vals = f(x_vals)

# Initialize figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_vals, y_vals, label=r"$g(x) = e^x - \log(x)$", color="blue", linewidth=2)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.legend()
ax.set_title("Newton’s Method for Minimization", fontsize=14)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("g(x)", fontsize=12)

# Initial guess for Newton’s method
x_start = 0.5  
x = x_start
prev_points = []  # Store previous calculated points

# Plot initialization
point, = ax.plot([], [], 'ro', markersize=8, label="Newton's Steps")
tangent_line, = ax.plot([], [], 'r--', linewidth=1.5)
path, = ax.plot([], [], 'go-', markersize=5, label="Path to Minimum")

# Annotation for iteration steps
annotation = ax.text(1.0, -2, "", fontsize=12, color='red')

# True minimum of the function found using Newton’s Method
x_min = 0.567  

# Animation update function
def update(frame):
    global x, prev_points

    # Reset animation if convergence is reached
    if abs(x - x_min) < 1e-3:
        x = x_start
        prev_points = []  # Reset stored points

    # Newton’s Method Update Step
    x_new = x - df(x) / ddf(x)

    # Store calculated points
    prev_points.append((x, f(x)))

    # Compute tangent line at current x
    tangent_x = np.linspace(x - 0.3, x + 0.3, 100)
    tangent_y = f(x) + df(x) * (tangent_x - x)

    # Update Newton step point
    point.set_data([x], [f(x)])

    # Update tangent line
    tangent_line.set_data(tangent_x, tangent_y)

    # Update path (previous points)
    if prev_points:
        path.set_data(*zip(*prev_points))

    # Update annotation text
    annotation.set_text(f"Step {frame}: x = {x:.3f}")
    annotation.set_position((x, f(x) + 0.5))

    # Update x for next frame
    x = x_new

    return point, tangent_line, path, annotation

# Create animation
ani = animation.FuncAnimation(fig, update, frames=10, interval=800, blit=False)

plt.legend()
plt.show()
