
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data for regression
np.random.seed(42)
X_reg = np.linspace(0, 10, 20)
y_reg = 2 * X_reg + 3 + np.random.randn(20) * 2  # y = 2x + 3 + noise

# Fit a perceptron-like linear model (for regression)
w1, b = np.polyfit(X_reg, y_reg, 1)  # Linear fit

# Plot Regression
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_reg, y_reg, label="Data points")
plt.plot(X_reg, w1 * X_reg + b, color='red', label=f"Regression Line: y={w1:.2f}x + {b:.2f}")
plt.title("Perceptron-Based Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Generate sample data for classification
np.random.seed(42)
X_class = np.random.randn(100, 2)
y_class = np.sign(X_class[:, 0] + X_class[:, 1] - 0.2)  # Decision boundary: x1 + x2 = 0.2

# Fit a perceptron decision boundary
w = np.array([1, 1])
b = -0.2

# Plot Classification
plt.subplot(1, 2, 2)
plt.scatter(X_class[:, 0], X_class[:, 1], c=y_class, cmap="coolwarm", edgecolors='k')
x_vals = np.linspace(-2, 2, 100)
y_vals = -(w[0] * x_vals + b) / w[1]  # Decision boundary
plt.plot(x_vals, y_vals, 'k--', label="Decision Boundary")
plt.title("Perceptron-Based Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

# Show plots
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed
torch.manual_seed(42)

# ----------------------
# 1. Regression with Neural Network
# ----------------------
X_reg = torch.linspace(0, 10, 100).reshape(-1, 1)
y_reg = 2 * X_reg + 3 + torch.randn(100, 1) * 2  # y = 2x + 3 + noise

# Define simple NN model for regression
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
losses = []
for _ in range(100):
    optimizer.zero_grad()
    y_pred = model(X_reg)
    loss = criterion(y_pred, y_reg)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Plot regression
plt.figure(figsize=(12, 10))

plt.subplot(3, 3, 1)
plt.scatter(X_reg.numpy(), y_reg.numpy(), label="Data")
plt.plot(X_reg.numpy(), model(X_reg).detach().numpy(), color='red', label="NN Prediction")
plt.title("Regression with Neural Network")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()

# ----------------------
# 2. Loss Function in Regression
# ----------------------
plt.subplot(3, 3, 2)
plt.plot(losses, marker='o', linestyle='-')
plt.title("Loss Function in Neural Regression")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid()

# ----------------------
# 3. Gradient Descent in Neural Regression
# ----------------------
weights = []
for param in model.parameters():
    weights.append(param.detach().numpy().flatten())

plt.subplot(3, 3, 3)
plt.plot(weights[0], marker='o', linestyle='-', label="Weight Updates")
plt.title("Gradient Descent in Neural Regression")
plt.xlabel("Iterations")
plt.ylabel("Weight Values")
plt.legend()
plt.grid()

# ----------------------
# 4. Classification with Neural Network
# ----------------------
X_class = torch.randn(100, 2)
y_class = (X_class[:, 0] + X_class[:, 1] > 0).float().reshape(-1, 1)  # Labels: 0 or 1

# Simple NN for classification
class ClassifierNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

clf = ClassifierNN()
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.SGD(clf.parameters(), lr=0.1)

# Training loop
losses_class = []
for _ in range(100):
    optimizer.zero_grad()
    y_pred = clf(X_class)
    loss = criterion(y_pred, y_class)
    loss.backward()
    optimizer.step()
    losses_class.append(loss.item())

plt.subplot(3, 3, 4)
plt.scatter(X_class[:, 0], X_class[:, 1], c=y_class.flatten(), cmap="coolwarm", edgecolors='k')
plt.title("Classification with Neural Network")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid()

# ----------------------
# 5. Sigmoid Function in Classification
# ----------------------
z_vals = torch.linspace(-6, 6, 100)
sigmoid_vals = torch.sigmoid(z_vals)

plt.subplot(3, 3, 5)
plt.plot(z_vals.numpy(), sigmoid_vals.numpy(), 'b')
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid()

# ----------------------
# 6. Gradient Descent in Classification
# ----------------------
weights_class = []
for param in clf.parameters():
    weights_class.append(param.detach().numpy().flatten())

plt.subplot(3, 3, 6)
plt.plot(weights_class[0], marker='o', linestyle='-', label="Weight Updates")
plt.title("Gradient Descent in Classification")
plt.xlabel("Iterations")
plt.ylabel("Weight Values")
plt.legend()
plt.grid()

# ----------------------
# 7. Derivative Calculations in Classification
# ----------------------
y_pred_class = clf(X_class)
derivative = y_pred_class * (1 - y_pred_class)

plt.subplot(3, 3, 7)
plt.hist(derivative.detach().numpy(), bins=20, color='g', edgecolor='k')
plt.title("Derivative Calculations in Neural Classification")
plt.xlabel("Derivative Value")
plt.ylabel("Frequency")
plt.grid()

# Show all plots
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Function for sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate sample data for regression
np.random.seed(42)
X_reg = np.linspace(0, 10, 20)
y_reg = 2 * X_reg + 3 + np.random.randn(20) * 2  # y = 2x + 3 + noise

# Fit perceptron-like linear model
w1, b = np.polyfit(X_reg, y_reg, 1)

# Perceptron Regression Plot
plt.figure(figsize=(12, 10))

plt.subplot(3, 3, 1)
plt.scatter(X_reg, y_reg, label="Data")
plt.plot(X_reg, w1 * X_reg + b, color='red', label=f"y={w1:.2f}x + {b:.2f}")
plt.title("Regression with Perceptron")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()

# Loss Function Visualization (MSE)
y_pred = w1 * X_reg + b
loss = 0.5 * (y_reg - y_pred) ** 2
plt.subplot(3, 3, 2)
plt.plot(X_reg, loss, marker='o', linestyle='-')
plt.title("Loss Function in Perceptron Regression")
plt.xlabel("X")
plt.ylabel("Loss")
plt.grid()

# Gradient Descent for Perceptron Regression
w, b = 0, 0  # Initial weights
lr = 0.01  # Learning rate
loss_values = []

for _ in range(50):
    y_pred = w * X_reg + b
    grad_w = -np.mean(X_reg * (y_reg - y_pred))  # dL/dw
    grad_b = -np.mean(y_reg - y_pred)  # dL/db
    w -= lr * grad_w
    b -= lr * grad_b
    loss_values.append(np.mean(0.5 * (y_reg - y_pred) ** 2))

plt.subplot(3, 3, 3)
plt.plot(loss_values, marker='o', linestyle='-')
plt.title("Gradient Descent in Perceptron Regression")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid()

# Generate sample data for classification
X_class = np.random.randn(100, 2)
y_class = np.sign(X_class[:, 0] + X_class[:, 1] - 0.2)

# Perceptron Classification Plot
plt.subplot(3, 3, 4)
plt.scatter(X_class[:, 0], X_class[:, 1], c=y_class, cmap="coolwarm", edgecolors='k')
plt.title("Classification with Perceptron")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid()

# Sigmoid Function Plot
z_vals = np.linspace(-6, 6, 100)
plt.subplot(3, 3, 5)
plt.plot(z_vals, sigmoid(z_vals), 'b')
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid()

# Gradient Descent for Perceptron Classification
w = np.array([1, 1])
b = -0.2
x_vals = np.linspace(-2, 2, 100)
y_vals = -(w[0] * x_vals + b) / w[1]

plt.subplot(3, 3, 6)
plt.scatter(X_class[:, 0], X_class[:, 1], c=y_class, cmap="coolwarm", edgecolors='k')
plt.plot(x_vals, y_vals, 'k--', label="Decision Boundary")
plt.title("Gradient Descent in Classification")
plt.legend()
plt.grid()

# Derivative Calculations in Perceptron Classification
y_pred_class = sigmoid(np.dot(X_class, w) + b)
derivative = y_pred_class * (1 - y_pred_class)

plt.subplot(3, 3, 7)
plt.hist(derivative, bins=20, color='g', edgecolor='k')
plt.title("Derivative Calculations in Perceptron Classification")
plt.xlabel("Derivative Value")
plt.ylabel("Frequency")
plt.grid()

# Show all plots
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from graphviz import Digraph

# Set random seed
torch.manual_seed(42)

# ----------------------
# 1. Neural Network Figure (Perceptron)
# ----------------------
dot = Digraph()
dot.node('X1', 'Input X1')
dot.node('X2', 'Input X2')
dot.node('W1', 'Weight W1')
dot.node('W2', 'Weight W2')
dot.node('Σ', 'Σ: Weighted Sum')
dot.node('σ', 'Activation (Sigmoid)')
dot.node('Y', 'Output Y')

dot.edge('X1', 'Σ')
dot.edge('X2', 'Σ')
dot.edge('Σ', 'σ')
dot.edge('σ', 'Y')

dot.render('perceptron_diagram', format='png', cleanup=False)
dot.view()

# ----------------------
# 2. Regression with Neural Network
# ----------------------
X_reg = torch.linspace(0, 10, 100).reshape(-1, 1)
y_reg = 2 * X_reg + 3 + torch.randn(100, 1) * 2  # y = 2x + 3 + noise

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = []
for _ in range(100):
    optimizer.zero_grad()
    y_pred = model(X_reg)
    loss = criterion(y_pred, y_reg)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.figure(figsize=(12, 10))

# Plot regression
plt.subplot(3, 3, 1)
plt.scatter(X_reg.numpy(), y_reg.numpy(), label="Data")
plt.plot(X_reg.numpy(), model(X_reg).detach().numpy(), color='red', label="NN Prediction")
plt.title("Regression with Neural Network")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()

# ----------------------
# 3. Loss Function Visualization
# ----------------------
plt.subplot(3, 3, 2)
plt.plot(losses, marker='o', linestyle='-')
plt.title("Loss Function in Regression")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid()

# ----------------------
# 4. Gradient Descent Visualization
# ----------------------
weights = []
for param in model.parameters():
    weights.append(param.detach().numpy().flatten())

plt.subplot(3, 3, 3)
plt.plot(weights[0], marker='o', linestyle='-', label="Weight Updates")
plt.title("Gradient Descent in Regression")
plt.xlabel("Iterations")
plt.ylabel("Weight Values")
plt.legend()
plt.grid()

# ----------------------
# 5. Classification with Perceptron
# ----------------------
X_class = torch.randn(100, 2)
y_class = (X_class[:, 0] + X_class[:, 1] > 0).float().reshape(-1, 1)

class ClassifierNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

clf = ClassifierNN()
criterion = nn.BCELoss()
optimizer = optim.SGD(clf.parameters(), lr=0.1)

losses_class = []
for _ in range(100):
    optimizer.zero_grad()
    y_pred = clf(X_class)
    loss = criterion(y_pred, y_class)
    loss.backward()
    optimizer.step()
    losses_class.append(loss.item())

plt.subplot(3, 3, 4)
plt.scatter(X_class[:, 0], X_class[:, 1], c=y_class.flatten(), cmap="coolwarm", edgecolors='k')
plt.title("Classification with Neural Network")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid()

# ----------------------
# 6. Sigmoid Function Visualization
# ----------------------
z_vals = torch.linspace(-6, 6, 100)
sigmoid_vals = torch.sigmoid(z_vals)

plt.subplot(3, 3, 5)
plt.plot(z_vals.numpy(), sigmoid_vals.numpy(), 'b')
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid()

# ----------------------
# 7. Gradient Descent in Classification
# ----------------------
weights_class = []
for param in clf.parameters():
    weights_class.append(param.detach().numpy().flatten())

plt.subplot(3, 3, 6)
plt.plot(weights_class[0], marker='o', linestyle='-', label="Weight Updates")
plt.title("Gradient Descent in Classification")
plt.xlabel("Iterations")
plt.ylabel("Weight Values")
plt.legend()
plt.grid()

# ----------------------
# 8. Derivative in Classification
# ----------------------
y_pred_class = clf(X_class)
derivative = y_pred_class * (1 - y_pred_class)

plt.subplot(3, 3, 7)
plt.hist(derivative.detach().numpy(), bins=20, color='g', edgecolor='k')
plt.title("Derivative Calculations in Classification")
plt.xlabel("Derivative Value")
plt.ylabel("Frequency")
plt.grid()

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
torch.manual_seed(42)

# ----------------------
# 1. 3D Regression with Perceptron
# ----------------------
fig = plt.figure(figsize=(15, 12))

X_reg = torch.linspace(-5, 5, 100)
Y_reg = torch.linspace(-5, 5, 100)
X_grid, Y_grid = torch.meshgrid(X_reg, Y_reg)
Z_reg = 2 * X_grid + 3 * Y_grid + torch.randn(100, 100)  # True function: 2x + 3y

ax1 = fig.add_subplot(3, 3, 1, projection='3d')
ax1.plot_surface(X_grid.numpy(), Y_grid.numpy(), Z_reg.numpy(), cmap='viridis', alpha=0.7)
ax1.set_title("Regression with a Perceptron")
ax1.set_xlabel("X1")
ax1.set_ylabel("X2")
ax1.set_zlabel("Output (y)")
ax1.view_init(30, 60)

# ----------------------
# 2. 3D Loss Function Visualization (MSE)
# ----------------------
X_loss = np.linspace(-3, 3, 50)
Y_loss = np.linspace(-3, 3, 50)
X_grid, Y_grid = np.meshgrid(X_loss, Y_loss)
Z_loss = (X_grid**2 + Y_grid**2) / 2  # Loss = 1/2 * (y - y_pred)^2

ax2 = fig.add_subplot(3, 3, 2, projection='3d')
ax2.plot_surface(X_grid, Y_grid, Z_loss, cmap='coolwarm', alpha=0.7)
ax2.set_title("Loss Function in Regression")
ax2.set_xlabel("Weight 1")
ax2.set_ylabel("Weight 2")
ax2.set_zlabel("Loss (MSE)")
ax2.view_init(30, 60)

# ----------------------
# 3. 3D Gradient Descent for Regression
# ----------------------
weights_x = np.linspace(-3, 3, 10)
weights_y = np.linspace(-3, 3, 10)
Z_weights = (weights_x[:, None]**2 + weights_y[None, :]**2) / 2

ax3 = fig.add_subplot(3, 3, 3, projection='3d')
ax3.plot_surface(weights_x[:, None], weights_y[None, :], Z_weights, cmap='inferno', alpha=0.7)
ax3.set_title("Gradient Descent in Regression")
ax3.set_xlabel("Weight 1")
ax3.set_ylabel("Weight 2")
ax3.set_zlabel("Loss")
ax3.view_init(30, 60)

# ----------------------
# 4. 3D Classification with Perceptron (Decision Boundary)
# ----------------------
X_class = torch.randn(100, 2)
Y_class = (X_class[:, 0] + X_class[:, 1] > 0).float().reshape(-1, 1)

X1_grid, X2_grid = torch.meshgrid(torch.linspace(-2, 2, 50), torch.linspace(-2, 2, 50))
Z_class = X1_grid + X2_grid  # Decision boundary

ax4 = fig.add_subplot(3, 3, 4, projection='3d')
ax4.scatter(X_class[:, 0], X_class[:, 1], Y_class.flatten(), c=Y_class.flatten(), cmap="coolwarm")
ax4.plot_surface(X1_grid.numpy(), X2_grid.numpy(), Z_class.numpy(), color='gray', alpha=0.3)
ax4.set_title("Classification with Perceptron")
ax4.set_xlabel("Feature 1")
ax4.set_ylabel("Feature 2")
ax4.set_zlabel("Decision Boundary")
ax4.view_init(30, 60)

# ----------------------
# 5. 3D Sigmoid Function
# ----------------------
X_sigmoid = np.linspace(-6, 6, 50)
Y_sigmoid = np.linspace(-6, 6, 50)
X_grid, Y_grid = np.meshgrid(X_sigmoid, Y_sigmoid)
Z_sigmoid = 1 / (1 + np.exp(-(X_grid + Y_grid)))  # Sigmoid function

ax5 = fig.add_subplot(3, 3, 5, projection='3d')
ax5.plot_surface(X_grid, Y_grid, Z_sigmoid, cmap='plasma', alpha=0.8)
ax5.set_title("Sigmoid Function in Classification")
ax5.set_xlabel("Input X1")
ax5.set_ylabel("Input X2")
ax5.set_zlabel("Sigmoid Output")
ax5.view_init(30, 60)

# ----------------------
# 6. 3D Gradient Descent in Classification
# ----------------------
Z_grad = 1 / (1 + np.exp(-X_grid)) * (1 - 1 / (1 + np.exp(-X_grid)))

ax6 = fig.add_subplot(3, 3, 6, projection='3d')
ax6.plot_surface(X_grid, Y_grid, Z_grad, cmap='cividis', alpha=0.8)
ax6.set_title("Gradient Descent in Classification")
ax6.set_xlabel("Weight 1")
ax6.set_ylabel("Weight 2")
ax6.set_zlabel("Loss Gradient")
ax6.view_init(30, 60)

# ----------------------
# 7. 3D Derivative Calculations in Classification
# ----------------------
Z_derivative = Z_sigmoid * (1 - Z_sigmoid)  # Derivative of sigmoid

ax7 = fig.add_subplot(3, 3, 7, projection='3d')
ax7.plot_surface(X_grid, Y_grid, Z_derivative, cmap='magma', alpha=0.8)
ax7.set_title("Derivative in Perceptron Classification")
ax7.set_xlabel("Input X1")
ax7.set_ylabel("Input X2")
ax7.set_zlabel("d(Sigmoid)/dx")
ax7.view_init(30, 60)

plt.tight_layout()
plt.show()
