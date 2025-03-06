import numpy as np
import matplotlib.pyplot as plt

# Cost function: J(w) = (w - 3)^2
J = lambda w: (w - 3)**2

def gradient(w):
    return 2 * (w - 3)

# Initialize parameters
w_gd = w_adam = -2  # Start at w = -2
alpha = 0.1  # Learning rate
beta1, beta2 = 0.9, 0.999  # Adam parameters
epsilon = 1e-8  # Smoothing term
m, v = 0, 0  # Adam moments

gd_path, adam_path = [w_gd], [w_adam]

# Run optimization for 30 iterations
for t in range(1, 31):
    # Standard Gradient Descent
    w_gd -= alpha * gradient(w_gd)
    gd_path.append(w_gd)
    
    # Adam Optimizer
    g = gradient(w_adam)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    w_adam -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    adam_path.append(w_adam)

# Plot the cost function
w_vals = np.linspace(-4, 6, 100)
plt.plot(w_vals, J(w_vals), 'k-', label="Cost function J(w)")
plt.plot(gd_path, J(np.array(gd_path)), 'bo-', label="Gradient Descent")
plt.plot(adam_path, J(np.array(adam_path)), 'go-', label="Adam Optimizer")

plt.xlabel("w")
plt.ylabel("J(w)")
plt.legend()
plt.title("Gradient Descent vs. Adam Optimization")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

# 1. Gradient Descent on a Quadratic Cost Function

def cost_function(w):
    return (w - 1) ** 2  # Parabola centered at w=1

w_values = np.linspace(-2, 4, 100)
j_values = cost_function(w_values)

plt.figure(figsize=(8, 5))
plt.plot(w_values, j_values, label='Cost Function J(w)', color='blue')
plt.scatter(1, 0, color='red', label='Minimum at w=1')
plt.xlabel('w')
plt.ylabel('J(w)')
plt.title('Gradient Descent on a Quadratic Cost Function')
plt.legend()
plt.grid()
plt.show()

# 2. Standard Gradient Descent vs. Adam Optimizer

w = tf.Variable(3.0)
alpha = 0.1
iterations = 10
gd_w_values = []
adam_w_values = []

# Standard Gradient Descent
for _ in range(iterations):
    with tf.GradientTape() as tape:
        loss = cost_function(w)
    grad = tape.gradient(loss, w)
    w.assign_sub(alpha * grad)
    gd_w_values.append(w.numpy())

# Reset w for Adam
w.assign(3.0)
adam = tf.keras.optimizers.Adam(learning_rate=0.1)

# Adam Optimizer
for _ in range(iterations):
    with tf.GradientTape() as tape:
        loss = cost_function(w)
    grads = tape.gradient(loss, [w])
    adam.apply_gradients(zip(grads, [w]))
    adam_w_values.append(w.numpy())

plt.figure(figsize=(8, 5))
plt.plot(gd_w_values, label='Gradient Descent', marker='o', linestyle='--')
plt.plot(adam_w_values, label='Adam Optimizer', marker='s', linestyle='--')
plt.xlabel('Iterations')
plt.ylabel('w values')
plt.title('Gradient Descent vs. Adam Optimizer')
plt.legend()
plt.grid()
plt.show()

# 3. 3D Visualization of Collaborative Filtering Cost Function

def collaborative_cost(w, b):
    return (w**2 + b**2) - 4 * w * b  # Example collaborative filtering cost

w_vals = np.linspace(-2, 2, 50)
b_vals = np.linspace(-2, 2, 50)
W, B = np.meshgrid(w_vals, b_vals)
J = collaborative_cost(W, B)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, J, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('J(w, b)')
ax.set_title('Collaborative Filtering Cost Function')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# Generate a random binary user-item interaction matrix (10 users, 10 items)
np.random.seed(42)
interaction_matrix = np.random.choice([0, 1], size=(10, 10), p=[0.7, 0.3])

# 1. Heatmap of User-Item Interaction Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(interaction_matrix, cmap='Blues', annot=True, cbar=False)
plt.xlabel("Items")
plt.ylabel("Users")
plt.title("User-Item Interaction Matrix")
plt.show()

# 2. Latent Factor Space Projection using SVD
svd = TruncatedSVD(n_components=2)
latent_factors = svd.fit_transform(interaction_matrix)

plt.figure(figsize=(8, 6))
plt.scatter(latent_factors[:, 0], latent_factors[:, 1], color='red', label='Users')
plt.xlabel("Latent Factor 1")
plt.ylabel("Latent Factor 2")
plt.title("2D Projection of Users in Latent Space")
plt.legend()
plt.show()

# 3. Clustering Users based on Interaction Similarity
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(latent_factors)
labels = kmeans.labels_

plt.figure(figsize=(8, 6))
plt.scatter(latent_factors[:, 0], latent_factors[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.xlabel("Latent Factor 1")
plt.ylabel("Latent Factor 2")
plt.title("Clustering Users Based on Interactions")
plt.colorbar(label="Cluster ID")
plt.show()
