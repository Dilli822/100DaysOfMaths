import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# K-Means Clustering Visualization (2D)
def plot_kmeans_2d():
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10  # Generate random points
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=100, edgecolor='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title("K-Means Clustering (2D)")
    plt.legend()
    plt.show()

# 3D Plot of Q-Values (State-Action Function) 
def plot_q_function_3d():
    states = np.linspace(1, 5, 5)
    actions = np.array(["Left", "Right"])
    q_values = np.array([[1.25, 0.5], [1.5, 0.8], [2.0, 1.2], [2.5, 1.5], [3.0, 2.0]])
    
    fig = go.Figure()
    for i in range(len(actions)):
        fig.add_trace(go.Scatter3d(x=states, y=[actions[i]]*len(states), z=q_values[:, i],
                                   mode='lines+markers', name=f'Action {actions[i]}'))
    
    fig.update_layout(title='Q-Function 3D Visualization',
                      scene=dict(xaxis_title='State', yaxis_title='Action', zaxis_title='Q-Value'))
    fig.show()

# Discounted Rewards Visualization (2D)
def plot_discounted_rewards():
    gamma = 0.75
    rewards = np.array([0, 0, 0, 100])  # Reward structure
    discounted_rewards = [gamma**i * rewards[i] for i in range(len(rewards))]
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 5), discounted_rewards, marker='o', linestyle='-', color='b', label='Discounted Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Discounted Rewards Over Time')
    plt.legend()
    plt.grid()
    plt.show()

# Execute plots
plot_kmeans_2d()
plot_q_function_3d()
plot_discounted_rewards()


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define MDP states and actions
states = [1, 2, 3, 4, 5]
actions = ["←", "→"]

# Define transitions and Q-values (arbitrarily chosen for demo)
transitions = {
    (2, "→"): (3, -10),  # From state 2, going right → leads to state 3 with Q(s,a)=-10
    (2, "←"): (1, 50),   # From state 2, going left ← leads to state 1 with Q(s,a)=50
    (3, "→"): (4, -20),
    (3, "←"): (2, 12.5),
    (4, "→"): (5, 10),
    (4, "←"): (3, 12.5),
    (5, "→"): (5, 0),
    (5, "←"): (4, 1.25),
}

# Create directed graph
G = nx.DiGraph()

# Add nodes (states)
for state in states:
    G.add_node(state)

# Add edges (state-action transitions)
for (s, a), (s_next, Q_val) in transitions.items():
    G.add_edge(s, s_next, label=f"{a}\nQ={Q_val}")

# Draw the MDP graph
pos = nx.spring_layout(G)  # Positioning
plt.figure(figsize=(8, 6))

# Draw nodes and edges
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="gray", font_size=12)
edge_labels = {(s, s_next): label for (s, a), (s_next, label) in transitions.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

# Title
plt.title("MDP Flow Graph with Q* Values")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define state space (x, y, z positions of helicopter)
x = np.linspace(-10, 10, 10)  # X-axis range (left-right)
y = np.linspace(-10, 10, 10)  # Y-axis range (forward-backward)
z = np.linspace(0, 20, 10)    # Z-axis range (altitude)

# Create mesh grid
X, Y, Z = np.meshgrid(x, y, z)

# Define Q-values as a function of (x, y, z) [Simulated Q* values]
Q_values = np.exp(-(X**2 + Y**2) / 50) + 0.1 * Z  # Example: Higher altitude has higher Q-values

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for Q-values at different positions
sc = ax.scatter(X, Y, Z, c=Q_values.flatten(), cmap='coolwarm', marker='o')

# Labels and title
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Altitude (Z)")
ax.set_title("Helicopter RL Q* Function Visualization")

# Add colorbar
plt.colorbar(sc, ax=ax, label="Q(s, a) Value")

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define grid size (Mars terrain)
grid_size = 10
x = np.arange(grid_size)
y = np.arange(grid_size)

# Define terrain elevation (Z-axis)
terrain = np.random.rand(grid_size, grid_size) * 50  # Random elevation between 0-50

# Define Q-values (higher values = better paths)
Q_values = np.exp(-(x[:, None]**2 + y**2) / 20) + terrain / 100

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid
X, Y = np.meshgrid(x, y)

# Plot surface
ax.plot_surface(X, Y, terrain, facecolors=plt.cm.viridis(Q_values), alpha=0.8)

# Labels and title
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Elevation (m)")
ax.set_title("Mars Rover Navigation using Q*")

plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define grid size (Mars terrain)
grid_size = 10
terrain = np.random.rand(grid_size, grid_size) * 50  # Random terrain elevation

# Define Q-values (higher values = better paths)
Q_values = np.exp(-(np.arange(grid_size)[:, None]**2 + np.arange(grid_size)**2) / 20) + terrain / 100

# Define start and goal
start = (0, 0)  # Top-left
goal = (grid_size - 1, grid_size - 1)  # Bottom-right

# Compute best path using greedy policy (argmax Q-values)
path = [start]
current = start

while current != goal:
    x, y = current
    # Get valid neighbors (Right, Down)
    neighbors = [(x+1, y), (x, y+1)]
    neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < grid_size and 0 <= ny < grid_size]
    
    # Select next step with highest Q-value
    current = max(neighbors, key=lambda pos: Q_values[pos])
    path.append(current)

# Convert path to X and Y coordinates
path_x, path_y = zip(*path)

# Plot terrain heatmap
plt.figure(figsize=(8, 6))
plt.imshow(Q_values, cmap='viridis', origin='upper')
plt.colorbar(label="Q-values (higher = better path)")

# Overlay best path
plt.plot(path_y, path_x, marker='o', color='red', linestyle='-', linewidth=2, markersize=6, label="Best Path")

# Labels
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Optimal Path for Mars Rover using Q*")
plt.legend()
plt.show()
