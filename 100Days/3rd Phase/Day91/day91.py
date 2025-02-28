import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_2d_vectors(vectors, labels, colors, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, v in enumerate(vectors):
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=colors[i], label=labels[i])
    
    max_val = max(np.max(np.abs(vectors)), 1) + 1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xticks(range(-int(max_val), int(max_val) + 1))
    ax.set_yticks(range(-int(max_val), int(max_val) + 1))
    ax.set_title(title)
    ax.legend()
    plt.grid()
    plt.show()

def plot_3d_vectors(vectors, labels, colors, title):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i, v in enumerate(vectors):
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=colors[i], label=labels[i])
    
    max_val = max(np.max(np.abs(vectors)), 1) + 1
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(title)
    ax.legend()
    plt.show()

# Example: Basic Vectors
v1 = np.array([3, 2])
v2 = np.array([-2, 3])
plot_2d_vectors([v1, v2], ["Vector 1", "Vector 2"], ['r', 'b'], "Basic 2D Vectors")

# Example: 3D Vectors
v3 = np.array([2, 1, 3])
v4 = np.array([-1, 2, 2])
plot_3d_vectors([v3, v4], ["Vector 1", "Vector 2"], ['g', 'm'], "Basic 3D Vectors")


