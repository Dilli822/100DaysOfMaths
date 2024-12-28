import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_vectors(vectors, ax, colors=None, labels=None, origin=[0, 0, 0], scale=1):
    if colors is None:
        colors = ['r', 'g', 'b']
    if labels is None:
        labels = [f"v{i}" for i in range(len(vectors))]
    for i, vec in enumerate(vectors):
        if len(vec) == 2:  # 2D case
            ax.quiver(*origin[:2], *vec, angles='xy', scale_units='xy', scale=scale, color=colors[i])
            ax.text(*vec, labels[i], color=colors[i])
        elif len(vec) == 3:  # 3D case
            ax.quiver(*origin, *vec, length=1, color=colors[i])
            ax.text(*vec, labels[i], color=colors[i])

def visualize_2d_linear_transform():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    original_vectors = [v1, v2]

    A = np.array([[2, 1], [1, 1]])
    transformed_vectors = [A @ v for v in original_vectors]

    plot_vectors(original_vectors, ax, colors=['blue', 'orange'], labels=['v1', 'v2'])
    plot_vectors(transformed_vectors, ax, colors=['red', 'green'], labels=['Av1', 'Av2'])

    ax.set_title('2D Linear Transformation')
    plt.show()

def visualize_3d_linear_transform():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    original_vectors = [v1, v2, v3]

    A = np.array([[1, 0.5, 0.5], [0, 1, 0.5], [0, 0, 1]])
    transformed_vectors = [A @ v for v in original_vectors]

    plot_vectors(original_vectors, ax, colors=['blue', 'orange', 'purple'], labels=['v1', 'v2', 'v3'], scale=1)
    plot_vectors(transformed_vectors, ax, colors=['red', 'green', 'black'], labels=['Av1', 'Av2', 'Av3'], scale=1)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_title('3D Linear Transformation')
    plt.show()

def linear_transformations():
    transformations = {
        "Dilation": np.array([[2, 0], [0, 2]]),
        "Contraction": np.array([[0.5, 0], [0, 0.5]]),
        "Shear": np.array([[1, 1], [0, 1]])
    }

    fig, axes = plt.subplots(1, len(transformations), figsize=(15, 5))
    for ax, (name, matrix) in zip(axes, transformations.items()):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        v1 = np.array([1, 0])
        v2 = np.array([0, 1])
        original_vectors = [v1, v2]

        transformed_vectors = [matrix @ v for v in original_vectors]

        plot_vectors(original_vectors, ax, colors=['blue', 'orange'], labels=['v1', 'v2'])
        plot_vectors(transformed_vectors, ax, colors=['red', 'green'], labels=['Tv1', 'Tv2'])
        ax.set_title(name)
    plt.tight_layout()
    plt.show()

visualize_2d_linear_transform()
linear_transformations()
visualize_3d_linear_transform()

def plot_vectors(vectors, ax, colors=None, labels=None, origin=[0, 0, 0], scale=1):
    if colors is None:
        colors = ['r', 'g', 'b']
    if labels is None:
        labels = [f"v{i}" for i in range(len(vectors))]
    for i, vec in enumerate(vectors):
        if len(vec) == 3:  # 3D case
            ax.quiver(*origin, *vec, length=1, color=colors[i])
            ax.text(*(vec + 0.1), labels[i], color=colors[i])

def visualize_3d_transformation(name, matrix, ax):
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    original_vectors = [v1, v2, v3]
    transformed_vectors = [matrix @ v for v in original_vectors]

    plot_vectors(original_vectors, ax, colors=['blue', 'orange', 'purple'], labels=['v1', 'v2', 'v3'])
    plot_vectors(transformed_vectors, ax, colors=['red', 'green', 'pink'], labels=['Tv1', 'Tv2', 'Tv3'])

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_title(name)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

def visualize_3d_transformations():
    transformations = {
        "Identity (Normal)": np.eye(3),
        "Dilation": np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        "Contraction": np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]),
        "Shear": np.array([[1, 0.5, 0], [0, 1, 0.5], [0, 0, 1]])
    }

    fig = plt.figure(figsize=(20, 10))
    for i, (name, matrix) in enumerate(transformations.items()):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        visualize_3d_transformation(name, matrix, ax)

    plt.tight_layout()
    plt.show()

visualize_3d_transformations()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def plot_vectors(vectors, ax, colors=None, labels=None, origin=[0, 0, 0]):
    if colors is None:
        colors = ['r', 'g', 'b']
    if labels is None:
        labels = [f"v{i}" for i in range(len(vectors))]
    for i, vec in enumerate(vectors):
        ax.quiver(*origin, *vec, color=colors[i], linewidth=2)
        ax.text(*(vec + 0.1), labels[i], color=colors[i])

def update_subplot(frame, ax, original_vectors, matrix, name):
    ax.clear()
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_title(name)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # Apply transformation matrix with animation scaling
    scale_factor = 1 + 0.1 * frame  # Scale over frames
    animated_matrix = matrix * scale_factor
    transformed_vectors = [animated_matrix @ v for v in original_vectors]

    plot_vectors(original_vectors, ax, colors=['blue', 'orange', 'purple'], labels=['v1', 'v2', 'v3'])
    plot_vectors(transformed_vectors, ax, colors=['red', 'green', 'yellow'], labels=['Tv1', 'Tv2', 'Tv3'])

def animate_subplots():
    transformations = [
        {"name": "Identity", "matrix": np.eye(3)},
        {"name": "Dilation", "matrix": np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])},
        {"name": "Contraction", "matrix": np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])},
        {"name": "Shear", "matrix": np.array([[1, 0.5, 0], [0, 1, 0.5], [0, 0, 1]])}
    ]

    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    original_vectors = [v1, v2, v3]

    fig = plt.figure(figsize=(12, 12))
    axes = [fig.add_subplot(2, 2, i + 1, projection='3d') for i in range(len(transformations))]

    def make_animate_func(ax, transformation):
        def animate(frame):
            update_subplot(frame, ax, original_vectors, transformation["matrix"], transformation["name"])
        return animate

    anims = []
    for ax, transformation in zip(axes, transformations):
        anim = FuncAnimation(fig, make_animate_func(ax, transformation), frames=20, interval=500, repeat=True)
        anims.append(anim)

    plt.tight_layout()
    plt.show()

animate_subplots()
