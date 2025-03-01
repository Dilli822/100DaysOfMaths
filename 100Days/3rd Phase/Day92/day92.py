import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def plot_linear_independence():
    fig, ax = plt.subplots()
    v1 = np.array([2, 3])
    v2 = np.array([4, 6])  # Linearly dependent (scalar multiple of v1)
    
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title("Linearly Dependent Vectors")
    ax.legend()
    plt.show()

def plot_vector_span():
    fig, ax = plt.subplots()
    v1 = np.array([1, 2])
    v2 = np.array([-1, 1])
    
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
    
    for a in np.linspace(-2, 2, 5):
        for b in np.linspace(-2, 2, 5):
            ax.scatter(a * v1[0] + b * v2[0], a * v1[1] + b * v2[1], color='gray', s=5)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title("Vector Span")
    ax.legend()
    plt.show()

def plot_determinant_area():
    v1 = np.array([2, 1])
    v2 = np.array([1, 3])
    matrix = np.column_stack((v1, v2))
    det = np.linalg.det(matrix)
    
    fig, ax = plt.subplots()
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b')
    
    vertices = np.array([[0, 0], v1, v1 + v2, v2, [0, 0]])
    ax.fill(vertices[:, 0], vertices[:, 1], 'g', alpha=0.3, label=f'Area = |det| = {abs(det):.2f}')
    
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_title("Determinant as Area")
    ax.legend()
    plt.show()

def plot_pca():
    np.random.seed(0)
    X = np.random.multivariate_normal([0, 0], [[3, 2], [2, 2]], 100)
    pca = PCA(n_components=2)
    pca.fit(X)
    
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Data')
    
    for i, (vector, variance) in enumerate(zip(pca.components_, pca.explained_variance_)):
        ax.quiver(0, 0, vector[0] * variance, vector[1] * variance, angles='xy', scale_units='xy', scale=1, label=f'PC{i+1}')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title("PCA: Principal Components")
    ax.legend()
    plt.show()

def plot_singularity():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    Z = 2 * X + 3 * Y  # Singular case: no independent third dimension
    
    ax.plot_surface(X, Y, Z, alpha=0.5, cmap='coolwarm')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Singular Transformation in 3D")
    plt.show()

# Call all functions
plot_linear_independence()
plot_vector_span()
plot_determinant_area()
plot_pca()
plot_singularity()
