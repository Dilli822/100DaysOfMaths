import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Define mean and covariance matrix for a bivariate normal distribution
mean = [0, 0]  # Mean vector [mu_x, mu_y]
covariance_matrix = [[1, 0.8], [0.8, 1]]  # Covariance matrix

# Generate random samples from the bivariate Gaussian distribution
x, y = np.random.multivariate_normal(mean, covariance_matrix, 1000).T

# Plot 2D contour visualization
fig, ax = plt.subplots()
ax.hist2d(x, y, bins=30, cmap='Blues')
ax.set_title('2D Histogram of Bivariate Gaussian Distribution')
ax.set_xlabel('X Variable')
ax.set_ylabel('Y Variable')
plt.colorbar(ax.collections[0])
plt.show()

# Create grid for 3D surface plot
x_grid = np.linspace(-3, 3, 100)
y_grid = np.linspace(-3, 3, 100)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

# Multivariate Gaussian PDF function
def multivariate_gaussian_pdf(x, y, mean, cov_matrix):
    pos = np.dstack((x, y))
    det = np.linalg.det(cov_matrix)
    inv_cov = np.linalg.inv(cov_matrix)
    norm_const = 1 / (2 * np.pi * np.sqrt(det))
    diff = pos - mean
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=2)
    return norm_const * np.exp(exponent)

# Calculate Z values for the surface plot
z_values = multivariate_gaussian_pdf(x_mesh, y_mesh, mean, np.array(covariance_matrix))

# Plot 3D surface plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_mesh, y_mesh, z_values, cmap=cm.coolwarm, edgecolor='none')
ax.set_title('3D Surface Plot of Bivariate Gaussian Distribution')
ax.set_xlabel('X Variable')
ax.set_ylabel('Y Variable')
ax.set_zlabel('Probability Density')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Example covariance matrix for 3 variables
cov_matrix = np.array([
    [4, 2, 1],
    [2, 3, 0.5],
    [1, 0.5, 2]
])

# Create labels for variables
labels = ['X', 'Y', 'Z']

# ================== 2D Heatmap ==================
plt.figure(figsize=(6, 6))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', xticklabels=labels, yticklabels=labels)
plt.title('2D Heatmap of Covariance Matrix')
plt.show()

# ================== 3D Surface Plot ==================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for 3D plotting
x = np.arange(cov_matrix.shape[0])
y = np.arange(cov_matrix.shape[1])
x, y = np.meshgrid(x, y)
z = cov_matrix

# Plot the 3D surface
ax.plot_surface(x, y, z, cmap='viridis')

# Customize the plot
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_zlabel('Covariance Value')
ax.set_title('3D Surface Plot of Covariance Matrix')
plt.show()

# ================== Animated 3D Plot ==================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Display the animation
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Generate some sample data
np.random.seed(42)
data = np.random.randn(100, 3)

# Covariance matrix calculation
cov_matrix = np.cov(data.T)

# Function to animate the plot
def update(frame):
    # Randomly change the covariance matrix (for demonstration)
    noise = np.random.normal(0, 0.1, cov_matrix.shape)
    noisy_cov = cov_matrix + noise * frame / 10
    
    # Eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(noisy_cov)
    
    # Create a new 3D plot
    ax.cla()
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])

    # Plot the eigenvectors
    for i in range(3):
        ax.quiver(0, 0, 0, eigenvectors[0, i], eigenvectors[1, i], eigenvectors[2, i], length=2, color='b')
    
    # Plot the data points
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='o')

    # Set title
    ax.set_title(f'Frame {frame}: Covariance matrix evolution')

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(1, 11), interval=500)

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the covariance calculations for the games
def covariance_game_1():
    X = np.array([1, -1])  # Possible outcomes for X (player 1)
    Y = np.array([1, -1])  # Possible outcomes for Y (player 2)
    P = np.array([0.5, 0.5])  # Probabilities
    return np.mean((X - np.mean(X)) * (Y - np.mean(Y)))  # Covariance calculation

def covariance_game_2():
    X = np.array([1, -1])  # Possible outcomes for X
    Y = np.array([-1, 1])  # Possible outcomes for Y
    P = np.array([0.5, 0.5])  # Probabilities
    return np.mean((X - np.mean(X)) * (Y - np.mean(Y)))

def covariance_game_3():
    X = np.array([1, 1, -1, -1])  # Possible outcomes for X
    Y = np.array([1, -1, 1, -1])  # Possible outcomes for Y
    P = np.array([0.25, 0.25, 0.25, 0.25])  # Probabilities
    return np.mean((X - np.mean(X)) * (Y - np.mean(Y)))

# Covariance values for each game
cov_game_1 = covariance_game_1()
cov_game_2 = covariance_game_2()
cov_game_3 = covariance_game_3()

# Set up the figure for 2D plot
fig, ax = plt.subplots(figsize=(8, 6))
games = ['Game 1', 'Game 2', 'Game 3']
covariances = [cov_game_1, cov_game_2, cov_game_3]
ax.bar(games, covariances, color=['blue', 'red', 'green'])
ax.set_title('Covariance for Each Game')
ax.set_ylabel('Covariance')

plt.show()

# 3D Covariance Animation Plot

# Generate data points for animation
def generate_game_data():
    X = np.array([1, -1, 1, -1])
    Y = np.array([1, -1, 1, -1])
    return np.column_stack((X, Y))

# Function to update the plot for animation
def update_plot(frame, scat, data):
    x = data[:frame, 0]
    y = data[:frame, 1]
    scat.set_offsets(np.c_[x, y])
    return scat,

# Create a 3D plot for covariance visualization
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
data = generate_game_data()

# Create scatter plot and set labels
scat = ax_3d.scatter(data[:, 0], data[:, 1], np.zeros_like(data[:, 0]), c='b')
ax_3d.set_xlabel('Player X')
ax_3d.set_ylabel('Player Y')
ax_3d.set_zlabel('Covariance Value')
ax_3d.set_title('3D Animation Covariance Visualization')

# Create animation
ani = FuncAnimation(fig_3d, update_plot, frames=len(data), fargs=(scat, data), interval=500, blit=False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ================== Single-variate Gaussian ==================
# Parameters for single-variate Gaussian
mu_uni = 0  # Mean
sigma_uni = 1  # Standard deviation
x_uni = np.linspace(-5, 5, 100)  # X-axis values
pdf_uni = norm.pdf(x_uni, mu_uni, sigma_uni)  # PDF values

# ================== Multi-variate Gaussian ==================
# Parameters for multi-variate Gaussian
mu_multi = [0, 0]  # Mean vector
cov_multi = [[1, 0.5], [0.5, 1]]  # Covariance matrix

# Create a grid for multi-variate Gaussian
x_multi, y_multi = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pos = np.dstack((x_multi, y_multi))
rv = multivariate_normal(mu_multi, cov_multi)
pdf_multi = rv.pdf(pos)  # PDF values

# ================== 3D Plot Setup ==================
fig = plt.figure(figsize=(12, 6))

# Subplot 1: Single-variate Gaussian
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x_uni, np.zeros_like(x_uni), pdf_uni, color='blue', label='Single-variate Gaussian')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('PDF')
ax1.set_title('Single-variate Gaussian')
ax1.legend()

# Subplot 2: Multi-variate Gaussian
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x_multi, y_multi, pdf_multi, cmap='viridis', label='Multi-variate Gaussian')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('PDF')
ax2.set_title('Multi-variate Gaussian')

# ================== Animation Function ==================
def update(frame):
    # Clear previous frames
    ax1.cla()
    ax2.cla()

    # Replot single-variate Gaussian
    ax1.plot(x_uni, np.zeros_like(x_uni), pdf_uni, color='blue', label='Single-variate Gaussian')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('PDF')
    ax1.set_title('Single-variate Gaussian')
    ax1.legend()

    # Replot multi-variate Gaussian
    ax2.plot_surface(x_multi, y_multi, pdf_multi, cmap='viridis', label='Multi-variate Gaussian')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('PDF')
    ax2.set_title('Multi-variate Gaussian')

    # Rotate the views
    ax1.view_init(elev=20, azim=frame)
    ax2.view_init(elev=20, azim=frame)

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)

# Display the animation
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Generate random data
x = np.random.rand(100)
y = 2 * x + np.random.randn(100) * 0.1  # Linear relationship with noise
z = 3 * x + np.random.randn(100) * 0.1  # Adding a third variable

# Compute correlation coefficient for 2D
corr_coef_2d, _ = pearsonr(x, y)

# Create the figure and subplots
fig = plt.figure(figsize=(12, 8))

# 2D Plot
ax1 = fig.add_subplot(131)
ax1.scatter(x, y, color='b', label="Data Points")
ax1.set_title(f"2D Scatter Plot\nCorrelation Coefficient: {corr_coef_2d:.2f}")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()
ax1.grid(True)

# 3D Plot
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(x, y, z, color='r')
ax2.set_title("3D Scatter Plot")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Function to update the plot for animation
def update(frame_num, x, y, scat):
    scat.set_offsets(np.c_[x[:frame_num], y[:frame_num]])
    return scat,

# Create animated plot
ax3 = fig.add_subplot(133)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 2)
scat = ax3.scatter([], [])
ax3.set_title(f"Animated 2D Scatter Plot")
ax3.set_xlabel('X')
ax3.set_ylabel('Y')

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(x), fargs=(x, y, scat), interval=50)

# Display the plot
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Generate random data
x = np.random.rand(100)
y = -2 * x + np.random.randn(100) * 0.1  # Negative linear relationship with noise
z = 3 * x + np.random.randn(100) * 0.1  # Adding a third variable

# Compute correlation coefficient for 2D
corr_coef_2d, _ = pearsonr(x, y)

# Determine correlation description
if corr_coef_2d > 0:
    correlation_type = "Positive"
elif corr_coef_2d < 0:
    correlation_type = "Negative"
else:
    correlation_type = "Zero"

# Create the figure and subplots
fig = plt.figure(figsize=(12, 8))

# 2D Plot
ax1 = fig.add_subplot(131)
ax1.scatter(x, y, color='b', label="Data Points")
ax1.set_title(f"2D Scatter Plot\nCorrelation: {correlation_type} ({corr_coef_2d:.2f})")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()
ax1.grid(True)

# 3D Plot
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(x, y, z, color='r')
ax2.set_title("3D Scatter Plot")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Function to update the plot for animation
def update(frame_num, x, y, scat):
    scat.set_offsets(np.c_[x[:frame_num], y[:frame_num]])
    return scat,

# Create animated plot
ax3 = fig.add_subplot(133)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 2)
scat = ax3.scatter([], [])
ax3.set_title(f"Animated 2D Scatter Plot\nCorrelation: {correlation_type} ({corr_coef_2d:.2f})")
ax3.set_xlabel('X')
ax3.set_ylabel('Y')

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(x), fargs=(x, y, scat), interval=50)

# Display the plot
plt.tight_layout()
plt.show()
