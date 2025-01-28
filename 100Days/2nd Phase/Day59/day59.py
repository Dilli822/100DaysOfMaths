import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Parameters for Gaussian Distributions
mean_T, std_T = 10, 2  # Processing time
mean_L, std_L = 5, 1   # Network latency

# Generate samples
samples = 10000
T_samples = np.random.normal(mean_T, std_T, samples)
L_samples = np.random.normal(mean_L, std_L, samples)
R_samples = T_samples + L_samples

# 2D Visualization
plt.figure(figsize=(10, 6))
plt.hist(T_samples, bins=50, density=True, alpha=0.6, color='blue', label='Processing Time (T)')
plt.hist(L_samples, bins=50, density=True, alpha=0.6, color='orange', label='Network Latency (L)')
plt.hist(R_samples, bins=50, density=True, alpha=0.6, color='green', label='Response Time (R)')
plt.title('2D Histogram Visualization of Gaussian Distributions')
plt.xlabel('Time (ms)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# 3D Visualization
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Create histogram bins for 3D
hist, xedges, yedges = np.histogram2d(T_samples, L_samples, bins=30, density=True)
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='purple')
ax.set_title('3D Visualization of Processing and Latency Distribution')
ax.set_xlabel('Processing Time (T)')
ax.set_ylabel('Network Latency (L)')
ax.set_zlabel('Density')
plt.show()

# Animation Visualization
fig, ax = plt.subplots(figsize=(10, 6))

n_frames = 100
hist_data = []

for i in range(1, n_frames + 1):
    current_samples = T_samples[:i * (samples // n_frames)] + L_samples[:i * (samples // n_frames)]
    hist_data.append(current_samples)

# Initialize plot
def init():
    ax.clear()
    ax.set_xlim(5, 25)
    ax.set_ylim(0, 0.15)
    ax.set_title("Animated Gaussian Distribution Over Time")
    ax.set_xlabel("Response Time")
    ax.set_ylabel("Density")

# Animation function
def update(frame):
    ax.clear()
    ax.hist(hist_data[frame], bins=50, density=True, color='green', alpha=0.7)
    ax.set_title("Animated Gaussian Distribution Over Time")
    ax.set_xlabel("Response Time")
    ax.set_ylabel("Density")
    ax.grid(True)

ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False)
plt.show()


# -*- coding: utf-8 -*-
"""
Normal Distribution Visualizations
Requires: numpy, matplotlib, scipy
Installation (if needed):
pip install numpy matplotlib scipy
"""

#%% 1. 2D Visualization: Basic Normal Distribution

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate normal distribution data
mu = 0    # Mean
sigma = 1 # Standard deviation
data = np.random.normal(mu, sigma, 10000)

# Create figure
plt.figure(figsize=(10,6))

# Plot histogram
count, bins, patches = plt.hist(data, 30, density=True, 
                               alpha=0.6, color='lightblue',
                               label='Sampled Data')

# Plot PDF (Probability Density Function)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=3, 
         label='Theoretical PDF')

# Add decorations
plt.title('2D Normal Distribution (μ=0, σ=1)', fontsize=14)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%% 2. 3D Visualization: Bivariate Normal Distribution

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Create grid of points
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Create bivariate normal distribution
mu = [0, 0]          # Mean vector
cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix
rv = multivariate_normal(mu, cov)
Z = rv.pdf(pos)

# Create 3D plot
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                      linewidth=0, antialiased=True)

# Add decorations
ax.set_title('3D Bivariate Normal Distribution', fontsize=14)
ax.set_xlabel('X Axis', fontsize=12)
ax.set_ylabel('Y Axis', fontsize=12)
ax.set_zlabel('Probability Density', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

#%% 3. Animation: Evolving Normal Distribution

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Set up animation figure
fig, ax = plt.subplots(figsize=(10,6))
x = np.linspace(-5, 5, 1000)
line, = ax.plot(x, np.zeros_like(x), lw=3, color='darkred')
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.8)
ax.set_title('Evolving Normal Distribution', fontsize=14)
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.grid(True, alpha=0.3)

# Animation function
def animate(frame):
    # Gradually increase standard deviation
    sigma = 0.5 + (frame % 50)/20
    y = norm.pdf(x, 0, sigma)
    line.set_ydata(y)
    ax.set_title(f'Evolving σ: {sigma:.2f}', fontsize=14)
    return line,

# Create animation
ani = FuncAnimation(fig, animate, frames=100, 
                   interval=50, blit=True)

# For Jupyter notebook display
# HTML(ani.to_jshtml())

# For saving to file
# ani.save('normal_distribution_evolution.mp4', 
#         writer='ffmpeg', fps=20)

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm

# Generate data for three distributions
x = np.linspace(-10, 10, 1000)

# Standard normal
normal_dist = norm.pdf(x, loc=0, scale=2)

# Positive skew (alpha > 0)
pos_skew = skewnorm.pdf(x, a=10, loc=0, scale=2)

# Negative skew (alpha < 0)
neg_skew = skewnorm.pdf(x, a=-10, loc=0, scale=2)

plt.figure(figsize=(10, 6))
plt.plot(x, normal_dist, label='Normal Distribution (Symmetric)', color='blue')
plt.plot(x, pos_skew, label='Positive Skew Distribution', color='green')
plt.plot(x, neg_skew, label='Negative Skew Distribution', color='red')

plt.title('2D Visualization: Gaussian Distribution and Skewness')
plt.xlabel('X-axis (values)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Generate grid points
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Create 3D Skewed Gaussian Surface
Z = skewnorm.pdf(X + Y, a=5)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title("3D Visualization: Positive Skew Gaussian Surface")
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Probability Density')

plt.show()


import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(-10, 10, 1000)
line, = ax.plot([], [], lw=2)

def init():
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 0.5)
    ax.set_title('Skewness Transition in Gaussian Distribution')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Probability Density')
    return line,

def update(frame):
    skew_val = frame - 50  # Dynamic skew value
    y = skewnorm.pdf(x, a=skew_val)
    line.set_data(x, y)
    return line,

ani = animation.FuncAnimation(fig, update, frames=100, init_func=init, blit=True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Step 1: 2D Visualization of Normal Distribution
def plot_2d():
    # Generate 1000 random data points from a normal distribution
    data = np.random.normal(loc=0, scale=1, size=1000)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, color='skyblue', edgecolor='black')
    plt.title('2D Histogram of Normally Distributed Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

# Step 2: 3D Visualization of Normal Distribution (Bell Curve)
def plot_3d():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create grid and multivariate normal distribution
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    x, y = np.meshgrid(x, y)
    z = (1 / (2 * np.pi)) * np.exp(-0.5 * (x ** 2 + y ** 2))

    # Surface plot
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    ax.set_title('3D Bell Curve')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Density')

    plt.show()

# Step 3: Animated Visualization of Normal Distribution Dynamics
def animate_distribution():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.5)
    ax.set_title('Animated Normal Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

    line, = ax.plot([], [], lw=2)

    # Function to initialize the plot
    def init():
        line.set_data([], [])
        return line,

    # Function to update each frame
    def update(frame):
        data = np.random.normal(0, 1, 1000)
        counts, bins = np.histogram(data, bins=30, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        line.set_data(bin_centers, counts)
        return line,

    ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True)

    plt.show()

if __name__ == "__main__":
    print("Select an option to visualize Normal Distribution:")
    print("1. 2D Histogram")
    print("2. 3D Bell Curve")
    print("3. Animated Visualization")
    
    plot_2d()
    plot_3d()
    animate_distribution()
 


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew

# Scenario parameters
lottery_outcomes = [-1, 99]
lottery_probs = [0.99, 0.01]
insurance_outcomes = [1, -99]
insurance_probs = [0.99, 0.01]

# Calculate moments function
def calculate_moments(outcomes, probs):
    mean = np.sum(np.multiply(outcomes, probs))
    variance = np.sum(np.multiply(np.square(outcomes), probs))
    third_moment = np.sum(np.multiply(np.power(outcomes, 3), probs))
    skewness = third_moment / (variance ** 1.5)
    return mean, variance, skewness

# Calculate moments for both scenarios
lottery_mean, lottery_var, lottery_skew = calculate_moments(lottery_outcomes, lottery_probs)
insurance_mean, insurance_var, insurance_skew = calculate_moments(insurance_outcomes, insurance_probs)

# 1. 2D Visualization: Comparison Plot
plt.figure(figsize=(12, 6))

# Lottery plot
plt.subplot(1, 2, 1)
markerline, stemlines, baseline = plt.stem(lottery_outcomes, lottery_probs, basefmt=' ')
plt.setp(stemlines, color='blue', linewidth=2)
plt.setp(markerline, color='blue', markersize=10)
plt.title('Lottery Distribution\n'
          f'Mean: {lottery_mean:.1f}, Var: {lottery_var:.1f}\n'
          f'Skewness: {lottery_skew:.1f}', fontsize=12)
plt.xlabel('Outcome ($)', fontsize=10)
plt.ylabel('Probability', fontsize=10)
plt.ylim(0, 1)
plt.grid(alpha=0.3)

# Insurance plot
plt.subplot(1, 2, 2)
markerline, stemlines, baseline = plt.stem(insurance_outcomes, insurance_probs, basefmt=' ')
plt.setp(stemlines, color='red', linewidth=2)
plt.setp(markerline, color='red', markersize=10)
plt.title('Insurance Distribution\n'
          f'Mean: {insurance_mean:.1f}, Var: {insurance_var:.1f}\n'
          f'Skewness: {insurance_skew:.1f}', fontsize=12)
plt.xlabel('Outcome ($)', fontsize=10)
plt.ylim(0, 1)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 2. 3D Visualization: Moment Comparison
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Moments data
categories = ['Mean', 'Variance', 'Skewness']
lottery_values = [lottery_mean, lottery_var, lottery_skew]
insurance_values = [insurance_mean, insurance_var, insurance_skew]

# X-axis positions
x_pos = np.arange(len(categories))

# Plot bars
width = 0.4
ax.bar(x_pos - width/2, lottery_values, width, label='Lottery', color='blue', alpha=0.7)
ax.bar(x_pos + width/2, insurance_values, width, label='Insurance', color='red', alpha=0.7)

# Configure axes
ax.set_xticks(x_pos)
ax.set_xticklabels(categories)
ax.set_title('3D Moment Comparison', fontsize=14)
ax.set_zlabel('Moment Value', fontsize=12)
ax.legend()
plt.show()

# 3. Animation: Cumulative Outcomes
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-150, 150)
ax.set_ylim(0, 1000)
ax.set_title('Cumulative Outcome Distribution', fontsize=14)
ax.set_xlabel('Net Gain/Loss ($)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.grid(alpha=0.3)

# Initialize containers
lottery_bins = np.arange(-100, 200, 5)
insurance_bins = np.arange(-200, 100, 5)
lottery_hist, = ax.plot([], [], 'b-', label='Lottery')
insurance_hist, = ax.plot([], [], 'r-', label='Insurance')
ax.legend()

# Animation cache
lottery_data = []
insurance_data = []

def animate(frame):
    # Simulate 50 new trials per frame
    lottery_samples = np.random.choice(lottery_outcomes, 50, p=lottery_probs)
    insurance_samples = np.random.choice(insurance_outcomes, 50, p=insurance_probs)
    
    # Update data stores
    lottery_data.extend(lottery_samples)
    insurance_data.extend(insurance_samples)
    
    # Create histograms
    l_counts, _ = np.histogram(lottery_data, bins=lottery_bins)
    i_counts, _ = np.histogram(insurance_data, bins=insurance_bins)
    
    # Update plots
    lottery_hist.set_data((lottery_bins[:-1] + lottery_bins[1:])/2, l_counts)
    insurance_hist.set_data((insurance_bins[:-1] + insurance_bins[1:])/2, i_counts)
    ax.relim()
    ax.autoscale_view()
    return lottery_hist, insurance_hist,

ani = FuncAnimation(fig, animate, frames=100, interval=50, blit=True, cache_frame_data=False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Mock advertising data for visualization
np.random.seed(42)  # Ensure reproducibility
newspaper_budget = np.random.uniform(10, 50, 12)
sales_revenue = 3 * newspaper_budget + np.random.normal(0, 5, 12)  # Simple linear relation with noise

def plot_2d():
    """
    2D Scatter Plot of Newspaper Budget vs Sales Revenue
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(newspaper_budget, sales_revenue, color='skyblue', edgecolor='black')
    plt.title('2D Scatter Plot: Newspaper Budget vs Sales Revenue')
    plt.xlabel('Newspaper Ad Budget ($K)')
    plt.ylabel('Sales Revenue ($K)')
    plt.grid(True)
    plt.show()


def plot_3d():
    """
    3D Scatter Plot of Advertising Budgets and Sales Revenue
    """
    tv_budget = np.random.uniform(10, 50, 12)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(newspaper_budget, tv_budget, sales_revenue, color='purple')

    ax.set_title('3D Plot: Advertising Budgets and Sales Revenue')
    ax.set_xlabel('Newspaper Budget ($K)')
    ax.set_ylabel('TV Budget ($K)')
    ax.set_zlabel('Sales Revenue ($K)')

    plt.show()


def animate_distribution():
    """
    Animated Visualization of Changing Quartile Lines
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 200)
    ax.set_title('Animated Quartile Visualization')
    ax.set_xlabel('Newspaper Budget ($K)')
    ax.set_ylabel('Sales Revenue ($K)')

    scatter = ax.scatter([], [], color='orange', edgecolor='black')
    q1_line, = ax.plot([], [], 'g--', label='Q1')
    q2_line, = ax.plot([], [], 'b--', label='Median (Q2)')
    q3_line, = ax.plot([], [], 'r--', label='Q3')
    plt.legend()

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        q1_line.set_data([], [])
        q2_line.set_data([], [])
        q3_line.set_data([], [])
        return scatter, q1_line, q2_line, q3_line

    def update(frame):
        # Randomize data for dynamic quartile visualization
        data = np.random.uniform(10, 50, 12)
        sales = 3 * data + np.random.normal(0, 5, 12)

        # Compute quartiles
        q1 = np.percentile(data, 25)
        q2 = np.percentile(data, 50)
        q3 = np.percentile(data, 75)

        scatter.set_offsets(np.column_stack((data, sales)))
        q1_line.set_data([q1, q1], [0, 200])
        q2_line.set_data([q2, q2], [0, 200])
        q3_line.set_data([q3, q3], [0, 200])
        return scatter, q1_line, q2_line, q3_line

    ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=False)
    plt.show()


if __name__ == "__main__":
    print("Select an option to visualize Advertising Sales Data:")
    print("1. 2D Scatter Plot")
    print("2. 3D Scatter Plot")
    print("3. Animated Quartile Visualization")
    
    plot_2d()
    plot_3d()
    animate_distribution()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Generate random data
np.random.seed(0)
data = np.random.normal(size=(100, 2))

# Fit Kernel Density Estimation model
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)

# Create a grid of points to evaluate the KDE
xgrid, ygrid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
grid_points = np.vstack([xgrid.ravel(), ygrid.ravel()]).T

# Evaluate the density on the grid points
log_dens = kde.score_samples(grid_points)
dens = np.exp(log_dens).reshape(xgrid.shape)

# Plotting the density estimation
plt.figure(figsize=(8, 6))
plt.contourf(xgrid, ygrid, dens, levels=20, cmap='Blues')
plt.scatter(data[:, 0], data[:, 1], color='red', alpha=0.5, label='Data points')
plt.title("2D Kernel Density Estimation")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.colorbar(label="Density")
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.neighbors import KernelDensity

# Generate random 2D data
np.random.seed(0)
data = np.random.normal(size=(100, 2))

# Initialize the Kernel Density Estimation model
kde = KernelDensity(kernel='gaussian')

# Create a grid of points for plotting
xgrid, ygrid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
grid_points = np.vstack([xgrid.ravel(), ygrid.ravel()]).T

# Set up the figure for the plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(data[:, 0], data[:, 1], color='red', alpha=0.5, label='Data points')

# Initialize the contour plot with empty density (just for initialization)
contour = ax.contourf(xgrid, ygrid, np.zeros(xgrid.shape), levels=20, cmap='Blues')

# Add a text box on the plot to explain the animation process
description = ax.text(-2.5, -2.5, '', fontsize=12, color='black', ha='left', va='top', 
                      bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# Function to update the plot in each frame of the animation
def update(i):
    kde.bandwidth = 0.1 + i * 0.05  # Change bandwidth over time
    kde.fit(data)  # Fit the KDE model to the data
    log_dens = kde.score_samples(grid_points)  # Get the log-density for the grid points
    dens = np.exp(log_dens).reshape(xgrid.shape)  # Convert log-density to actual density

    # Remove previous contours from the plot
    for c in ax.collections:
        c.remove()

    # Create new contours based on the updated density
    ax.contourf(xgrid, ygrid, dens, levels=20, cmap='Blues')
    
    # Update the title to reflect the current bandwidth value
    ax.set_title(f"Bandwidth: {kde.bandwidth:.2f}")
    
    # Update the description text
    description.set_text(f"Kernel Density Estimation in action.\nBandwidth is adjusting to smooth the density.\n\n"
                         f"Red points are data, and the contours show density.\n\n"
                         f"Current bandwidth: {kde.bandwidth:.2f}")

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=20, interval=200)

# Display the plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde

# Scenario parameters
lottery_outcomes = [-1, 99]
lottery_probs = [0.99, 0.01]
insurance_outcomes = [1, -99]
insurance_probs = [0.99, 0.01]

# Generate samples for violin plots
def generate_samples(outcomes, probs, n_samples=10000):
    return np.random.choice(outcomes, n_samples, p=probs)

lottery_samples = generate_samples(lottery_outcomes, lottery_probs)
insurance_samples = generate_samples(insurance_outcomes, insurance_probs)

# 1. 2D Violin Plot Comparison
plt.figure(figsize=(10, 6))

# Create violin plots
violin_parts = plt.violinplot([lottery_samples, insurance_samples], 
                              positions=[1, 2], 
                              showmeans=True, 
                              showmedians=True)

# Customize violin plot colors
for pc in violin_parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

# Add labels and titles
plt.xticks([1, 2], ['Lottery', 'Insurance'])
plt.title('2D Violin Plot Comparison', fontsize=14)
plt.xlabel('Scenario', fontsize=12)
plt.ylabel('Outcome ($)', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add annotations for skewness
plt.text(1, 80, f'Skewness: {skew(lottery_samples):.2f}', 
         fontsize=10, ha='center', color='darkblue')
plt.text(2, -80, f'Skewness: {skew(insurance_samples):.2f}', 
         fontsize=10, ha='center', color='darkred')

plt.show()

print("violi plot animation visualization starting...........")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import skew

# Sample data initialization
n_samples = 100
lottery_outcomes = [1, 2, 3, 4, 5]
lottery_probs = [0.1, 0.2, 0.3, 0.2, 0.2]

insurance_outcomes = [0, 1, 2, 3, 4]
insurance_probs = [0.25, 0.25, 0.2, 0.2, 0.1]

# Generate random samples for the plots
def generate_samples(outcomes, probabilities, n_samples):
    return np.random.choice(outcomes, size=n_samples, p=probabilities)

# Initial empty data
lottery_samples = np.zeros(n_samples)
insurance_samples = np.zeros(n_samples)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Initialize violin plot with some sample data
violin_parts = ax.violinplot([lottery_samples, insurance_samples], 
                             positions=[1, 2], showmeans=True, showmedians=True)

# Style initial violins
for pc in violin_parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

# Skewness text
skew_lottery_text = ax.text(1, 100, f'Skewness: {skew(lottery_samples):.2f}', fontsize=10, ha='center', color='darkblue')
skew_insurance_text = ax.text(2, -100, f'Skewness: {skew(insurance_samples):.2f}', fontsize=10, ha='center', color='darkred')

# Explanation text
ax.text(0.5, -0.2, 
                'Violin plots show the distribution density at different values.\n'
                'Wider sections represent more frequent values.',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                fontsize=10)

plt.title('Violin Plots of Lottery and Insurance.\n Samples Violin plots show the distribution density at different values.\n'
                'Wider sections represent more frequent values.', fontsize=14, fontweight='bold', color='darkblue')


# Function to update the animation
def animate(frame):
    global lottery_samples, insurance_samples

    # Generate new samples for this frame
    new_lottery = generate_samples(lottery_outcomes, lottery_probs, n_samples=100)
    new_insurance = generate_samples(insurance_outcomes, insurance_probs, n_samples=100)
    
    # Update data
    lottery_samples = np.concatenate([lottery_samples, new_lottery])
    insurance_samples = np.concatenate([insurance_samples, new_insurance])
    
    # Clear previous violins
    for pc in violin_parts['bodies']:
        pc.remove()
    
    # Create new violins with updated data
    new_violin = ax.violinplot([lottery_samples, insurance_samples], 
                               positions=[1, 2], 
                               showmeans=True, 
                               showmedians=True)
    
    # Update violin parts
    violin_parts['bodies'] = new_violin['bodies']
    violin_parts['cmins'] = new_violin['cmins']
    violin_parts['cmaxes'] = new_violin['cmaxes']
    violin_parts['cmeans'] = new_violin['cmeans']
    violin_parts['cmedians'] = new_violin['cmedians']
    
    # Style violins
    for pc in violin_parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Update skewness annotations
    skew_lottery_text.set_text(f'Skewness: {skew(lottery_samples):.2f}')
    skew_insurance_text.set_text(f'Skewness: {skew(insurance_samples):.2f}')
    
    return violin_parts['bodies']

# Create animation without blit (for compatibility)
ani = FuncAnimation(fig, animate, frames=100, interval=200, blit=False)

# Show the plot
plt.show()
