import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Data for visualization
np.random.seed(42)
population_mean = 50
population_std = 10

# Function to generate sample means
def sample_means(sample_size, num_samples=100):
    return [np.mean(np.random.normal(population_mean, population_std, sample_size)) for _ in range(num_samples)]

# Prepare figure and axis
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Data placeholders for the animation
sample_sizes = [1, 2, 10]
colors = ['skyblue', 'lightgreen', 'coral']
intervals_2d = []
intervals_3d = []

# Initialize the plots
def init():
    ax[0].set_xlim(0, 100)
    ax[0].set_ylim(population_mean - 30, population_mean + 30)
    ax[1].set_xlim(0, 100)
    ax[1].set_ylim(population_mean - 30, population_mean + 30)
    ax[0].set_title("2D: Confidence Intervals for Sample Size n=1, 2, 10")
    ax[1].set_title("3D: Sequential Confidence Intervals Over Time")
    return ax

# Function for animation
def update(frame):
    ax[0].clear()
    ax[1].clear()

    ax[0].set_title("2D: Confidence Intervals for Sample Size n=1, 2, 10")
    ax[1].set_title("3D: Sequential Confidence Intervals Over Time")
    ax[0].set_xlim(0, 100)
    ax[0].set_ylim(population_mean - 30, population_mean + 30)
    ax[1].set_xlim(0, 100)
    ax[1].set_ylim(population_mean - 30, population_mean + 30)

    for i, n in enumerate(sample_sizes):
        means = sample_means(n, num_samples=frame + 1)
        lower_bound = [mean - 1.96 * (population_std / np.sqrt(n)) for mean in means]
        upper_bound = [mean + 1.96 * (population_std / np.sqrt(n)) for mean in means]
        
        # Plot 2D sample means with confidence intervals
        ax[0].plot(range(len(means)), means, 'o-', color=colors[i], label=f"n={n}")
        ax[0].fill_between(range(len(means)), lower_bound, upper_bound, color=colors[i], alpha=0.2)

        # Plot 3D-like sequential intervals
        ax[1].plot(range(len(means)), means, 'o-', color=colors[i])
        ax[1].fill_between(range(len(means)), lower_bound, upper_bound, color=colors[i], alpha=0.1)

    ax[0].legend()
    return ax

# Create animation
ani = FuncAnimation(fig, update, frames=20, init_func=init, blit=False)

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Generate data for normal distribution with varying sample sizes
mu = 0  # Mean of the distribution
sigma = 1  # Standard deviation of the population
sample_sizes = [1, 5, 10, 30, 50, 100]  # Different sample sizes
confidence_level = 1.96  # For 95% confidence interval

# Colors for confidence intervals
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

# Create 2D Plot: Normal distribution curves with shrinking confidence intervals
plt.figure(figsize=(10, 6))
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)

for i, n in enumerate(sample_sizes):
    sample_std = sigma / np.sqrt(n)
    pdf = (1 / (np.sqrt(2 * np.pi) * sample_std)) * np.exp(-0.5 * ((x - mu) / sample_std) ** 2)
    plt.plot(x, pdf, color=colors[i], label=f'n={n}')
    lower_bound = mu - confidence_level * sample_std
    upper_bound = mu + confidence_level * sample_std
    plt.axvline(lower_bound, color=colors[i], linestyle='--')
    plt.axvline(upper_bound, color=colors[i], linestyle='--')

plt.title('2D Normal Distributions with Confidence Intervals')
plt.xlabel('Sample Mean')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# 3D Plot: Stacked normal distributions with shrinking intervals
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)

for i, n in enumerate(sample_sizes):
    sample_std = sigma / np.sqrt(n)
    pdf = (1 / (np.sqrt(2 * np.pi) * sample_std)) * np.exp(-0.5 * ((x - mu) / sample_std) ** 2)
    y = np.ones_like(x) * n  # Stacking sample sizes on the y-axis
    ax.plot(x, y, pdf, color=colors[i], label=f'n={n}')

ax.set_title('3D Confidence Interval Visualization')
ax.set_xlabel('Sample Mean')
ax.set_ylabel('Sample Size (n)')
ax.set_zlabel('Probability Density')
plt.legend()
plt.show()

# Animation: Shrinking curves as sample size increases
fig, ax = plt.subplots(figsize=(10, 6))

# Set up plot limits
ax.set_xlim(mu - 4 * sigma, mu + 4 * sigma)
ax.set_ylim(0, 1)
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
line, = ax.plot([], [], lw=2)

# Initialization function
def init():
    line.set_data([], [])
    return line,

# Animation function
def animate(i):
    n = sample_sizes[i % len(sample_sizes)]
    sample_std = sigma / np.sqrt(n)
    pdf = (1 / (np.sqrt(2 * np.pi) * sample_std)) * np.exp(-0.5 * ((x - mu) / sample_std) ** 2)
    line.set_data(x, pdf)
    ax.set_title(f'n={n}')
    return line,

ani = FuncAnimation(fig, animate, init_func=init, frames=len(sample_sizes), interval=1000, blit=True)

plt.title('Shrinking Confidence Interval Animation')
plt.xlabel('Sample Mean')
plt.ylabel('Probability Density')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

# Set Seaborn style for better visuals
sns.set(style="whitegrid")

# Define population parameters
mu = 50  # Population mean
sigma = 10  # Population standard deviation

# Sample sizes for illustration
sample_sizes = [1, 2, 5, 10, 30, 40, 50]
confidence_level = 0.95

# Function to calculate margin of error
def margin_of_error(sigma, n, z_value):
    return z_value * (sigma / np.sqrt(n))

# Z-value for 95% confidence level
z_value = 1.96

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 6))
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)

# Plot the base population distribution
sns.lineplot(x=x, y=1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2), color='gray', lw=2)
plt.axvline(mu, color='black', linestyle='--', label=f'Population Mean (\u03BC={mu})')
plt.title('Confidence Interval Shrinking as Sample Size Increases')
plt.xlabel('Value')
plt.ylabel('Probability Density')

# Animation function
def update(frame):
    ax.clear()
    sample_size = sample_sizes[frame]
    # Compute new standard error and margin of error
    std_error = sigma / np.sqrt(sample_size)
    margin_error = z_value * std_error
    
    # Plot distribution with narrower standard error
    y = 1/(std_error * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std_error)**2)
    sns.lineplot(x=x, y=y, color='blue', lw=2, ax=ax)
    
    # Plot confidence interval
    ci_lower = mu - margin_error
    ci_upper = mu + margin_error

    ax.axvline(mu, color='black', linestyle='--', label=f'Mean (\u03BC={mu})')
    ax.axvline(ci_lower, color='red', linestyle='--', label=f'Lower CI ({ci_lower:.2f})')
    ax.axvline(ci_upper, color='green', linestyle='--', label=f'Upper CI ({ci_upper:.2f})')

    ax.fill_between(x, y, where=(x >= ci_lower) & (x <= ci_upper), color='skyblue', alpha=0.3)

    ax.set_title(f'Confidence Interval with Sample Size n={sample_size}', fontsize=14)
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.set_xlim(mu - 4 * sigma, mu + 4 * sigma)
    ax.set_ylim(0, 0.1)

# Create and save animation
ani = FuncAnimation(fig, update, frames=len(sample_sizes), repeat=False)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mu = 0  # Mean
sigma = 1  # Standard deviation
alpha = 0.05  # Significance level for 95% CI
z_score = norm.ppf(1 - alpha / 2)  # Z-score for 95% CI (approximately 1.96)

# Generate x values (range of data points)
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, mu, sigma)  # PDF of the normal distribution

# Plot the normal distribution curve
plt.plot(x, y, label="Normal Distribution", color='blue')

# Fill the areas within the confidence interval (95% CI)
plt.fill_between(x, y, where=(x >= -z_score) & (x <= z_score), color='orange', alpha=0.5, label="95% Confidence Interval")

# Labels and titles
plt.title('Normal Distribution with 95% Confidence Interval')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Show the plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

# Create a grid for plotting
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)

# Define the parameters for the normal distribution
mu = 0
sigma = 1

# Calculate the Z-scores for varying confidence levels (alpha) and sample sizes (n)
alpha_values = [0.01, 0.05, 0.1]  # Different significance levels
sample_sizes = [10, 50, 100]  # Different sample sizes

# Create the figure for 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Loop over sample sizes and confidence levels
for alpha in alpha_values:
    for n in sample_sizes:
        # Calculate Z-score for the given alpha (CI)
        z_score = norm.ppf(1 - alpha / 2)
        
        # Create the Gaussian distribution for each sample size
        Z = norm.pdf(X, mu, sigma / np.sqrt(n))
        
        # Plot the surface for each combination
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

# Labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Density')
ax.set_title('3D Plot of Normal Distribution with Different Z-scores and Sample Sizes')

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

# Parameters for the animation
mu = 0
sigma = 1
alpha = 0.05
sample_size = 30  # Starting sample size

# Create the figure for animation
fig, ax = plt.subplots(figsize=(8, 6))

# Create an empty plot for the normal distribution
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, mu, sigma)

# Initial plot of normal distribution
line, = ax.plot(x, y, color='blue', label='Normal Distribution')

# Animation update function
def update(frame):
    sample_size = frame + 10  # Increase sample size dynamically
    z_score = norm.ppf(1 - alpha / 2)  # Z-score for 95% CI
    margin_of_error = z_score * (sigma / np.sqrt(sample_size))  # Margin of error
    ci_lower = mu - margin_of_error
    ci_upper = mu + margin_of_error

    ax.clear()
    ax.plot(x, y, color='blue', label='Normal Distribution')
    
    # Highlight the CI area
    ax.fill_between(x, y, where=(x >= ci_lower) & (x <= ci_upper), color='orange', alpha=0.5, label='Confidence Interval')
    
    ax.axvline(ci_lower, color='red', linestyle='--', label='Lower CI')
    ax.axvline(ci_upper, color='red', linestyle='--', label='Upper CI')
    
    ax.set_title(f'Sample Size: {sample_size} | CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=30, repeat=False)

# Show the animation
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for normal distribution
mu = 50
sigma = 10
data = np.random.normal(mu, sigma, 1000)

# Z-score normalization
z_scores = (data - mu) / sigma

# Create the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the original normal distribution
ax1.hist(data, bins=30, density=True, color='blue', alpha=0.7)
xmin, xmax = ax1.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)
ax1.plot(x, p, 'k', linewidth=2)
ax1.set_title('Original Data Distribution')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')

# Plot the Z-score normalized distribution
ax2.hist(z_scores, bins=30, density=True, color='orange', alpha=0.7)
ax2.set_title('Z-Score Normalized Data')
ax2.set_xlabel('Z-Score')
ax2.set_ylabel('Density')

plt.tight_layout()
plt.show()


from sklearn.preprocessing import MinMaxScaler

# Min-Max normalization
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Create the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the original data distribution
ax1.hist(data, bins=30, density=True, color='blue', alpha=0.7)
ax1.set_title('Original Data Distribution')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')

# Plot the Min-Max normalized data
ax2.hist(normalized_data, bins=30, density=True, color='green', alpha=0.7)
ax2.set_title('Min-Max Normalized Data (Range: [0, 1])')
ax2.set_xlabel('Normalized Value')
ax2.set_ylabel('Density')

plt.tight_layout()
plt.show()


# Apply Z-score normalization first
z_score_normalized = (data - mu) / sigma

# Then apply Min-Max normalization on the Z-score normalized data
final_normalized_data = scaler.fit_transform(z_score_normalized.reshape(-1, 1)).flatten()

# Create the figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

# Plot original data
ax1.hist(data, bins=30, density=True, color='blue', alpha=0.7)
ax1.set_title('Original Data Distribution')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')

# Plot Z-score normalized data
ax2.hist(z_score_normalized, bins=30, density=True, color='orange', alpha=0.7)
ax2.set_title('Z-Score Normalized Data')
ax2.set_xlabel('Z-Score')
ax2.set_ylabel('Density')

# Plot Min-Max normalized data
ax3.hist(final_normalized_data, bins=30, density=True, color='green', alpha=0.7)
ax3.set_title('Z-Score + Min-Max Normalized Data')
ax3.set_xlabel('Normalized Value')
ax3.set_ylabel('Density')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Generate some random data for 3D plot
data = np.random.normal(50, 10, (100, 3))  # 100 points, 3 features

# Apply Z-score normalization
z_score_normalized = (data - data.mean(axis=0)) / data.std(axis=0)

# Apply Min-Max normalization
scaler = MinMaxScaler()
min_max_normalized = scaler.fit_transform(data)

# Create the figure and subplots
fig = plt.figure(figsize=(14, 6))

# Choose the plot type (replace with condition or input)
plot_type = "z-score"  # Can be "original", "z-score", or "min-max"

# Plot the original data in 3D
if plot_type == "original":
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], color='blue', alpha=0.7)
    ax1.set_title('Original Data')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')

# Plot the Z-score normalized data in 3D
elif plot_type == "z-score":
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(z_score_normalized[:, 0], z_score_normalized[:, 1], z_score_normalized[:, 2], color='orange', alpha=0.7)
    ax2.set_title('Z-Score Normalized Data')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_zlabel('Feature 3')

# Plot the Min-Max normalized data in 3D
elif plot_type == "min-max":
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(min_max_normalized[:, 0], min_max_normalized[:, 1], min_max_normalized[:, 2], color='green', alpha=0.7)
    ax3.set_title('Min-Max Normalized Data')
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    ax3.set_zlabel('Feature 3')

plt.tight_layout()
plt.show()
