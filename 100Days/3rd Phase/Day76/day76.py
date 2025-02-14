import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Sampling and Population
def sampling_population():
    population = np.random.normal(loc=50, scale=15, size=1000)  # population data
    sample = np.random.choice(population, size=100, replace=False)  # sample data
    
    plt.figure(figsize=(10, 6))
    plt.hist(population, bins=30, alpha=0.6, label='Population', color='blue')
    plt.hist(sample, bins=30, alpha=0.6, label='Sample', color='green')
    plt.title("Sampling vs Population")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Empirical Mean
def empirical_mean():
    data = np.random.normal(50, 15, 1000)  # sample data
    mean = np.mean(data)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.6, label='Data', color='orange')
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Empirical Mean: {mean:.2f}')
    plt.title("Empirical Mean")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Descriptive vs Inferential Statistics
def descriptive_inferential():
    population = np.random.normal(50, 15, 1000)
    sample = np.random.choice(population, size=100, replace=False)
    
    # Descriptive statistics
    pop_mean = np.mean(population)
    pop_std = np.std(population)
    
    # Inferential statistics (estimating population mean from sample)
    sample_mean = np.mean(sample)
    
    plt.figure(figsize=(10, 6))
    plt.hist(population, bins=30, alpha=0.5, label="Population", color='blue')
    plt.hist(sample, bins=30, alpha=0.5, label="Sample", color='green')
    plt.axvline(pop_mean, color='red', linestyle='dashed', linewidth=2, label=f"Population Mean: {pop_mean:.2f}")
    plt.axvline(sample_mean, color='yellow', linestyle='dashed', linewidth=2, label=f"Sample Mean: {sample_mean:.2f}")
    
    plt.title("Descriptive vs Inferential Statistics")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Binomial Distribution
def binomial_distribution():
    n = 10  # number of trials
    p = 0.5  # probability of success
    x = np.arange(0, n+1)
    pmf = binom.pmf(x, n, p)
    
    plt.figure(figsize=(10, 6))
    plt.stem(x, pmf, basefmt=" ", use_line_collection=True, label=f'Binomial Distribution (n={n}, p={p})', linefmt='green')
    plt.title("Binomial Distribution PMF")
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

# Standard Deviation in Normal Distribution
def std_dev_normal_distribution():
    mean = 0
    std_dev = 1
    x = np.linspace(-5, 5, 100)
    y = norm.pdf(x, mean, std_dev)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f"Normal Distribution (Mean={mean}, Std={std_dev})", color='purple')
    plt.fill_between(x, y, alpha=0.3, color='purple')
    
    # Show std deviation bands
    plt.axvline(mean + std_dev, color='red', linestyle='dashed', label=f"+1 Std Dev")
    plt.axvline(mean - std_dev, color='red', linestyle='dashed', label=f"-1 Std Dev")
    
    plt.title("Standard Deviation in Normal Distribution")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# 3D Plot of Normal Distribution
def normal_distribution_3d():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid for 3D plotting
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)
    z = norm.pdf(x) * norm.pdf(y)
    
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    ax.set_title("3D Normal Distribution")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Density")
    plt.show()

# Animation of Binomial Distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def binomial_distribution():
    n = 10  # number of trials
    p = 0.5  # probability of success

    # x values for the number of successes
    x = np.arange(0, n + 1)

    # Probability mass function (pmf) of the binomial distribution
    pmf = binom.pmf(x, n, p)

    # Create stem plot without the 'use_line_collection' argument
    plt.stem(x, pmf, basefmt=" ", label=f'Binomial Distribution (n={n}, p={p})', linefmt='green')

    # Adding labels and title
    plt.xlabel('Number of successes')
    plt.ylabel('Probability')
    plt.title(f'Binomial Distribution (n={n}, p={p})')

    # Adding a legend
    plt.legend()

    # Show the plot
    plt.show()

# Call the function
binomial_distribution()


# Run all visualizations
sampling_population()
empirical_mean()
descriptive_inferential()
binomial_distribution()
std_dev_normal_distribution()
normal_distribution_3d()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Generate random data for demonstration
np.random.seed(42)
data = np.random.normal(0, 1, (100, 2))  # 100 points, 2 features
data[:, 0] = data[:, 0] * 10  # Scale the first feature to show effect of standardization
data[:, 1] = data[:, 1] + 5   # Add offset to the second feature

# Standardize the data (Z-score transformation)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
standardized_data = (data - mean) / std

# 2D Plot: Original data vs standardized data
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Original Data
ax[0].scatter(data[:, 0], data[:, 1], color='red', alpha=0.6)
ax[0].set_title('Original Data')
ax[0].set_xlabel('Feature 1')
ax[0].set_ylabel('Feature 2')

# Standardized Data
ax[1].scatter(standardized_data[:, 0], standardized_data[:, 1], color='blue', alpha=0.6)
ax[1].set_title('Standardized Data (Z-Score)')
ax[1].set_xlabel('Feature 1 (Standardized)')
ax[1].set_ylabel('Feature 2 (Standardized)')

plt.tight_layout()
plt.show()

# 3D Plot: Showing effect of standardization on 3D data
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Generate 3D data
data_3d = np.random.normal(0, 1, (100, 3))
data_3d[:, 0] = data_3d[:, 0] * 10
data_3d[:, 1] = data_3d[:, 1] + 5
data_3d[:, 2] = data_3d[:, 2] * 2

# Standardize 3D data
mean_3d = np.mean(data_3d, axis=0)
std_3d = np.std(data_3d, axis=0)
standardized_3d = (data_3d - mean_3d) / std_3d

# Plot original 3D data
ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], color='red', label='Original Data')
# Plot standardized 3D data
ax.scatter(standardized_3d[:, 0], standardized_3d[:, 1], standardized_3d[:, 2], color='blue', label='Standardized Data')

ax.set_title('Effect of Standardization on 3D Data')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend()
plt.show()

# Animation: Showing the standardization process
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
line_original, = ax.plot([], [], 'ro', label='Original Data')
line_standardized, = ax.plot([], [], 'bo', label='Standardized Data')
ax.set_title("Animating the Standardization Process")
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

def init():
    line_original.set_data([], [])
    line_standardized.set_data([], [])
    return line_original, line_standardized,

def animate(i):
    # Plotting original data points (in red)
    line_original.set_data(data[:i, 0], data[:i, 1])
    # Standardize and plot (in blue)
    standardized_partial = (data[:i] - mean) / std
    line_standardized.set_data(standardized_partial[:, 0], standardized_partial[:, 1])
    return line_original, line_standardized,

ani = animation.FuncAnimation(fig, animate, frames=len(data), init_func=init, blit=True, interval=50)
plt.legend()
plt.show()
