import matplotlib.pyplot as plt

# Sample heights (in cm) of a population
population_heights = [150, 160, 170, 180, 165, 175, 155, 185, 190, 175]
sample_heights = [160, 170, 175, 180]

plt.figure(figsize=(8, 5))
plt.plot(population_heights, 'o-', label='Population Heights')
plt.plot(range(0, len(sample_heights)), sample_heights, 's--', label='Sample Heights')
plt.title('Population vs Sample Heights')
plt.xlabel('Individuals')
plt.ylabel('Height (cm)')
plt.legend()
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
np.random.seed(42)
heights = np.random.normal(170, 10, 100)
weights = np.random.normal(70, 15, 100)
ages = np.random.normal(30, 5, 100)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(heights, weights, ages, c='green', marker='o')
ax.set_xlabel('Height (cm)')
ax.set_ylabel('Weight (kg)')
ax.set_zlabel('Age (years)')

plt.title('3D Scatter Plot of Height, Weight, and Age')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate sample data
x = np.arange(0, 10)
heights = np.random.normal(170, 5, len(x))

fig, ax = plt.subplots()
line, = ax.plot(x, heights, 'ro-')
ax.set_xlim(0, 10)
ax.set_ylim(150, 190)
ax.set_title('Height Animation over Time')
ax.set_xlabel('Individual')
ax.set_ylabel('Height (cm)')

def update(frame):
    new_heights = np.random.normal(170, 5, len(x))
    line.set_ydata(new_heights)
    return line,

ani = FuncAnimation(fig, update, frames=20, interval=500, blit=True)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Population data (heights in cm)
population = np.array([150, 152, 154, 156, 158, 160, 162, 164, 166, 168])
population_mean = np.mean(population)  # True population mean

# Function to calculate sample means for different sample sizes
def sample_means(population, sample_size, num_samples):
    means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size, replace=False)
        means.append(np.mean(sample))
    return means

# ==================================================
# 1. 2D Plot: Sample Means vs Sample Size
# ==================================================
sample_sizes = [2, 4, 6, 8, 10]
num_samples = 1000  # Number of samples to take for each sample size

plt.figure(figsize=(10, 6))
for size in sample_sizes:
    means = sample_means(population, size, num_samples)
    plt.hist(means, bins=30, alpha=0.5, label=f"Sample Size = {size}")

plt.axvline(population_mean, color="red", linestyle="--", label="Population Mean")
plt.title("Distribution of Sample Means for Different Sample Sizes")
plt.xlabel("Sample Mean (Height in cm)")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()

# ==================================================
# 2. 3D Plot: Sample Means vs Sample Size vs Frequency
# ==================================================
sample_sizes_3d = np.arange(2, 11)  # Sample sizes from 2 to 10
num_samples_3d = 1000  # Number of samples to take for each sample size

# Generate data for 3D plot
x = []
y = []
z = []
for size in sample_sizes_3d:
    means = sample_means(population, size, num_samples_3d)
    x.extend([size] * len(means))
    y.extend(means)
    z.extend(np.random.uniform(0, 1, len(means)))  # Random jitter for visualization

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c=x, cmap="viridis")
ax.set_xlabel("Sample Size")
ax.set_ylabel("Sample Mean (Height in cm)")
ax.set_zlabel("Frequency (Jittered)")
ax.set_title("3D Plot: Sample Means vs Sample Size")
plt.show()

# ==================================================
# 3. Animation: Convergence of Sample Means to Population Mean
# ==================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(2, 10)
ax.set_ylim(140, 180)
ax.axhline(population_mean, color="red", linestyle="--", label="Population Mean")
ax.set_xlabel("Sample Size")
ax.set_ylabel("Sample Mean (Height in cm)")
ax.set_title("Animation: Convergence of Sample Means to Population Mean")
ax.grid()

# Animation function
def animate(frame):
    sample_size = frame + 2  # Sample sizes from 2 to 10
    means = sample_means(population, sample_size, num_samples=100)
    ax.scatter([sample_size] * len(means), means, color="blue", alpha=0.5)
    ax.legend()

ani = FuncAnimation(fig, animate, frames=9, interval=500, repeat=False)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

def create_2d_visualization(n_samples=1000):
    """Create a 2D visualization of the Law of Large Numbers"""
    # Simulate dice rolls (1-4)
    rolls = np.random.randint(1, 5, size=n_samples)
    # Calculate running mean
    running_mean = np.cumsum(rolls) / np.arange(1, len(rolls) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_samples + 1), running_mean, 'b-', label='Sample Mean')
    plt.axhline(y=2.5, color='r', linestyle='--', label='Population Mean')
    plt.xlabel('Number of Samples')
    plt.ylabel('Mean Value')
    plt.title('Law of Large Numbers - 2D Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_3d_visualization(n_samples=1000, n_experiments=50):
    """Create a 3D visualization showing multiple experiments"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate multiple experiments
    for i in range(n_experiments):
        rolls = np.random.randint(1, 5, size=n_samples)
        running_mean = np.cumsum(rolls) / np.arange(1, len(rolls) + 1)
        
        # Create z-offset for each experiment
        z_offset = i * 0.1
        
        # Plot the line
        ax.plot(range(n_samples), running_mean, zs=z_offset, 
                zdir='y', alpha=0.6)
    
    # Add population mean plane
    xx, yy = np.meshgrid(range(n_samples), range(n_experiments))
    zz = np.full_like(xx, 2.5)
    ax.plot_surface(xx, yy*0.1, zz, alpha=0.2, color='r')
    
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Experiment Number')
    ax.set_zlabel('Mean Value')
    ax.set_title('Law of Large Numbers - Multiple Experiments')
    plt.show()

def create_animation(n_frames=200):
    """Create an animated visualization of the Law of Large Numbers"""
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'b-', label='Sample Mean')
    population_mean, = ax.plot([], [], 'r--', label='Population Mean')
    
    ax.set_xlim(0, n_frames)
    ax.set_ylim(1, 4)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Mean Value')
    ax.set_title('Law of Large Numbers - Animation')
    ax.grid(True)
    ax.legend()
    
    rolls = np.random.randint(1, 5, size=n_frames)
    running_mean = np.cumsum(rolls) / np.arange(1, len(rolls) + 1)
    x_data = np.arange(1, n_frames + 1)
    
    def init():
        line.set_data([], [])
        population_mean.set_data([0, n_frames], [2.5, 2.5])
        return line, population_mean
    
    def animate(frame):
        line.set_data(x_data[:frame], running_mean[:frame])
        return line, population_mean
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=20, blit=True)
    plt.show()

# Generate all visualizations
print("Generating 2D visualization...")
create_2d_visualization()

print("\nGenerating 3D visualization...")
create_3d_visualization()

print("\nGenerating animation...")
create_animation()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

def create_pair_outcomes_table():
    """Create and visualize the table of all possible pair outcomes"""
    # All possible pairs for dice with faces 1,2,3,4
    outcomes = []
    averages = []
    for i in range(1, 5):
        for j in range(1, 5):
            outcomes.append((i, j))
            averages.append((i + j) / 2)
    
    # Reshape into 4x4 grid
    avg_matrix = np.array(averages).reshape(4, 4)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[1,2,3,4], yticklabels=[1,2,3,4])
    plt.title('Average Values for All Possible Dice Pair Outcomes')
    plt.xlabel('Second Roll')
    plt.ylabel('First Roll')
    plt.show()

def demonstrate_specific_sequence():
    """Demonstrate the specific sequence mentioned in the text"""
    # Specific sequence mentioned: 4,3, then 3,4,1,3
    specific_numbers = [4, 3, 3, 4, 1, 3]
    n_points = len(specific_numbers)
    
    running_means = np.cumsum(specific_numbers) / np.arange(1, len(specific_numbers) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_points + 1), running_means, 'bo-', label='Sample Mean')
    plt.axhline(y=2.5, color='r', linestyle='--', label='Population Mean (2.5)')
    
    # Annotate specific points
    for i, (mean, numbers) in enumerate(zip(running_means, specific_numbers)):
        plt.annotate(f'Mean: {mean:.2f}\nNew number: {numbers}', 
                    (i + 1, mean), xytext=(10, 10),
                    textcoords='offset points')
    
    plt.xlabel('Number of Samples')
    plt.ylabel('Running Mean')
    plt.title('Convergence of Sample Mean - Specific Sequence')
    plt.grid(True)
    plt.legend()
    plt.show()

def simulate_multiple_sequences(n_sequences=50, n_samples=100):
    """Simulate multiple sequences to show general convergence"""
    plt.figure(figsize=(12, 6))
    
    # Generate multiple sequences
    for _ in range(n_sequences):
        samples = np.random.randint(1, 5, size=n_samples)
        running_means = np.cumsum(samples) / np.arange(1, len(samples) + 1)
        plt.plot(range(1, n_samples + 1), running_means, 'b-', alpha=0.1)
    
    # Add population mean
    plt.axhline(y=2.5, color='r', linestyle='--', label='Population Mean (2.5)')
    
    # Add confidence intervals
    samples = np.random.randint(1, 5, size=(1000, n_samples))
    running_means = np.cumsum(samples, axis=1) / np.arange(1, n_samples + 1)
    
    plt.fill_between(range(1, n_samples + 1),
                     np.percentile(running_means, 5, axis=0),
                     np.percentile(running_means, 95, axis=0),
                     color='gray', alpha=0.2, label='90% Confidence Interval')
    
    plt.xlabel('Number of Samples')
    plt.ylabel('Running Mean')
    plt.title('Multiple Sequences Showing Law of Large Numbers')
    plt.grid(True)
    plt.legend()
    plt.show()

# Generate all visualizations
print("1. Generating pair outcomes table...")
create_pair_outcomes_table()

print("\n2. Demonstrating specific sequence...")
demonstrate_specific_sequence()

print("\n3. Simulating multiple sequences...")
simulate_multiple_sequences()


import numpy as np
import matplotlib.pyplot as plt

# Dataset 1 (Low variance)
dataset1 = np.array([160, 162, 159, 161, 160])

# Dataset 2 (High variance)
dataset2 = np.array([130, 180, 140, 210, 120])

# Calculate means
mean1 = np.mean(dataset1)
mean2 = np.mean(dataset2)

# Plotting
plt.figure(figsize=(8, 6))

# Dataset 1 (Low Variance)
plt.subplot(1, 2, 1)
plt.scatter(np.arange(len(dataset1)), dataset1, color='blue', label='Data Points')
plt.axhline(mean1, color='red', linestyle='--', label=f'Mean: {mean1}')
plt.title('Low Variance')
plt.xlabel('Index')
plt.ylabel('Height')
plt.legend()

# Dataset 2 (High Variance)
plt.subplot(1, 2, 2)
plt.scatter(np.arange(len(dataset2)), dataset2, color='green', label='Data Points')
plt.axhline(mean2, color='red', linestyle='--', label=f'Mean: {mean2}')
plt.title('High Variance')
plt.xlabel('Index')
plt.ylabel('Height')
plt.legend()

plt.tight_layout()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Generate random 3D data (Sample Variance)
np.random.seed(42)
points = np.random.randn(30, 3) * 5 + np.array([10, 15, 20])

# Calculate mean for each axis (dimension)
mean = np.mean(points, axis=0)

# Plotting 3D scatter
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='Data Points')

# Plotting the mean
ax.scatter(mean[0], mean[1], mean[2], color='red', s=100, label='Mean', marker='X')

# Lines from points to the mean
for point in points:
    ax.plot([point[0], mean[0]], [point[1], mean[1]], [point[2], mean[2]], color='gray', linestyle=':')

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Sample Variance')
ax.legend()

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import binom, norm

# Parameters
n_coin_flips = 10  # Number of coin flips (Bernoulli trials)
p = 0.5  # Probability of heads
sample_sizes = [1, 2, 3, 4, 10]  # Sample sizes to visualize
num_samples = 10000  # Number of samples to generate for each sample size

# ==================================================
# 1. 2D Plot: Distribution of Sample Means
# ==================================================
plt.figure(figsize=(10, 6))
for n in sample_sizes:
    # Generate samples of size n and calculate their means
    sample_means = [np.mean(binom.rvs(n, p, size=n_coin_flips)) for _ in range(num_samples)]
    plt.hist(sample_means, bins=30, density=True, alpha=0.5, label=f"n = {n}")

# Plot the theoretical normal distribution
x = np.linspace(0, n_coin_flips, 1000)
mu = n_coin_flips * p  # Mean of the binomial distribution
sigma = np.sqrt(n_coin_flips * p * (1 - p))  # Standard deviation of the binomial distribution
plt.plot(x, norm.pdf(x, mu, sigma), color="red", linestyle="--", label="Normal Distribution")

plt.title("Central Limit Theorem: Distribution of Sample Means")
plt.xlabel("Sample Mean (Number of Heads)")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()

# ==================================================
# 2. 3D Plot: Distribution of Sample Means vs Sample Size
# ==================================================
sample_sizes_3d = np.arange(1, 11)  # Sample sizes from 1 to 10
num_samples_3d = 1000  # Number of samples to generate for each sample size

# Generate data for 3D plot
x = []
y = []
z = []
for n in sample_sizes_3d:
    sample_means = [np.mean(binom.rvs(n, p, size=n_coin_flips)) for _ in range(num_samples_3d)]
    x.extend([n] * len(sample_means))
    y.extend(sample_means)
    z.extend(np.random.uniform(0, 1, len(sample_means)))  # Random jitter for visualization

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c=x, cmap="viridis")
ax.set_xlabel("Sample Size (n)")
ax.set_ylabel("Sample Mean (Number of Heads)")
ax.set_zlabel("Frequency (Jittered)")
ax.set_title("3D Plot: Sample Means vs Sample Size")
plt.show()

# ==================================================
# 3. Animation: Convergence of Sample Means to Normal Distribution
# ==================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, n_coin_flips)
ax.set_ylim(0, 0.5)
ax.set_xlabel("Sample Mean (Number of Heads)")
ax.set_ylabel("Density")
ax.set_title("Animation: Convergence of Sample Means to Normal Distribution")
ax.grid()

# Animation function
def animate(frame):
    n = frame + 1  # Sample sizes from 1 to 10
    sample_means = [np.mean(binom.rvs(n, p, size=n_coin_flips)) for _ in range(num_samples)]
    ax.clear()
    ax.hist(sample_means, bins=30, density=True, alpha=0.5, label=f"n = {n}")
    ax.plot(x, norm.pdf(x, mu, sigma), color="red", linestyle="--", label="Normal Distribution")
    ax.set_xlim(0, n_coin_flips)
    ax.set_ylim(0, 0.5)
    ax.legend()
    ax.grid()

ani = FuncAnimation(fig, animate, frames=10, interval=500, repeat=False)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# Set random seed for reproducibility
np.random.seed(42)

class VarianceAnimation:
    def __init__(self, n_frames=200):
        # Setup the figure with two subplots
        self.fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 1, height_ratios=[2, 1])
        
        # Top plot for scatter and mean
        self.ax1 = self.fig.add_subplot(gs[0])
        self.scatter = self.ax1.scatter([], [], alpha=0.5)
        self.mean_line = self.ax1.axvline(x=2.5, color='r', linestyle='--', label='Population Mean')
        self.sample_mean_line = self.ax1.axvline(x=0, color='g', linestyle='-', label='Sample Mean')
        
        # Bottom plot for variance convergence
        self.ax2 = self.fig.add_subplot(gs[1])
        self.variance_line, = self.ax2.plot([], [], 'b-', label='Sample Variance')
        self.pop_var_line = self.ax2.axhline(y=1.25, color='r', linestyle='--', label='Population Variance')
        
        # Generate data (using dice rolls 1-4 as in previous example)
        self.data = np.random.randint(1, 5, size=n_frames)
        self.n_frames = n_frames
        
        # Initialize arrays for storing means and variances
        self.means = np.zeros(n_frames)
        self.variances = np.zeros(n_frames)
        
        # Setup plots
        self.setup_plots()
        
        # Create animation
        self.anim = FuncAnimation(
            self.fig, self.update, frames=n_frames,
            init_func=self.init_animation, blit=True, interval=50
        )
    
    def setup_plots(self):
        # Setup top plot
        self.ax1.set_xlim(0, 5)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_title('Sample Distribution and Mean')
        self.ax1.set_xlabel('Value')
        self.ax1.set_ylabel('Density')
        self.ax1.legend()
        
        # Setup bottom plot
        self.ax2.set_xlim(0, self.n_frames)
        self.ax2.set_ylim(0, 2.5)
        self.ax2.set_title('Sample Variance Convergence')
        self.ax2.set_xlabel('Number of Samples')
        self.ax2.set_ylabel('Variance')
        self.ax2.legend()
        self.ax2.grid(True)
        
        plt.tight_layout()
    
    def init_animation(self):
        self.scatter.set_offsets(np.empty((0, 2)))
        self.sample_mean_line.set_xdata([0])
        self.variance_line.set_data([], [])
        return self.scatter, self.sample_mean_line, self.variance_line
    
    def update(self, frame):
        # Update data points
        current_data = self.data[:frame+1]
        
        # Calculate current mean and variance
        current_mean = np.mean(current_data)
        if frame > 0:
            current_var = np.var(current_data, ddof=1)  # Using sample variance
        else:
            current_var = 0
            
        self.means[frame] = current_mean
        self.variances[frame] = current_var
        
        # Update scatter plot
        y_positions = np.random.random(size=len(current_data)) * 0.8
        self.scatter.set_offsets(np.column_stack((current_data, y_positions)))
        
        # Update mean line
        self.sample_mean_line.set_xdata([current_mean, current_mean])
        self.sample_mean_line.set_ydata([0, 1])
        
        # Update variance plot
        x = np.arange(frame + 1)
        self.variance_line.set_data(x, self.variances[:frame+1])
        
        return self.scatter, self.sample_mean_line, self.variance_line

# Create and show animation
anim = VarianceAnimation(n_frames=200)
plt.show()

# Save animation (optional)
# anim.anim.save('variance_animation.gif', writer='pillow')