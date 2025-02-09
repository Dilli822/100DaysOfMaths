import numpy as np
import matplotlib.pyplot as plt

# Generate random data
data = np.random.normal(loc=5, scale=2, size=1000)  # Mean=5, Std=2

mean = np.mean(data)
variance = np.var(data)

plt.hist(data, bins=30, alpha=0.6, color='skyblue', edgecolor='black')
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
plt.title('Histogram with Mean Line')
plt.xlabel('Data Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()



# Generate data for multiple random variables
data1 = np.random.normal(2, 1, 1000)
data2 = np.random.normal(4, 1.5, 1000)
data3 = np.random.normal(6, 0.5, 1000)

sample_mean = (data1 + data2 + data3) / 3

plt.plot(sample_mean[:100], marker='o', linestyle='', color='purple')
plt.axhline(np.mean(sample_mean), color='red', linestyle='dashed', label=f'Sample Mean: {np.mean(sample_mean):.2f}')
plt.title('Sample Mean Visualization')
plt.legend()
plt.show()



from scipy.stats import bernoulli, binom

# Bernoulli Plot
bernoulli_data = bernoulli.rvs(p=0.3, size=1000)
plt.hist(bernoulli_data, bins=2, color='skyblue', edgecolor='black')
plt.title('Bernoulli Distribution (p=0.3)')
plt.show()

# Binomial Plot
binomial_data = binom.rvs(n=10, p=0.5, size=1000)
plt.hist(binomial_data, bins=10, color='orange', edgecolor='black')
plt.title('Binomial Distribution (n=10, p=0.5)')
plt.show()


from mpl_toolkits.mplot3d import Axes3D

x = np.random.normal(0, 1, 500)
y = np.random.normal(5, 2, 500)
z = x + y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='purple', marker='o')
ax.set_title('3D Plot of Multiple Random Variables')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate random data for sample mean calculation
data = np.random.normal(0, 1, 1000)  # Mean=0, Std=1
sample_means = []

fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot([], [], lw=2, color='purple')
mean_text = ax.text(0.7, 0.9, '', transform=ax.transAxes, fontsize=12, color='darkgreen')

# Initialization function
def init():
    ax.set_xlim(0, 50)
    ax.set_ylim(-2, 2)
    ax.set_title("Sample Mean Animation", fontsize=14)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Sample Mean")
    ax.grid(True, linestyle='--', alpha=0.6)
    return line, mean_text

# Update function
def update(frame):
    sample_means.append(np.mean(data[:frame + 1]))
    line.set_data(range(len(sample_means)), sample_means)
    mean_text.set_text(f'Mean So Far: {np.mean(sample_means):.2f}')
    return line, mean_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=50, init_func=init, blit=True)

# Show the plot with details
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate random data for sample variance calculation
data = np.random.normal(0, 1, 1000)  # Mean=0, Std=1
sample_variances = []

fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot([], [], lw=2, color='orange')
variance_text = ax.text(0.7, 0.9, '', transform=ax.transAxes, fontsize=12, color='darkblue')

# Initialization function
def init():
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 2.5)  # Variance typically positive
    ax.set_title("Sample Variance Animation", fontsize=14)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Sample Variance")
    ax.grid(True, linestyle='--', alpha=0.6)
    return line, variance_text

# Update function
def update(frame):
    sample_variances.append(np.var(data[:frame + 1]))
    line.set_data(range(len(sample_variances)), sample_variances)
    variance_text.set_text(f'Variance So Far: {np.var(data[:frame + 1]):.2f}')
    return line, variance_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=50, init_func=init, blit=True)

# Show the plot with details
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate random data for mean and variance calculations
data = np.random.normal(0, 1, 1000)  # Mean=0, Std=1
sample_means = []
sample_variances = []

fig, ax = plt.subplots(figsize=(8, 5))
mean_line, = ax.plot([], [], lw=2, color='purple', label='Sample Mean')
variance_line, = ax.plot([], [], lw=2, color='orange', label='Sample Variance')

mean_text = ax.text(0.6, 0.9, '', transform=ax.transAxes, fontsize=12, color='darkgreen')
variance_text = ax.text(0.6, 0.85, '', transform=ax.transAxes, fontsize=12, color='darkblue')

# Initialization function
def init():
    ax.set_xlim(0, 50)
    ax.set_ylim(-1, 3)
    ax.set_title("Mean vs Variance Animation", fontsize=14)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    return mean_line, variance_line, mean_text, variance_text

# Update function
def update(frame):
    sample_means.append(np.mean(data[:frame + 1]))
    sample_variances.append(np.var(data[:frame + 1]))
    
    mean_line.set_data(range(len(sample_means)), sample_means)
    variance_line.set_data(range(len(sample_variances)), sample_variances)
    
    mean_text.set_text(f'Mean: {np.mean(sample_means):.2f}')
    variance_text.set_text(f'Variance: {np.var(data[:frame + 1]):.2f}')
    
    return mean_line, variance_line, mean_text, variance_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=50, init_func=init, blit=True)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Generate random data for expectation calculation (discrete outcomes)
np.random.seed(0)
outcomes = np.array([1, 2, 3, 4, 5])  # Discrete outcomes
probabilities = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # Probabilities sum to 1
data = np.random.choice(outcomes, 1000, p=probabilities)  # Generate random samples based on the probabilities

cumulative_expectation = []

fig = plt.figure(figsize=(12, 6))

# Create a 2D plot on the left
ax2d = fig.add_subplot(1, 2, 1)
line2d, = ax2d.plot([], [], lw=2, color='teal')
expectation_text = ax2d.text(0.6, 0.9, '', transform=ax2d.transAxes, fontsize=12, color='darkblue')

# Theoretical expectation
expected_value_theoretical = np.sum(outcomes * probabilities)

ax2d.set_xlim(0, 50)
ax2d.set_ylim(0, 5)
ax2d.set_title("2D Expectation Visualization", fontsize=14)
ax2d.set_xlabel("Number of Samples")
ax2d.set_ylabel("Expectation Value")
ax2d.axhline(y=expected_value_theoretical, color='red', linestyle='--', label=f'Theoretical E[X] = {expected_value_theoretical:.2f}')
ax2d.legend()
ax2d.grid(True, linestyle='--', alpha=0.6)

# Create a 3D plot on the right
ax3d = fig.add_subplot(1, 2, 2, projection='3d')
ax3d.set_title("3D Expectation Visualization")
ax3d.set_xlim(0, 50)
ax3d.set_ylim(1, 5)
ax3d.set_zlim(0, 5)
ax3d.set_xlabel("Samples")
ax3d.set_ylabel("Empirical Expectation")
ax3d.set_zlabel("Theoretical Expectation")

# Initialize 3D plot elements
line3d, = ax3d.plot([], [], [], lw=2, color='purple')

def init():
    return line2d, line3d, expectation_text

# Update function for animation
def update(frame):
    cumulative_expectation.append(np.mean(data[:frame + 1]))

    # Update the 2D plot
    line2d.set_data(range(len(cumulative_expectation)), cumulative_expectation)
    expectation_text.set_text(f'Empirical E[X]: {cumulative_expectation[-1]:.2f}')

    # Update the 3D plot
    x_data = range(len(cumulative_expectation))
    y_data = cumulative_expectation
    z_data = [expected_value_theoretical] * len(cumulative_expectation)
    line3d.set_data(x_data, y_data)
    line3d.set_3d_properties(z_data)

    return line2d, line3d, expectation_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=50, init_func=init, blit=True)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setting up parameters for distributions
n_samples = 1000

# Distribution parameters
p_bernoulli = 0.5  # Probability for Bernoulli
n_binomial = 10  # Number of trials for Binomial
p_binomial = 0.5
p_geometric = 0.2  # Probability for Geometric
lambda_poisson = 3  # Lambda for Poisson

# Generate random data
data_bernoulli = np.random.binomial(1, p_bernoulli, n_samples)
data_binomial = np.random.binomial(n_binomial, p_binomial, n_samples)
data_geometric = np.random.geometric(p_geometric, n_samples)
data_poisson = np.random.poisson(lambda_poisson, n_samples)

# Compute means and variances
distributions = {
    "Bernoulli": data_bernoulli,
    "Binomial": data_binomial,
    "Geometric": data_geometric,
    "Poisson": data_poisson
}

means = {name: np.mean(data) for name, data in distributions.items()}
variances = {name: np.var(data) for name, data in distributions.items()}

# Plotting 2D visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot mean and variance in 2D
axes[0].bar(means.keys(), means.values(), color='teal', alpha=0.7)
axes[0].set_title('Mean of Distributions')
axes[0].set_ylabel('Mean')

axes[1].bar(variances.keys(), variances.values(), color='orange', alpha=0.7)
axes[1].set_title('Variance of Distributions')
axes[1].set_ylabel('Variance')

plt.tight_layout()
plt.show()

# 3D Plot for Mean and Variance
fig_3d = plt.figure(figsize=(10, 6))
ax = fig_3d.add_subplot(111, projection='3d')

# Plot bars for 3D visualization
x_pos = np.arange(len(means))
ax.bar(x_pos - 0.15, list(means.values()), 0.3, label='Mean', color='teal')
ax.bar(x_pos + 0.15, list(variances.values()), 0.3, label='Variance', color='orange')

# Labels and titles
ax.set_xticks(x_pos)
ax.set_xticklabels(means.keys())
ax.set_ylabel('Value')
ax.set_zlabel('Mean/Variance')
ax.set_title('3D Visualization of Mean and Variance')
ax.legend()

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Bernoulli Distribution Visualization
p = 0.5  # Probability of success
n_samples = 1000

# Generate Bernoulli samples
data = np.random.binomial(1, p, n_samples)

mean_values = []
variance_values = []

fig, ax = plt.subplots(figsize=(8, 4))
mean_line, = ax.plot([], [], label='Empirical Mean', color='blue')
variance_line, = ax.plot([], [], label='Empirical Variance', color='green')
ax.axhline(y=p, color='orange', linestyle='--', label=f'Theoretical Mean = {p:.2f}')
ax.axhline(y=p * (1 - p), color='red', linestyle='--', label=f'Theoretical Variance = {p * (1 - p):.2f}')
ax.legend()
ax.set_xlim(0, 50)
ax.set_ylim(0, 1)
ax.set_title("Bernoulli Mean and Variance Visualization")
ax.set_xlabel("Number of Samples")
ax.set_ylabel("Value")

def init():
    return mean_line, variance_line

def update(frame):
    sample_data = data[:frame + 1]
    mean_values.append(np.mean(sample_data))
    variance_values.append(np.var(sample_data))

    mean_line.set_data(range(len(mean_values)), mean_values)
    variance_line.set_data(range(len(variance_values)), variance_values)
    return mean_line, variance_line

ani = animation.FuncAnimation(fig, update, frames=50, init_func=init, blit=True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Bernoulli Distribution Visualization
p_bernoulli = 0.5
n_samples = 1000
data_bernoulli = np.random.binomial(1, p_bernoulli, n_samples)

# Empirical Mean and Variance for Bernoulli
mean_bernoulli = np.mean(data_bernoulli)
variance_bernoulli = np.var(data_bernoulli)

# Plot Bernoulli Distribution's Mean and Variance
plt.figure(figsize=(8, 6))
plt.hist(data_bernoulli, bins=2, edgecolor='black', alpha=0.7)
plt.axvline(mean_bernoulli, color='blue', linestyle='dashed', label=f'Empirical Mean: {mean_bernoulli:.2f}')
plt.axvline(variance_bernoulli, color='green', linestyle='dashed', label=f'Empirical Variance: {variance_bernoulli:.2f}')
plt.legend()
plt.title('Bernoulli Distribution - Mean and Variance')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Binomial Distribution Visualization
n_binomial = 10
p_binomial = 0.5
data_binomial = np.random.binomial(n_binomial, p_binomial, n_samples)

# Empirical Mean and Variance for Binomial
mean_binomial = np.mean(data_binomial)
variance_binomial = np.var(data_binomial)

# Plot Binomial Distribution's Mean and Variance
plt.figure(figsize=(8, 6))
plt.hist(data_binomial, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(mean_binomial, color='blue', linestyle='dashed', label=f'Empirical Mean: {mean_binomial:.2f}')
plt.axvline(variance_binomial, color='green', linestyle='dashed', label=f'Empirical Variance: {variance_binomial:.2f}')
plt.legend()
plt.title('Binomial Distribution - Mean and Variance')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Geometric Distribution Visualization
p_geometric = 0.2
data_geometric = np.random.geometric(p_geometric, n_samples)

# Empirical Mean and Variance for Geometric
mean_geometric = np.mean(data_geometric)
variance_geometric = np.var(data_geometric)

# Plot Geometric Distribution's Mean and Variance
plt.figure(figsize=(8, 6))
plt.hist(data_geometric, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(mean_geometric, color='blue', linestyle='dashed', label=f'Empirical Mean: {mean_geometric:.2f}')
plt.axvline(variance_geometric, color='green', linestyle='dashed', label=f'Empirical Variance: {variance_geometric:.2f}')
plt.legend()
plt.title('Geometric Distribution - Mean and Variance')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Poisson Distribution Visualization
lambda_poisson = 3
data_poisson = np.random.poisson(lambda_poisson, n_samples)

# Empirical Mean and Variance for Poisson
mean_poisson = np.mean(data_poisson)
variance_poisson = np.var(data_poisson)

# Plot Poisson Distribution's Mean and Variance
plt.figure(figsize=(8, 6))
plt.hist(data_poisson, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(mean_poisson, color='blue', linestyle='dashed', label=f'Empirical Mean: {mean_poisson:.2f}')
plt.axvline(variance_poisson, color='green', linestyle='dashed', label=f'Empirical Variance: {variance_poisson:.2f}')
plt.legend()
plt.title('Poisson Distribution - Mean and Variance')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to create and animate distribution visualization
def animate_distribution(data, n_samples, p, distribution_type, ax, n=None):
    mean_values = []
    variance_values = []

    mean_line, = ax.plot([], [], label='Empirical Mean', color='blue')
    variance_line, = ax.plot([], [], label='Empirical Variance', color='green')
    
    # Theoretical values depending on the distribution type
    if distribution_type == 'Bernoulli':
        ax.axhline(y=p, color='orange', linestyle='--', label=f'Theoretical Mean = {p:.2f}')
        ax.axhline(y=p * (1 - p), color='red', linestyle='--', label=f'Theoretical Variance = {p * (1 - p):.2f}')
    elif distribution_type == 'Binomial':
        ax.axhline(y=n * p, color='orange', linestyle='--', label=f'Theoretical Mean = {n * p}')
        ax.axhline(y=n * p * (1 - p), color='red', linestyle='--', label=f'Theoretical Variance = {n * p * (1 - p)}')
    elif distribution_type == 'Geometric':
        ax.axhline(y=1/p, color='orange', linestyle='--', label=f'Theoretical Mean = {1/p}')
        ax.axhline(y=(1 - p) / p**2, color='red', linestyle='--', label=f'Theoretical Variance = {(1 - p) / p**2}')
    elif distribution_type == 'Poisson':
        ax.axhline(y=p, color='orange', linestyle='--', label=f'Theoretical Mean = {p}')
        ax.axhline(y=p, color='red', linestyle='--', label=f'Theoretical Variance = {p}')
    
    ax.legend()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1)
    ax.set_title(f"{distribution_type} Mean and Variance Visualization")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Value")

    def init():
        return mean_line, variance_line

    def update(frame):
        sample_data = data[:frame + 1]
        mean_values.append(np.mean(sample_data))
        variance_values.append(np.var(sample_data))

        mean_line.set_data(range(len(mean_values)), mean_values)
        variance_line.set_data(range(len(variance_values)), variance_values)
        return mean_line, variance_line

    ani = animation.FuncAnimation(fig, update, frames=50, init_func=init, blit=True)
    plt.show()


# Example of Bernoulli distribution
p_bernoulli = 0.5
n_samples = 1000
data_bernoulli = np.random.binomial(1, p_bernoulli, n_samples)
fig, ax = plt.subplots(figsize=(8, 4))
animate_distribution(data_bernoulli, n_samples, p_bernoulli, 'Bernoulli', ax)

# Example of Binomial distribution
n_binomial = 10
p_binomial = 0.5
data_binomial = np.random.binomial(n_binomial, p_binomial, n_samples)
fig, ax = plt.subplots(figsize=(8, 4))
animate_distribution(data_binomial, n_samples, p_binomial, 'Binomial', ax, n=n_binomial)

# Example of Geometric distribution
p_geometric = 0.2
data_geometric = np.random.geometric(p_geometric, n_samples)
fig, ax = plt.subplots(figsize=(8, 4))
animate_distribution(data_geometric, n_samples, p_geometric, 'Geometric', ax)

# Example of Poisson distribution
lambda_poisson = 3
data_poisson = np.random.poisson(lambda_poisson, n_samples)
fig, ax = plt.subplots(figsize=(8, 4))
animate_distribution(data_poisson, n_samples, lambda_poisson, 'Poisson', ax)


import numpy as np
import matplotlib.pyplot as plt

# Geometric Distribution Visualization
p_geometric = 0.2
n_samples = 1000
data_geometric = np.random.geometric(p_geometric, n_samples)

# Empirical Mean and Variance for Geometric
mean_geometric = np.mean(data_geometric)
variance_geometric = np.var(data_geometric)

# Plot Geometric Distribution's Mean and Variance
plt.figure(figsize=(8, 6))
plt.hist(data_geometric, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(mean_geometric, color='blue', linestyle='dashed', label=f'Empirical Mean: {mean_geometric:.2f}')
plt.axvline(variance_geometric, color='green', linestyle='dashed', label=f'Empirical Variance: {variance_geometric:.2f}')
plt.legend()
plt.title('Geometric Distribution - Mean and Variance')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()


import matplotlib.animation as animation

# Animation for Geometric Distribution
def animate_geometric(data, p, n_samples):
    mean_values = []
    variance_values = []

    fig, ax = plt.subplots(figsize=(8, 4))
    mean_line, = ax.plot([], [], label='Empirical Mean', color='blue')
    variance_line, = ax.plot([], [], label='Empirical Variance', color='green')
    ax.axhline(y=1/p, color='orange', linestyle='--', label=f'Theoretical Mean = {1/p}')
    ax.axhline(y=(1 - p) / p**2, color='red', linestyle='--', label=f'Theoretical Variance = {(1 - p) / p**2}')
    ax.legend()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 5)
    ax.set_title("Geometric Distribution - Mean and Variance")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Value")

    def init():
        return mean_line, variance_line

    def update(frame):
        sample_data = data[:frame + 1]
        mean_values.append(np.mean(sample_data))
        variance_values.append(np.var(sample_data))

        mean_line.set_data(range(len(mean_values)), mean_values)
        variance_line.set_data(range(len(variance_values)), variance_values)
        return mean_line, variance_line

    ani = animation.FuncAnimation(fig, update, frames=50, init_func=init, blit=True)
    plt.show()

# Geometric animation call
animate_geometric(data_geometric, p_geometric, n_samples)


# Animation for Poisson Distribution
def animate_poisson(data, lambda_value, n_samples):
    mean_values = []
    variance_values = []

    fig, ax = plt.subplots(figsize=(8, 4))
    mean_line, = ax.plot([], [], label='Empirical Mean', color='blue')
    variance_line, = ax.plot([], [], label='Empirical Variance', color='green')
    ax.axhline(y=lambda_value, color='orange', linestyle='--', label=f'Theoretical Mean = {lambda_value}')
    ax.axhline(y=lambda_value, color='red', linestyle='--', label=f'Theoretical Variance = {lambda_value}')
    ax.legend()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 10)
    ax.set_title("Poisson Distribution - Mean and Variance")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Value")

    def init():
        return mean_line, variance_line

    def update(frame):
        sample_data = data[:frame + 1]
        mean_values.append(np.mean(sample_data))
        variance_values.append(np.var(sample_data))

        mean_line.set_data(range(len(mean_values)), mean_values)
        variance_line.set_data(range(len(variance_values)), variance_values)
        return mean_line, variance_line

    ani = animation.FuncAnimation(fig, update, frames=50, init_func=init, blit=True)
    plt.show()

# Poisson animation call
animate_poisson(data_poisson, lambda_poisson, n_samples)
