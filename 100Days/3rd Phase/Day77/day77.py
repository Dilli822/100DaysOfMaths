import matplotlib.pyplot as plt
import numpy as np

# Population data
population = np.random.normal(50, 10, 1000)

# Random sampling
sample = np.random.choice(population, 100)

# Plot
plt.figure(figsize=(10, 6))
plt.hist(population, bins=30, alpha=0.5, label='Population')
plt.hist(sample, bins=30, alpha=0.5, label='Random Sample')
plt.title("Random Sampling")
plt.legend()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Stratified sampling example
np.random.seed(0)
strata1 = np.random.normal(30, 5, 300)
strata2 = np.random.normal(60, 5, 700)
population = np.concatenate([strata1, strata2])

# Stratified sample
sample1 = np.random.choice(strata1, 30)
sample2 = np.random.choice(strata2, 70)
sample = np.concatenate([sample1, sample2])

# 3D Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(range(len(strata1)), strata1, np.zeros_like(strata1), label='Stratum 1')
ax.scatter(range(len(strata2)), strata2, np.zeros_like(strata2), label='Stratum 2')
ax.scatter(range(len(sample)), sample, np.ones_like(sample), label='Stratified Sample')
ax.set_title("Stratified Sampling")
ax.legend()
plt.show()


# Systematic sampling
population = np.arange(1000)
k = 10  # Interval
sample = population[::k]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(population, np.zeros_like(population), 'o', label='Population')
plt.plot(sample, np.zeros_like(sample), 'ro', label='Systematic Sample')
plt.title("Systematic Sampling")
plt.legend()
plt.show()

# Cluster sampling example
np.random.seed(0)
clusters = [np.random.normal(i, 2, 100) for i in range(5)]
population = np.concatenate(clusters)

# Randomly select clusters
selected_clusters = np.random.choice(range(5), 2, replace=False)
sample = np.concatenate([clusters[i] for i in selected_clusters])

# 3D Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
for i, cluster in enumerate(clusters):
    ax.scatter(range(len(cluster)), cluster, np.full_like(cluster, i), label=f'Cluster {i+1}')
ax.scatter(range(len(sample)), sample, np.full_like(sample, 5), label='Cluster Sample')
ax.set_title("Cluster Sampling")
ax.legend()
plt.show()

# Population with a specific characteristic
population = np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # 30% have the characteristic
sample = np.random.choice(population, 100)

# Calculate proportions
pop_proportion = np.mean(population)
sample_proportion = np.mean(sample)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(['Population', 'Sample'], [pop_proportion, sample_proportion], color=['blue', 'orange'])
plt.title("Distribution Proportion")
plt.ylabel("Proportion")
plt.show()

# Sampling distribution of proportions
np.random.seed(0)
proportions = [np.mean(np.random.choice(population, 100)) for _ in range(1000)]

# 3D Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.hist(proportions, bins=30, color='blue', alpha=0.7)
ax.set_title("Sampling Distribution of Proportions")
ax.set_xlabel("Proportion")
ax.set_ylabel("Frequency")
ax.set_zlabel("Height")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Standard Error Visualization with Clearer Illustration
def standard_error_plot():
    sample_sizes = np.arange(1, 101, 1)
    std_dev = 10  # Assume fixed population standard deviation
    standard_errors = std_dev / np.sqrt(sample_sizes)
    
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, standard_errors, marker='o', linestyle='-', color='purple', label='Standard Error')
    plt.axhline(y=std_dev, color='r', linestyle='--', label='Population Std Dev')
    plt.xlabel('Sample Size')
    plt.ylabel('Standard Error')
    plt.title('Standard Error vs. Sample Size (Illustrative)')
    plt.legend()
    plt.grid()
    plt.show()

# 2. Central Limit Theorem Animation with Clearer Illustration
def clt_animation():
    np.random.seed(42)
    population = np.random.exponential(scale=2, size=10000)
    sample_size = 30
    num_samples = 100
    means = []
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 50)
    ax.set_title('Central Limit Theorem: Sample Means Distribution')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Frequency')
    
    def update(frame):
        means.append(np.mean(np.random.choice(population, sample_size)))
        ax.clear()
        ax.hist(means, bins=20, color='teal', alpha=0.7, edgecolor='black', label=f'Samples: {frame+1}')
        ax.axvline(x=np.mean(means), color='r', linestyle='--', label='Mean of Means')
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 50)
        ax.set_title('Central Limit Theorem: Sample Means Distribution')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    ani = animation.FuncAnimation(fig, update, frames=num_samples, repeat=False)
    plt.show()

# 3. Variance Visualization with Better Clarity
def variance_plot():
    np.random.seed(42)
    low_variance = np.random.normal(loc=0, scale=1, size=1000)
    high_variance = np.random.normal(loc=0, scale=5, size=1000)
    
    plt.figure(figsize=(8, 5))
    plt.hist(low_variance, bins=30, alpha=0.6, label='Low Variance', color='blue', edgecolor='black', density=True)
    plt.hist(high_variance, bins=30, alpha=0.6, label='High Variance', color='red', edgecolor='black', density=True)
    plt.axvline(np.std(low_variance), color='blue', linestyle='dashed', label='Low Variance Std Dev')
    plt.axvline(np.std(high_variance), color='red', linestyle='dashed', label='High Variance Std Dev')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Low vs. High Variance Distributions')
    plt.legend()
    plt.grid()
    plt.show()

# Run the improved visualizations
standard_error_plot()
clt_animation()
variance_plot()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Standard Error Visualization with Clearer Illustration
def standard_error_plot():
    sample_sizes = np.arange(1, 101, 1)
    std_dev = 10  # Assume fixed population standard deviation
    standard_errors = std_dev / np.sqrt(sample_sizes)
    
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, standard_errors, marker='o', linestyle='-', color='purple', label='Standard Error')
    plt.axhline(y=std_dev, color='r', linestyle='--', label='Population Std Dev')
    plt.xlabel('Sample Size')
    plt.ylabel('Standard Error')
    plt.title('Standard Error vs. Sample Size (Illustrative)')
    plt.legend()
    plt.grid()
    plt.show()

# 2. Central Limit Theorem Animation with Clearer Illustration
def clt_animation():
    np.random.seed(42)
    population = np.random.exponential(scale=2, size=10000)
    sample_size = 30
    num_samples = 100
    means = []
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 50)
    ax.set_title('Central Limit Theorem: Sample Means Distribution')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Frequency')
    
    def update(frame):
        means.append(np.mean(np.random.choice(population, sample_size)))
        ax.clear()
        ax.hist(means, bins=20, color='teal', alpha=0.7, edgecolor='black', label=f'Samples: {frame+1}')
        ax.axvline(x=np.mean(means), color='r', linestyle='--', label='Mean of Means')
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 50)
        ax.set_title('Central Limit Theorem: Sample Means Distribution')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    ani = animation.FuncAnimation(fig, update, frames=num_samples, repeat=False)
    plt.show()

# 3. Variance Visualization with Better Clarity
def variance_plot():
    np.random.seed(42)
    low_variance = np.random.normal(loc=0, scale=1, size=1000)
    high_variance = np.random.normal(loc=0, scale=5, size=1000)
    
    plt.figure(figsize=(8, 5))
    plt.hist(low_variance, bins=30, alpha=0.6, label='Low Variance', color='blue', edgecolor='black', density=True)
    plt.hist(high_variance, bins=30, alpha=0.6, label='High Variance', color='red', edgecolor='black', density=True)
    plt.axvline(np.std(low_variance), color='blue', linestyle='dashed', label='Low Variance Std Dev')
    plt.axvline(np.std(high_variance), color='red', linestyle='dashed', label='High Variance Std Dev')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Low vs. High Variance Distributions')
    plt.legend()
    plt.grid()
    plt.show()

# 4. Sampling Methods Visualization
def sampling_methods():
    np.random.seed(42)
    population = np.random.normal(loc=50, scale=15, size=10000)
    sample_size = 100
    
    # Simple Random Sampling
    simple_random_sample = np.random.choice(population, sample_size, replace=False)
    
    # Systematic Sampling
    step = len(population) // sample_size
    systematic_sample = population[::step][:sample_size]
    
    # Stratified Sampling (Using bins to simulate strata)
    strata = np.digitize(population, bins=np.histogram_bin_edges(population, bins=5))
    stratified_sample = np.concatenate([np.random.choice(population[strata == i], sample_size // 5, replace=False) for i in range(1, 6)])
    
    # Cluster Sampling (Random clusters)
    cluster_size = sample_size // 5
    clusters = np.array_split(population, 5)
    chosen_cluster = np.random.choice(range(5))
    cluster_sample = np.random.choice(clusters[chosen_cluster], cluster_size, replace=False)
    
    plt.figure(figsize=(10, 6))
    plt.hist(simple_random_sample, bins=20, alpha=0.5, label='Simple Random', color='blue', edgecolor='black')
    plt.hist(systematic_sample, bins=20, alpha=0.5, label='Systematic', color='green', edgecolor='black')
    plt.hist(stratified_sample, bins=20, alpha=0.5, label='Stratified', color='red', edgecolor='black')
    plt.hist(cluster_sample, bins=20, alpha=0.5, label='Cluster', color='purple', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Comparison of Sampling Methods')
    plt.legend()
    plt.grid()
    plt.show()

# Run the improved visualizations
standard_error_plot()
clt_animation()
variance_plot()
sampling_methods()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# 1. Standard Error Visualization with Clearer Illustration
def standard_error_plot():
    sample_sizes = np.arange(1, 101, 1)
    std_dev = 10  # Assume fixed population standard deviation
    standard_errors = std_dev / np.sqrt(sample_sizes)
    
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, standard_errors, marker='o', linestyle='-', color='purple', label='Standard Error')
    plt.axhline(y=std_dev, color='r', linestyle='--', label='Population Std Dev')
    plt.xlabel('Sample Size')
    plt.ylabel('Standard Error')
    plt.title('Standard Error vs. Sample Size (Illustrative)')
    plt.legend()
    plt.grid()
    plt.show()

# 2. Central Limit Theorem Animation with Clearer Illustration
def clt_animation():
    np.random.seed(42)
    population = np.random.exponential(scale=2, size=10000)
    sample_size = 30
    num_samples = 100
    means = []
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 50)
    ax.set_title('Central Limit Theorem: Sample Means Distribution')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Frequency')
    
    def update(frame):
        means.append(np.mean(np.random.choice(population, sample_size)))
        ax.clear()
        ax.hist(means, bins=20, color='teal', alpha=0.7, edgecolor='black', label=f'Samples: {frame+1}')
        ax.axvline(x=np.mean(means), color='r', linestyle='--', label='Mean of Means')
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 50)
        ax.set_title('Central Limit Theorem: Sample Means Distribution')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    ani = animation.FuncAnimation(fig, update, frames=num_samples, repeat=False)
    plt.show()

# 3. Variance Visualization with Better Clarity
def variance_plot():
    np.random.seed(42)
    low_variance = np.random.normal(loc=0, scale=1, size=1000)
    high_variance = np.random.normal(loc=0, scale=5, size=1000)
    
    plt.figure(figsize=(8, 5))
    plt.hist(low_variance, bins=30, alpha=0.6, label='Low Variance', color='blue', edgecolor='black', density=True)
    plt.hist(high_variance, bins=30, alpha=0.6, label='High Variance', color='red', edgecolor='black', density=True)
    plt.axvline(np.std(low_variance), color='blue', linestyle='dashed', label='Low Variance Std Dev')
    plt.axvline(np.std(high_variance), color='red', linestyle='dashed', label='High Variance Std Dev')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Low vs. High Variance Distributions')
    plt.legend()
    plt.grid()
    plt.show()

# 4. Non-Probability Sampling Methods Visualization in 2D and 3D
def non_probability_sampling_methods():
    np.random.seed(42)
    population = np.random.normal(loc=50, scale=15, size=10000)
    sample_size = 100
    
    # Convenience Sampling
    convenience_sample = population[:sample_size]
    
    # Judgmental Sampling
    judgmental_sample = population[population > np.percentile(population, 70)][:sample_size]
    
    # Snowball Sampling (Simulated by selecting elements with similar characteristics)
    seed = np.random.choice(population, 1)
    snowball_sample = population[np.abs(population - seed) < 5][:sample_size]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(convenience_sample, bins=20, alpha=0.5, label='Convenience', color='cyan', edgecolor='black')
    ax.hist(judgmental_sample, bins=20, alpha=0.5, label='Judgmental', color='orange', edgecolor='black')
    ax.hist(snowball_sample, bins=20, alpha=0.5, label='Snowball', color='magenta', edgecolor='black')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Comparison of Non-Probability Sampling Methods')
    ax.legend()
    ax.grid()
    plt.show()
    
    # 3D Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(convenience_sample, np.zeros_like(convenience_sample), np.zeros_like(convenience_sample), label='Convenience', color='cyan')
    ax.scatter(judgmental_sample, np.ones_like(judgmental_sample), np.zeros_like(judgmental_sample), label='Judgmental', color='orange')
    ax.scatter(snowball_sample, np.ones_like(snowball_sample) * 2, np.zeros_like(snowball_sample), label='Snowball', color='magenta')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Sampling Method')
    ax.set_zlabel('Frequency')
    ax.set_title('3D Representation of Non-Probability Sampling Methods')
    ax.legend()
    plt.show()

# Run the improved visualizations
standard_error_plot()
clt_animation()
variance_plot()
non_probability_sampling_methods()
