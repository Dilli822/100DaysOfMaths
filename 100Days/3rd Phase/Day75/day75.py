import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.animation as animation

# Setting style
sns.set(style="whitegrid")

# Poisson mean (lambda)
lam = 5  

# Range of n values for the Binomial distribution
n_values = [10, 20, 50, 100, 500, 1000]  
p_values = [lam / n for n in n_values]  # Ensuring mean remains lambda = np

# Range of x-axis values
x = np.arange(0, 20)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

def update(frame):
    ax.clear()
    
    # Plot Poisson Distribution (fixed)
    poisson_probs = stats.poisson.pmf(x, lam)
    ax.plot(x, poisson_probs, 'bo-', label=f"Poisson(λ={lam})", lw=2)

    # Plot Binomial Distribution (changing n)
    binomial_probs = stats.binom.pmf(x, n_values[frame], p_values[frame])
    ax.plot(x, binomial_probs, 'ro--', label=f"Binomial(n={n_values[frame]}, p={p_values[frame]:.4f})", lw=2)

    ax.set_title("Poisson vs Binomial Distribution")
    ax.set_xlabel("Number of Events")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 0.3)
    ax.legend()
    plt.pause(0.5)

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(n_values), repeat=True)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Generate data
np.random.seed(42)
mu, sigma1, sigma2 = 50, 10, 20  # Mean and two different standard deviations

# Normal distributions
data1 = np.random.normal(mu, sigma1, 1000)
data2 = np.random.normal(mu, sigma2, 1000)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data1, bins=30, kde=True, color="blue", label=f"SD={sigma1}, Variance={sigma1**2}", stat="density")
sns.histplot(data2, bins=30, kde=True, color="red", label=f"SD={sigma2}, Variance={sigma2**2}", stat="density")

ax.set_title("Variance vs Standard Deviation")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
plt.show()


# Generate normal distributions with different means
mu_values = [30, 50, 70]
sigma = 10
x = np.linspace(0, 100, 1000)

fig, ax = plt.subplots(figsize=(8, 5))
for mu in mu_values:
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, pdf, label=f"Mean = {mu}")

ax.set_title("Normal Distributions with Different Means")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
plt.show()


# Population data
pop_data = np.random.normal(50, 10, 10000)

# Sample sizes
sample_sizes = [5, 20, 50, 100, 500, 1000]
sample_means = [np.mean(np.random.choice(pop_data, size, replace=False)) for size in sample_sizes]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sample_sizes, sample_means, marker='o', linestyle='-', color="blue", label="Sample Mean")
ax.axhline(np.mean(pop_data), color="red", linestyle="--", label="Population Mean")

ax.set_xscale("log")
ax.set_xlabel("Sample Size (log scale)")
ax.set_ylabel("Mean Value")
ax.set_title("Sample Mean vs Population Mean")
ax.legend()
plt.show()


means = [50, 50, 50]  # Same mean
variances = [5, 15, 30]  # Different variances
colors = ["blue", "green", "red"]
x = np.linspace(10, 90, 1000)

fig, ax = plt.subplots(figsize=(8, 5))
for mu, var, color in zip(means, variances, colors):
    sigma = np.sqrt(var)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, pdf, color=color, label=f"Variance = {var}")

ax.set_title("Expected Mean vs Variance")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
plt.show()
