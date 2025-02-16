import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from scipy.stats import norm, t

# Set random seed for reproducibility
np.random.seed(42)

# True population parameters
mu, sigma = 50, 10  
sample_size = 30  
num_samples = 50  

# Confidence level
confidence = 0.95
alpha = 1 - confidence

# Generate animation data
sample_means = []
conf_intervals = []

for _ in range(num_samples):
    sample = np.random.normal(mu, sigma, sample_size)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)  # Sample standard deviation
    t_critical = t.ppf(1 - alpha / 2, df=sample_size - 1)  # T critical value
    
    margin_of_error = t_critical * (sample_std / np.sqrt(sample_size))
    conf_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

    sample_means.append(sample_mean)
    conf_intervals.append(conf_interval)

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(mu, color='red', linestyle='--', label='True Mean (μ)')

def update(frame):
    ax.clear()
    ax.axhline(mu, color='red', linestyle='--', label='True Mean (μ)')
    ax.scatter([0], [mu], color='orange', label='Error CI', marker='o', s=100)
    ax.scatter([1], [mu], color='green', label='Accept CI', marker='o', s=100)

    ax.set_title("Confidence Interval Animation")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Mean ± CI")
    ax.set_ylim(mu - 20, mu + 20)
    ax.set_xlim(0, num_samples)

    for i in range(frame):
        x = i + 1
        mean = sample_means[i]
        ci_low, ci_high = conf_intervals[i]
        color = 'green' if ci_low <= mu <= ci_high else 'orange'
        ax.errorbar(x, mean, yerr=[[mean - ci_low], [ci_high - mean]], fmt='o', color=color)

    ax.legend()

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_samples, interval=300)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)

# Population parameters
mu, sigma = 50, 10  # True Mean and Standard Deviation
sample_sizes = [5, 10, 30, 50, 100]  # Different sample sizes
num_samples = 100  # Number of samples per size

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

def update(frame):
    ax.clear()
    sample_size = sample_sizes[frame % len(sample_sizes)]  # Cycle through sample sizes
    sample_means = []

    for _ in range(num_samples):
        sample = np.random.normal(mu, sigma, sample_size)
        sample_means.append(np.mean(sample))

    # Standard Error Calculation
    se = sigma / np.sqrt(sample_size)
    
    # Plot histogram of sample means
    sns.histplot(sample_means, bins=15, kde=True, color='blue', alpha=0.6, ax=ax)
    
    # Plot normal distribution curve
    x_vals = np.linspace(mu - 4*se, mu + 4*se, 100)
    y_vals = norm.pdf(x_vals, mu, se) * num_samples * (max(sample_means) - min(sample_means)) / 5
    ax.plot(x_vals, y_vals, 'r-', label=f'Normal Curve (SE={se:.2f})')

    # Z-Values (±1.96 SE)
    ax.axvline(mu + 1.96 * se, color='green', linestyle='--', label=f'Z=1.96 (Upper CI)')
    ax.axvline(mu - 1.96 * se, color='green', linestyle='--', label=f'Z=-1.96 (Lower CI)')
    ax.axvline(mu, color='red', linestyle='-', label='Population Mean (μ)')

    ax.set_title(f'Standard Error & Z-Value Animation (n={sample_size})')
    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Frequency")
    ax.legend()

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(sample_sizes) * 2, interval=1000)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

# Set random seed for reproducibility
np.random.seed(42)

# Population parameters
mu, sigma = 50, 10  # Mean and Standard Deviation
sample_sizes = [5, 10, 30, 50, 100]  # Different sample sizes
num_samples = 100  # Number of samples per size

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

def update(frame):
    ax.clear()
    sample_size = sample_sizes[frame % len(sample_sizes)]  # Cycle through sample sizes
    sample_means = []

    for _ in range(num_samples):
        sample = np.random.normal(mu, sigma, sample_size)
        sample_means.append(np.mean(sample))

    # Standard Error Calculation
    se = sigma / np.sqrt(sample_size)
    
    # Plot histogram of sample means
    sns.histplot(sample_means, bins=15, kde=True, color='blue', alpha=0.6, ax=ax)
    
    # Vertical Line at True Mean
    ax.axvline(mu, color='red', linestyle='-', label='True Mean (μ)')

    # Show decreasing SE effect
    ax.set_title(f'Standard Error of Mean (n={sample_size}, SE={se:.2f})')
    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Frequency")
    ax.legend()

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(sample_sizes) * 2, interval=1000)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set random seed
np.random.seed(42)

# Population parameters
mu, sigma = 50, 10  # True mean and standard deviation
sample_sizes = np.arange(1, 500)  # Increasing sample sizes
sample_means = []

# Generate sample means
for n in sample_sizes:
    sample = np.random.normal(mu, sigma, n)
    sample_means.append(np.mean(sample))

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

def update(frame):
    ax.clear()
    ax.plot(sample_sizes[:frame], sample_means[:frame], color='blue', label="Sample Mean")
    ax.axhline(mu, color='red', linestyle='--', label="True Mean (μ)")
    
    ax.set_title(f"Law of Large Numbers (n={sample_sizes[frame]})")
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Sample Mean")
    ax.legend()

ani = animation.FuncAnimation(fig, update, frames=len(sample_sizes), interval=50)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.animation as animation

# Set random seed for reproducibility
np.random.seed(42)

# Population parameters
mu, sigma = 50, 10  # True mean and standard deviation
sample_sizes = np.arange(5, 200, 5)  # Increasing sample sizes
confidence_level = 0.95
z_value = stats.norm.ppf((1 + confidence_level) / 2)  # Z-score for 95% CI

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

def update(frame):
    ax.clear()
    sample_size = sample_sizes[frame]
    
    # Generate sample and compute confidence interval
    sample = np.random.normal(mu, sigma, sample_size)
    sample_mean = np.mean(sample)
    se = sigma / np.sqrt(sample_size)  # Standard Error
    margin_of_error = z_value * se
    
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    # Plot confidence interval
    ax.errorbar(sample_size, sample_mean, yerr=margin_of_error, fmt='o', color='blue', label="Confidence Interval")
    ax.axhline(mu, color='red', linestyle='--', label="True Mean (μ)")
    
    ax.set_title(f"Confidence Interval (n={sample_size}, 95% CI)")
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Mean Estimate")
    ax.legend()

ani = animation.FuncAnimation(fig, update, frames=len(sample_sizes), interval=100)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set random seed
np.random.seed(42)

# Population parameters
mu, sigma = 50, 10  # True mean and standard deviation
sample_sizes = np.arange(5, 500, 5)  # Increasing sample sizes
sample_means = []

# Generate sample means
for n in sample_sizes:
    sample = np.random.normal(mu, sigma, n)
    sample_means.append(np.mean(sample))

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

def update(frame):
    ax.clear()
    ax.plot(sample_sizes[:frame], sample_means[:frame], color='blue', label="Sample Mean")
    ax.axhline(mu, color='red', linestyle='--', label="True Mean (μ)")
    
    ax.set_title(f"Asymptotics: Sample Mean vs. Population Mean (n={sample_sizes[frame]})")
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Sample Mean")
    ax.legend()

ani = animation.FuncAnimation(fig, update, frames=len(sample_sizes), interval=50)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.animation as animation

# Set random seed for reproducibility
np.random.seed(42)

# Population parameters
mu, sigma = 50, 10  # True mean and standard deviation
sample_sizes = np.arange(5, 200, 5)  # Increasing sample sizes
confidence_level = 0.95
z_value = stats.norm.ppf((1 + confidence_level) / 2)  # Z-score for 95% CI

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

def update(frame):
    ax.clear()
    sample_size = sample_sizes[frame]
    
    # Generate sample and compute statistics
    sample = np.random.normal(mu, sigma, sample_size)
    sample_mean = np.mean(sample)
    se = sigma / np.sqrt(sample_size)  # Standard Error
    margin_of_error = z_value * se
    
    # Confidence Interval
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    
    # Plot components
    ax.scatter(sample_size, sample_mean, color='blue', label="Sample Mean (Point Estimate)")
    ax.errorbar(sample_size, sample_mean, yerr=margin_of_error, fmt='o', color='green', label="Margin of Error")
    ax.plot([sample_size, sample_size], [lower_bound, upper_bound], color='green', label="Confidence Interval")
    ax.axhline(mu, color='red', linestyle='--', label="True Mean (μ)")
    
    # Labels and legend
    ax.set_title(f"Sample Statistics, Margin of Error & CI (n={sample_size}, 95% CI)")
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Mean Estimate")
    ax.legend()

ani = animation.FuncAnimation(fig, update, frames=len(sample_sizes), interval=100)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.animation as animation

# Set random seed for reproducibility
np.random.seed(42)

# Population parameters
mu, sigma = 50, 10  # True mean and standard deviation
sample_sizes = np.arange(5, 200, 5)  # Increasing sample sizes
confidence_level = 0.95
z_value = stats.norm.ppf((1 + confidence_level) / 2)  # Z-score for 95% CI

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

def update(frame):
    ax.clear()
    sample_size = sample_sizes[frame]
    
    # Generate sample and compute statistics
    sample = np.random.normal(mu, sigma, sample_size)
    sample_mean = np.mean(sample)
    se = sigma / np.sqrt(sample_size)  # Standard Error
    margin_of_error = z_value * se
    
    # Confidence Interval
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    
    # Plot components
    ax.scatter(sample_size, sample_mean, color='blue', label="Sample Mean (Point Estimate)")
    ax.errorbar(sample_size, sample_mean, yerr=margin_of_error, fmt='o', color='green', label="Margin of Error")
    ax.plot([sample_size, sample_size], [lower_bound, upper_bound], color='green', label="Confidence Interval")
    ax.axhline(mu, color='red', linestyle='--', label="True Mean (μ)")
    
    # Plot Upper and Lower Limits
    ax.scatter(sample_size, lower_bound, color='purple', label="Lower Limit")
    ax.scatter(sample_size, upper_bound, color='purple', label="Upper Limit")
    
    # Labels and legend
    ax.set_title(f"Sample Statistics, Margin of Error & CI (n={sample_size}, 95% CI)")
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Mean Estimate")
    ax.legend(loc='upper left')

ani = animation.FuncAnimation(fig, update, frames=len(sample_sizes), interval=100)

plt.show()
