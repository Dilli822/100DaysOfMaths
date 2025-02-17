import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

# Create data
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, 0, 1)

# Define significance level (alpha)
alpha = 0.05
z_critical_one_tailed = norm.ppf(1 - alpha)
z_critical_two_tailed = norm.ppf(1 - alpha/2)

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, 'b', label='Normal Distribution')
ax.fill_between(x, y, where=(x >= z_critical_one_tailed), color='red', alpha=0.5, label='One-tailed Rejection')
ax.fill_between(x, y, where=(x >= z_critical_two_tailed) | (x <= -z_critical_two_tailed), color='orange', alpha=0.5, label='Two-tailed Rejection')
ax.axvline(z_critical_one_tailed, color='red', linestyle='dashed')
ax.axvline(-z_critical_two_tailed, color='orange', linestyle='dashed')
ax.axvline(z_critical_two_tailed, color='orange', linestyle='dashed')
ax.set_title("One-Tailed vs. Two-Tailed Test Visualization")
ax.legend()

# Animation function
def update(frame):
    ax.collections.clear()
    ax.fill_between(x, y, where=(x >= z_critical_one_tailed if frame % 2 == 0 else (x >= z_critical_two_tailed) | (x <= -z_critical_two_tailed)),
                     color='red' if frame % 2 == 0 else 'orange', alpha=0.5)
    return ax,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=10, interval=1000, repeat=True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

# Generate data
np.random.seed(42)
sample_size = 30
population_mean = 50
sample_1 = np.random.normal(loc=population_mean, scale=10, size=sample_size)
sample_2 = np.random.normal(loc=55, scale=10, size=sample_size)  # Different mean for two-sample test

# Define x-axis for normal curves
x = np.linspace(30, 70, 300)
pdf_pop = norm.pdf(x, loc=population_mean, scale=10)

# Create figure
fig, ax = plt.subplots()
ax.set_xlim(30, 70)
ax.set_ylim(0, 0.05)
line1, = ax.plot([], [], 'r-', label="Sample 1 Distribution")
line2, = ax.plot([], [], 'g-', label="Sample 2 Distribution")
ax.plot(x, pdf_pop, 'b--', label="Population Distribution")

ax.legend()
ax.set_title("One-Sample vs Two-Sample Test Animation")
ax.set_xlabel("Value")
ax.set_ylabel("Density")

def update(frame):
    if frame < 25:
        # One-sample test
        pdf_sample = norm.pdf(x, loc=np.mean(sample_1[:frame+5]), scale=10)
        line1.set_data(x, pdf_sample)
        line2.set_data([], [])  # Hide second sample
        ax.set_title("One-Sample Test")
    else:
        # Two-sample test
        pdf_sample1 = norm.pdf(x, loc=np.mean(sample_1), scale=10)
        pdf_sample2 = norm.pdf(x, loc=np.mean(sample_2[:frame-20]), scale=10)
        line1.set_data(x, pdf_sample1)
        line2.set_data(x, pdf_sample2)
        ax.set_title("Two-Sample Test")

ani = animation.FuncAnimation(fig, update, frames=50, interval=100)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm, t

# Define x-axis range
x = np.linspace(-4, 4, 300)

# Z-distribution (Standard Normal)
z_dist = norm.pdf(x, loc=0, scale=1)

# T-distribution (for different degrees of freedom)
t_dist_10 = t.pdf(x, df=10)
t_dist_5 = t.pdf(x, df=5)
t_dist_2 = t.pdf(x, df=2)

# Create figure
fig, ax = plt.subplots()
ax.set_xlim(-4, 4)
ax.set_ylim(0, 0.45)
ax.set_title("Z-Test vs T-Test: Distribution Comparison")
ax.set_xlabel("Test Statistic")
ax.set_ylabel("Probability Density")

# Lines to update
line_z, = ax.plot(x, z_dist, 'b-', label="Z-Test (Normal Dist.)")
line_t, = ax.plot([], [], 'r-', label="T-Test (T Dist.)")
text = ax.text(-3, 0.4, "", fontsize=12, color="black")

ax.legend()

# Animation update function
def update(frame):
    if frame < 10:
        line_t.set_data(x, t_dist_10)
        text.set_text("T-Test (df=10): Closer to Normal")
    elif frame < 20:
        line_t.set_data(x, t_dist_5)
        text.set_text("T-Test (df=5): Wider Spread")
    else:
        line_t.set_data(x, t_dist_2)
        text.set_text("T-Test (df=2): More Spread Out")

ani = animation.FuncAnimation(fig, update, frames=30, interval=200)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# Define x-axis range
x = np.linspace(-4, 4, 300)

# Z-distribution (Standard Normal)
z_dist = norm.pdf(x, loc=0, scale=1)

# T-distributions (varying degrees of freedom)
t_dist_10 = t.pdf(x, df=10)
t_dist_5 = t.pdf(x, df=5)
t_dist_2 = t.pdf(x, df=2)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Z-Test (Normal Distribution)
axes[0].plot(x, z_dist, 'b-', label="Z-Test (Normal Dist.)")
axes[0].set_title("Z-Test: Normal Distribution")
axes[0].set_xlabel("Test Statistic")
axes[0].set_ylabel("Probability Density")
axes[0].legend()
axes[0].grid()

# Plot T-Test (T-Distribution)
axes[1].plot(x, t_dist_10, 'r-', label="T-Test (df=10)")
axes[1].plot(x, t_dist_5, 'g-', label="T-Test (df=5)")
axes[1].plot(x, t_dist_2, 'm-', label="T-Test (df=2)")
axes[1].set_title("T-Test: T-Distribution (Varying df)")
axes[1].set_xlabel("Test Statistic")
axes[1].set_ylabel("Probability Density")
axes[1].legend()
axes[1].grid()

# Show plots
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Generate sample data (e.g., heights in cm)
data = np.random.normal(170, 10, 100)  # Mean=170, Std=10, n=100

# Bootstrap resampling
n_iterations = 1000
bootstrap_means = []

for _ in range(n_iterations):
    sample = np.random.choice(data, size=len(data), replace=True)
    bootstrap_means.append(np.mean(sample))

# Plot bootstrap distribution
plt.hist(bootstrap_means, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(data), color='red', linestyle="dashed", label="Original Mean")
plt.axvline(np.percentile(bootstrap_means, 2.5), color='blue', linestyle="dashed", label="2.5% CI")
plt.axvline(np.percentile(bootstrap_means, 97.5), color='blue', linestyle="dashed", label="97.5% CI")
plt.xlabel("Bootstrap Sample Mean")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of Sample Mean")
plt.legend()
plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate sample data (e.g., heights in cm)
np.random.seed(42)
data = np.random.normal(170, 10, 100)  # Mean=170, Std=10, n=100

# Bootstrap resampling
n_iterations = 1000
bootstrap_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_iterations)]

# Plot KDE (Density under the curve)
plt.figure(figsize=(8,5))
sns.kdeplot(bootstrap_means, fill=True, color='skyblue', alpha=0.6)

# Confidence Interval (95%)
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
plt.axvline(np.mean(data), color='red', linestyle="dashed", label="Original Mean")
plt.axvline(ci_lower, color='blue', linestyle="dashed", label="2.5% CI")
plt.axvline(ci_upper, color='blue', linestyle="dashed", label="97.5% CI")

plt.xlabel("Bootstrap Sample Mean")
plt.ylabel("Density")
plt.title("Bootstrap Distribution with Density Curve")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

# Parameters
population_mean = 50
population_std = 10
sample_size = 30
alpha = 0.05
n_frames = 100  # Number of frames for animation

# Create the plot and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set the x range for plotting the population distribution
x = np.linspace(30, 70, 500)
population_distribution = norm.pdf(x, population_mean, population_std)

# Plot the population distribution
ax.plot(x, population_distribution, label="Population Distribution", color='blue')

# Plot the critical region
critical_value = norm.ppf(1 - alpha / 2, population_mean, population_std / np.sqrt(sample_size))
ax.fill_between(x, population_distribution, where=(x >= critical_value), color='red', alpha=0.3, label="Critical Region")
ax.fill_between(x, population_distribution, where=(x <= -critical_value), color='red', alpha=0.3)

# Set labels and title
ax.set_xlabel('Value')
ax.set_ylabel('Probability Density')
ax.set_title('Hypothesis Testing Visualization (Z-test)')

# Add the text for the population mean
ax.axvline(population_mean, color='green', linestyle='dashed', label='Population Mean')

# Function to update the animation frame
def update(frame):
    ax.clear()

    # Plot the population distribution again
    ax.plot(x, population_distribution, label="Population Distribution", color='blue')

    # Generate a random sample
    sample_data = np.random.normal(population_mean, population_std, sample_size)

    # Calculate the sample mean
    sample_mean = np.mean(sample_data)

    # Calculate the z-score for the sample mean
    z_score = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))

    # Plot the sample mean
    ax.plot([sample_mean, sample_mean], [0, norm.pdf(sample_mean, population_mean, population_std)], color='orange', linewidth=2, label="Sample Mean")

    # Plot the z-score as a vertical line
    ax.axvline(sample_mean, color='orange', linestyle='dashed')

    # Plot the critical region
    ax.fill_between(x, population_distribution, where=(x >= critical_value), color='red', alpha=0.3, label="Critical Region")
    ax.fill_between(x, population_distribution, where=(x <= -critical_value), color='red', alpha=0.3)

    # Add the population mean line again
    ax.axvline(population_mean, color='green', linestyle='dashed', label='Population Mean')

    # Annotate the z-score
    ax.annotate(f"z = {z_score:.2f}", xy=(sample_mean + 2, 0.02), color='orange', fontsize=12)

    # Set labels and title again
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('Hypothesis Testing Visualization (Z-test)')

    # Add the legend
    ax.legend()

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=n_frames, repeat=False, interval=100)

# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

# Parameters
population_mean = 50
population_std = 10
sample_size = 30
alpha = 0.05  # Significance level
n_frames = 100  # Number of frames for animation

# Create the plot and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set the x range for plotting the population distribution
x = np.linspace(30, 70, 500)
population_distribution = norm.pdf(x, population_mean, population_std)

# Plot the population distribution
ax.plot(x, population_distribution, label="Population Distribution", color='blue')

# Plot the critical region
critical_value = norm.ppf(1 - alpha / 2, population_mean, population_std / np.sqrt(sample_size))
ax.fill_between(x, population_distribution, where=(x >= critical_value), color='red', alpha=0.3, label="Critical Region")
ax.fill_between(x, population_distribution, where=(x <= -critical_value), color='red', alpha=0.3)

# Set labels and title
ax.set_xlabel('Value')
ax.set_ylabel('Probability Density')
ax.set_title('Hypothesis Testing Visualization (Z-test)')

# Add the text for the population mean
ax.axvline(population_mean, color='green', linestyle='dashed', label='Population Mean')

# Function to update the animation frame
def update(frame):
    ax.clear()

    # Plot the population distribution again
    ax.plot(x, population_distribution, label="Population Distribution", color='blue')

    # Generate a random sample
    sample_data = np.random.normal(population_mean, population_std, sample_size)

    # Calculate the sample mean
    sample_mean = np.mean(sample_data)

    # Calculate the z-score for the sample mean
    z_score = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))

    # Calculate the p-value from the z-score (two-tailed test)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    # Plot the sample mean
    ax.plot([sample_mean, sample_mean], [0, norm.pdf(sample_mean, population_mean, population_std)], color='orange', linewidth=2, label="Sample Mean")

    # Plot the z-score as a vertical line
    ax.axvline(sample_mean, color='orange', linestyle='dashed')

    # Plot the critical region
    ax.fill_between(x, population_distribution, where=(x >= critical_value), color='red', alpha=0.3, label="Critical Region")
    ax.fill_between(x, population_distribution, where=(x <= -critical_value), color='red', alpha=0.3)

    # Add the population mean line again
    ax.axvline(population_mean, color='green', linestyle='dashed', label='Population Mean')

    # Annotate the z-score and p-value
    ax.annotate(f"z = {z_score:.2f}", xy=(sample_mean + 2, 0.02), color='orange', fontsize=12)
    ax.annotate(f"p = {p_value:.3f}", xy=(sample_mean + 2, 0.015), color='orange', fontsize=12)

    # Decision based on p-value
    if p_value < alpha:
        decision_text = "Reject H₀"
        decision_color = 'red'
    else:
        decision_text = "Fail to Reject H₀"
        decision_color = 'green'
    
    # Display the hypothesis testing result
    ax.text(0.5, 0.95, f"Decision: {decision_text}", ha="center", va="top", fontsize=14, color=decision_color, transform=ax.transAxes)

    # Set labels and title again
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('Hypothesis Testing Visualization (Z-test)')

    # Add the legend
    ax.legend()

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=n_frames, repeat=False, interval=800)

# Show the plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

# Parameters
mu_null = 50  # Population mean under H0
mu_alt = 52   # Population mean under H1 (alternative hypothesis)
std_dev = 10  # Standard deviation
alpha = 0.05  # Significance level (5%)
sample_size = 30
n_frames = 100  # Number of frames in the animation

# Create the plot and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set the x range for plotting the distributions
x = np.linspace(30, 70, 500)

# Plot the null and alternative hypothesis distributions
ax.plot(x, norm.pdf(x, mu_null, std_dev), label='H₀: μ = 50 (Null Hypothesis)', color='blue')
ax.plot(x, norm.pdf(x, mu_alt, std_dev), label='H₁: μ = 52 (Alternative Hypothesis)', color='green')

# Critical value for Type I error
critical_value = norm.ppf(1 - alpha, mu_null, std_dev / np.sqrt(sample_size))
ax.axvline(critical_value, color='red', linestyle='dashed', label='Critical Value')

# Fill the rejection region for Type I error
ax.fill_between(x, norm.pdf(x, mu_null, std_dev), where=(x >= critical_value), color='red', alpha=0.3, label="Rejection Region (Type I Error)")

# Set labels and title
ax.set_xlabel('Value')
ax.set_ylabel('Probability Density')
ax.set_title('Type I and Type II Errors in Hypothesis Testing')

# Function to update the animation frame
def update(frame):
    ax.clear()

    # Plot the null and alternative hypothesis distributions again
    ax.plot(x, norm.pdf(x, mu_null, std_dev), label='H₀: μ = 50 (Null Hypothesis)', color='blue')
    ax.plot(x, norm.pdf(x, mu_alt, std_dev), label='H₁: μ = 52 (Alternative Hypothesis)', color='green')

    # Generate a random sample mean
    sample_data = np.random.normal(mu_null, std_dev, sample_size)
    sample_mean = np.mean(sample_data)

    # Calculate the z-score for the sample mean
    z_score = (sample_mean - mu_null) / (std_dev / np.sqrt(sample_size))

    # Plot the sample mean
    ax.plot([sample_mean, sample_mean], [0, norm.pdf(sample_mean, mu_null, std_dev)], color='orange', linewidth=2, label="Sample Mean")
    
    # Plot the critical value and the rejection region for Type I error
    ax.axvline(critical_value, color='red', linestyle='dashed', label='Critical Value')
    ax.fill_between(x, norm.pdf(x, mu_null, std_dev), where=(x >= critical_value), color='red', alpha=0.3, label="Rejection Region (Type I Error)")

    # Annotate the z-score and decision text
    ax.annotate(f"z = {z_score:.2f}", xy=(sample_mean + 2, 0.02), color='orange', fontsize=12)
    
    # Decision based on z-score and p-value (reject H₀ or fail to reject)
    if abs(z_score) > norm.ppf(1 - alpha / 2):
        decision_text = "Reject H₀ (Type I Error Possible)"
        decision_color = 'red'
    else:
        decision_text = "Fail to Reject H₀ (Type II Error Possible)"
        decision_color = 'green'

    # Display the decision on the plot
    ax.text(0.5, 0.95, f"Decision: {decision_text}", ha="center", va="top", fontsize=14, color=decision_color, transform=ax.transAxes)

    # Set labels and title again
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('Type I and Type II Errors in Hypothesis Testing')

    # Add the legend
    ax.legend()

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=n_frames, repeat=False, interval=100)

# Show the plot
plt.show()
