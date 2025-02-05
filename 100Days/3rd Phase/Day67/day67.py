import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mu = 66.7  # Null hypothesis mean
sigma = 2  # Population standard deviation
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

# Plot normal distribution
plt.plot(x, norm.pdf(x, mu, sigma), label='Null Hypothesis (μ=66.7)')
plt.axvline(68.442, color='red', linestyle='dashed', linewidth=1.5, label='Sample Mean (68.442)')
plt.fill_betweenx(norm.pdf(x, mu, sigma), 68, x, where=(x > 68), color='lightcoral', alpha=0.5, label='Critical Region')
plt.title("Right-Tailed Hypothesis Test Visualization")
plt.xlabel("Height (inches)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

mu_values = np.linspace(65, 70, 100)
sigma_values = np.linspace(1, 3, 100)
mu_grid, sigma_grid = np.meshgrid(mu_values, sigma_values)
pdf_values = norm.pdf(68.442, mu_grid, sigma_grid)

ax.plot_surface(mu_grid, sigma_grid, pdf_values, cmap='viridis', alpha=0.7)
ax.set_title("Surface Plot of Test Statistic")
ax.set_xlabel("Mean (μ)")
ax.set_ylabel("Standard Deviation (σ)")
ax.set_zlabel("Probability Density")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.animation as animation

# Parameters
mu = 66.7  # Null hypothesis mean
sigma = 2  # Population standard deviation
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
critical_value = 68.442  # Hypothetical sample mean for testing

fig, ax = plt.subplots()
line, = ax.plot(x, norm.pdf(x, mu, sigma), label='Null Hypothesis Distribution (μ=66.7)')
sample_point, = ax.plot([], [], 'ro', markersize=8, label='Sample Mean')
text = ax.text(mu + 1, 0.1, '', fontsize=12)

ax.set_title("Dynamic Visualization of Hypothesis Testing")
ax.set_xlabel("Sample Mean (x̄)")
ax.set_ylabel("Probability Density")
ax.legend()

# Initialize plot elements
def init():
    sample_point.set_data([], [])
    text.set_text('')
    return sample_point, text

# Animation update function
def update(frame):
    sample_mean = mu + frame * 0.1
    y = norm.pdf(sample_mean, mu, sigma)
    
    # Update point
    sample_point.set_data([sample_mean], [y])  # Pass single values as lists
    
    # Update p-value text
    p_value = 1 - norm.cdf(sample_mean, mu, sigma)
    text.set_text(f'Sample Mean: {sample_mean:.2f}, p-value: {p_value:.4f}')
    
    return sample_point, text

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 40), init_func=init, blit=False)

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.animation as animation

# Parameters
mu = 66.7  # Null hypothesis mean
sigma = 2  # Population standard deviation
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
critical_value_left = 64.5  # Example for left-tailed
critical_value_right = 68.5  # Example for right-tailed
critical_value_two_tail = 70.0  # Arbitrary for demonstration

fig, ax = plt.subplots()
line, = ax.plot(x, norm.pdf(x, mu, sigma), label='Null Hypothesis Distribution (μ=66.7)')
sample_point, = ax.plot([], [], 'ro', markersize=8, label='Sample Mean')
text = ax.text(mu + 1, 0.1, '', fontsize=12)

ax.set_title("Dynamic Visualization of Hypothesis Tests")
ax.set_xlabel("Sample Mean (x̄)")
ax.set_ylabel("Probability Density")
ax.legend()

# Initialize plot elements
def init():
    sample_point.set_data([], [])
    text.set_text('')
    return sample_point, text

# Animation update function
def update(frame):
    sample_mean = mu + (frame - 20) * 0.2  # Dynamic shift left and right
    y = norm.pdf(sample_mean, mu, sigma)

    # Update point position
    sample_point.set_data([sample_mean], [y])

    # Determine test type based on frame number
    if frame < 20:
        # Left-tailed test
        p_value = norm.cdf(sample_mean, mu, sigma)
        test_type = 'Left-tailed'
    elif frame < 40:
        # Right-tailed test
        p_value = 1 - norm.cdf(sample_mean, mu, sigma)
        test_type = 'Right-tailed'
    else:
        # Two-tailed test
        p_value = 2 * min(norm.cdf(sample_mean, mu, sigma), 1 - norm.cdf(sample_mean, mu, sigma))
        test_type = 'Two-tailed'

    # Update text
    text.set_text(f'Sample Mean: {sample_mean:.2f}, p-value: {p_value:.4f} ({test_type})')

    return sample_point, text

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 60), init_func=init, blit=False)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Given parameters
mu_0 = 66.7  # Null hypothesis population mean
sigma = 3  # Population standard deviation
n = 10  # Sample size
alpha = 0.05  # Significance level

# Calculate standard error
standard_error = sigma / np.sqrt(n)

# Z-critical values for each test
z_right = (68.442 - mu_0) / standard_error  # Right-tailed
z_two_tail = 1.742 / standard_error  # Two-tailed (distance from mean)
z_left = (64.252 - mu_0) / standard_error  # Left-tailed

# Generate x values for plotting
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, 0, 1)

# Plot the standard normal distribution
plt.figure(figsize=(12, 6))
plt.plot(x, y, color='black', lw=2, label="Standard Normal PDF")

# Right-tail rejection region
x_right = np.linspace(z_right, 4, 500)
y_right = norm.pdf(x_right, 0, 1)
plt.fill_between(x_right, y_right, color='red', alpha=0.4, label='Right-Tail Rejection')

# Two-tailed rejection region
x_two_left = np.linspace(-4, -z_two_tail, 500)
x_two_right = np.linspace(z_two_tail, 4, 500)
plt.fill_between(x_two_left, norm.pdf(x_two_left, 0, 1), color='blue', alpha=0.3, label='Left Tail (Two-tailed)')
plt.fill_between(x_two_right, norm.pdf(x_two_right, 0, 1), color='blue', alpha=0.3, label='Right Tail (Two-tailed)')

# Left-tail rejection region
x_left = np.linspace(-4, z_left, 500)
plt.fill_between(x_left, norm.pdf(x_left, 0, 1), color='green', alpha=0.4, label='Left-Tail Rejection')

# Add axis labels and legend
plt.axvline(z_right, color='red', linestyle='--', label=f'Observed Z (Right) = {z_right:.3f}')
plt.axvline(z_left, color='green', linestyle='--', label=f'Observed Z (Left) = {z_left:.3f}')
plt.axvline(-z_two_tail, color='blue', linestyle='--', label=f'Left Bound (Two-Tail) = {-z_two_tail:.3f}')
plt.axvline(z_two_tail, color='blue', linestyle='--', label=f'Right Bound (Two-Tail) = {z_two_tail:.3f}')

plt.title('Hypothesis Testing - Standard Normal Distribution')
plt.xlabel('Z-score')
plt.ylabel('Probability Density')
plt.legend(loc='upper left')
plt.grid(True)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define standard normal distribution parameters
mu = 0
sigma = 1
x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)
Z = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-0.5 * (X ** 2 + Y ** 2))

# Create a 3D plot for the distribution and rejection regions
fig_3d = plt.figure(figsize=(12, 8))
ax = fig_3d.add_subplot(111, projection='3d')

# Plot surface
surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# Highlight rejection regions
ax.contour(X, Y, Z, zdir='z', offset=0, levels=10, cmap='coolwarm')
ax.set_title("3D Gaussian Distribution for Hypothesis Testing")
ax.set_xlabel("X-axis (Test Statistics)")
ax.set_ylabel("Y-axis (Test Statistics)")
ax.set_zlabel("Density")
plt.colorbar(surface, shrink=0.5, aspect=10)
plt.show()

# Create animated visualization
fig_anim, ax_anim = plt.subplots(figsize=(8, 4))

# Plot initial standard normal curve
x = np.linspace(-4, 4, 400)
pdf = (1 / (np.sqrt(2 * np.pi))) * np.exp(-x**2 / 2)
line, = ax_anim.plot(x, pdf, color='black', lw=2)
ax_anim.axhline(0, color='black', linewidth=0.5)
ax_anim.set_title("Animated Visualization of p-Values and Rejection Regions")
ax_anim.set_xlim(-4, 4)
ax_anim.set_ylim(0, 0.5)
ax_anim.set_xlabel("Z-Statistic")
ax_anim.set_ylabel("Density")

# Markers for rejection regions
right_tail = ax_anim.axvline(1.645, color='red', linestyle='--', label="Right Tail (α = 0.05)")
left_tail = ax_anim.axvline(-1.645, color='green', linestyle='--', label="Left Tail (α = 0.05)")
two_tail_pos = ax_anim.axvline(1.96, color='blue', linestyle='--', label="Two-Tail (α = 0.025)")
two_tail_neg = ax_anim.axvline(-1.96, color='blue', linestyle='--')

# Animation function
def update(frame):
    threshold = frame / 100
    line.set_ydata((1 / (np.sqrt(2 * np.pi))) * np.exp(-x**2 / (2 * threshold**2)))
    return line,

anim = FuncAnimation(fig_anim, update, frames=np.arange(50, 150, 1), interval=100)

plt.legend()
plt.show()



import plotly.graph_objects as go
import numpy as np

# Function to calculate the critical value based on alpha
def critical_value(alpha, mu_0, sigma, n):
    std_error = sigma / np.sqrt(n)
    return norm.ppf(1 - alpha, mu_0, std_error)

# Alpha range
alpha_values = np.linspace(0.01, 0.1, 100)
critical_values = [critical_value(alpha, mu_0, sigma, n) for alpha in alpha_values]

# 3D plot
fig = go.Figure(data=[go.Scatter3d(
    x=alpha_values,
    y=critical_values,
    z=np.zeros_like(alpha_values),
    mode='lines+markers',
    marker=dict(size=5, color='red', opacity=0.7)
)])

# Add labels and title
fig.update_layout(
    title="3D Visualization of Critical Value vs Alpha",
    scene=dict(
        xaxis_title='Alpha',
        yaxis_title='Critical Value',
        zaxis_title='Density'
    ),
    showlegend=False
)

# Show plot
fig.show()


import plotly.graph_objects as go
import numpy as np

# Function to calculate the critical value based on alpha
def critical_value(alpha, mu_0, sigma, n):
    std_error = sigma / np.sqrt(n)
    return norm.ppf(1 - alpha, mu_0, std_error)

# Alpha range
alpha_values = np.linspace(0.01, 0.1, 100)
critical_values = [critical_value(alpha, mu_0, sigma, n) for alpha in alpha_values]

# 3D plot
fig = go.Figure(data=[go.Scatter3d(
    x=alpha_values,
    y=critical_values,
    z=np.zeros_like(alpha_values),
    mode='lines+markers',
    marker=dict(size=5, color='red', opacity=0.7)
)])

# Add labels and title
fig.update_layout(
    title="3D Visualization of Critical Value vs Alpha",
    scene=dict(
        xaxis_title='Alpha',
        yaxis_title='Critical Value',
        zaxis_title='Density'
    ),
    showlegend=False
)

# Show plot
fig.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

# Parameters
mu_0 = 66.7
sigma = 3
n = 10

# Generate x values for the sample mean distribution
x = np.linspace(mu_0 - 4 * sigma / np.sqrt(n), mu_0 + 4 * sigma / np.sqrt(n), 1000)

# Setup the plot
fig, ax = plt.subplots(figsize=(10,6))
line, = ax.plot([], [], lw=2, label='Sampling Distribution')
critical_line, = ax.plot([], [], 'k--', label='Critical Value')

# Initialize the plot
def init():
    ax.set_xlim(mu_0 - 4 * sigma / np.sqrt(n), mu_0 + 4 * sigma / np.sqrt(n))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Density')
    ax.set_title('Animation of Critical Value with Changing Alpha')
    ax.legend()
    return line, critical_line

# Update function for animation
def update(alpha):
    std_error = sigma / np.sqrt(n)
    critical_value = norm.ppf(1 - alpha, mu_0, std_error)
    y = norm.pdf(x, mu_0, std_error)
    line.set_data(x, y)
    critical_line.set_data([critical_value, critical_value], [0, max(y)])
    return line, critical_line

# Animate the plot
ani = FuncAnimation(fig, update, frames=np.linspace(0.01, 0.1, 100), init_func=init, blit=True)

# Show plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mu = 66.7   # Population mean (null hypothesis)
sigma = 3   # Population standard deviation
n = 10      # Sample size
alpha = 0.05  # Significance level

# Critical value (K_alpha) for a right-tailed test at alpha = 0.05
critical_value = norm.ppf(1 - alpha, loc=mu, scale=sigma / np.sqrt(n))

# Observed sample mean (for demonstration)
observed_statistic = 68.44

# Generate the x values for the plot
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
y = norm.pdf(x, loc=mu, scale=sigma / np.sqrt(n))

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the normal distribution curve
plt.plot(x, y, label='Sampling Distribution under H₀', color='blue')

# Highlight the rejection region (right tail)
x_rejection = np.linspace(critical_value, mu + 4 * sigma, 1000)
y_rejection = norm.pdf(x_rejection, loc=mu, scale=sigma / np.sqrt(n))
plt.fill_between(x_rejection, y_rejection, color='red', alpha=0.5, label='Rejection Region')

# Plot the critical value as a vertical line
plt.axvline(x=critical_value, color='black', linestyle='--', label=f'Critical Value = {critical_value:.2f}')

# Plot the observed statistic
plt.axvline(x=observed_statistic, color='green', linestyle='-', label=f'Observed Statistic = {observed_statistic:.2f}')

# Labels and title
plt.title('Hypothesis Test: Right-Tailed Test with Critical Value and Rejection Region')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

# Parameters
mu = 66.7   # Population mean (null hypothesis)
sigma = 3   # Population standard deviation
n = 10      # Sample size
alpha = 0.05  # Significance level
confidence_level = 0.95
critical_value = norm.ppf(1 - alpha, loc=mu, scale=sigma / np.sqrt(n))  # Right tail critical value

# Confidence Interval bounds (for 95% confidence)
z_value = norm.ppf((1 + confidence_level) / 2)  # Z-score for 95% CI
ci_lower = mu - z_value * (sigma / np.sqrt(n))
ci_upper = mu + z_value * (sigma / np.sqrt(n))

# Range of values for the x-axis (test statistics)
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
y = np.linspace(-0.1, 1.1, 100)  # p-value from 0 to 1

# Create meshgrid for 3D surface plot
X, Y = np.meshgrid(x, y)

# Calculate the PDF values for the surface plot
Z = norm.pdf(X, loc=mu, scale=sigma / np.sqrt(n))

# Plotting the 3D surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot for the distribution
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Highlight the critical value region (right-tail)
ax.plot_surface(X, np.full_like(X, 1), Z, color='r', alpha=0.3, label='Rejection Region (Critical Value)')

# Plot the critical value (K_alpha) as a vertical line on the surface
ax.plot([critical_value, critical_value], [0, 1], [0, norm.pdf(critical_value, loc=mu, scale=sigma / np.sqrt(n))],
        color='k', linestyle='--', linewidth=3, label=f'Critical Value = {critical_value:.2f}')

# Plot the Confidence Interval (CI) region
ax.plot([ci_lower, ci_lower], [0, 1], [0, norm.pdf(ci_lower, loc=mu, scale=sigma / np.sqrt(n))], color='b', linestyle='-.', linewidth=2)
ax.plot([ci_upper, ci_upper], [0, 1], [0, norm.pdf(ci_upper, loc=mu, scale=sigma / np.sqrt(n))], color='b', linestyle='-.', linewidth=2)

# Set the plot labels
ax.set_xlabel('Sample Mean (X)')
ax.set_ylabel('p-value Area')
ax.set_zlabel('Density')
ax.set_title('3D Visualization: Hypothesis Testing with Critical Value, P-value, and Confidence Interval')

# Add the legend
ax.legend(loc='upper left')

# Show the plot
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

# Parameters
mu = 66.7   # Population mean (null hypothesis)
sigma = 3   # Population standard deviation
n = 10      # Sample size
alpha = 0.05  # Significance level
confidence_level = 0.95

# Critical value for right-tailed test
critical_value = norm.ppf(1 - alpha, loc=mu, scale=sigma / np.sqrt(n))

# Confidence Interval (95%)
z_value = norm.ppf((1 + confidence_level) / 2)
ci_lower = mu - z_value * (sigma / np.sqrt(n))
ci_upper = mu + z_value * (sigma / np.sqrt(n))

# Range of values for the x-axis (test statistics)
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)

# Create the normal distribution (H₀ under null hypothesis)
y = norm.pdf(x, loc=mu, scale=sigma / np.sqrt(n))

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the normal distribution curve
ax.plot(x, np.zeros_like(x), y, label='Normal Distribution (H₀)', color='g')

# Highlight the critical value region (rejection region)
ax.plot([critical_value, critical_value], [0, 0], [0, norm.pdf(critical_value, loc=mu, scale=sigma / np.sqrt(n))],
        color='r', linestyle='--', label=f'Critical Value = {critical_value:.2f}')

# Plot Confidence Interval bounds
ax.plot([ci_lower, ci_lower], [0, 0], [0, norm.pdf(ci_lower, loc=mu, scale=sigma / np.sqrt(n))], color='b', linestyle='-.', label=f'CI Lower = {ci_lower:.2f}')
ax.plot([ci_upper, ci_upper], [0, 0], [0, norm.pdf(ci_upper, loc=mu, scale=sigma / np.sqrt(n))], color='b', linestyle='-.', label=f'CI Upper = {ci_upper:.2f}')

# Set labels
ax.set_xlabel('Sample Mean (X)')
ax.set_ylabel('p-value Area')
ax.set_zlabel('Density')
ax.set_title('Simplified 3D Visualization: Hypothesis Testing')

# Add the legend
ax.legend()

# Show the plot
plt.show()

