import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.stats import t, norm

# Generate Data for Visualization
df_values = [1, 5, 10, 30]
x = np.linspace(-5, 5, 500)
norm_pdf = norm.pdf(x, 0, 1)  # Normal Distribution

# 1. 3D Plot for Different t-Distributions
def plot_3d_distributions():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx, df in enumerate(df_values):
        y = t.pdf(x, df)
        ax.plot(x, [df] * len(x), y, label=f't-Distribution (df={df})')

    ax.plot(x, [max(df_values) + 5] * len(x), norm_pdf, label='Normal Distribution', color='black')
    
    ax.set_xlabel('X Values')
    ax.set_ylabel('Degrees of Freedom (df)')
    ax.set_zlabel('Probability Density')
    plt.title('3D Plot: Normal vs t-Distribution')
    ax.legend()
    plt.show()

plot_3d_distributions()

# 2. 2D Plot for Static Comparison
def plot_2d_comparison():
    plt.figure(figsize=(10, 6))

    # Plot Normal Distribution
    plt.plot(x, norm_pdf, label='Normal Distribution', color='black', linewidth=2)

    # Plot Different t-Distributions
    for df in df_values:
        plt.plot(x, t.pdf(x, df), label=f't-Distribution (df={df})')

    plt.title('2D Comparison: Normal vs t-Distribution')
    plt.xlabel('X Values')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_2d_comparison()

# 3. Animation for t-Distribution Approaching Normal Distribution
def animate_t_distribution():
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.45)
    ax.set_title('t-Distribution Approaching Normal Distribution')
    ax.set_xlabel('X Values')
    ax.set_ylabel('Probability Density')

    line, = ax.plot([], [], lw=2)
    text_label = ax.text(2, 0.4, '', fontsize=12)

    def init():
        line.set_data([], [])
        return line, text_label

    def update(df):
        y = t.pdf(x, df)
        line.set_data(x, y)
        text_label.set_text(f'Degrees of Freedom: {df}')
        return line, text_label

    ani = animation.FuncAnimation(
        fig, update, frames=range(1, 50), init_func=init, blit=True
    )

    plt.show()

animate_t_distribution()


# Confidence Interval Visualization
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# True population mean (fixed and unknown in real scenarios)
true_mean = 50

# Number of samples and intervals to visualize
num_intervals = 30
sample_size = 30
confidence_level = 0.95

# Collect confidence intervals
intervals = []
contains_mean_count = 0

# Generate and visualize sample-based confidence intervals
plt.figure(figsize=(10, 6))
for i in range(num_intervals):
    # Generate a random sample
    sample = np.random.normal(loc=true_mean, scale=5, size=sample_size)
    sample_mean = np.mean(sample)
    sample_std_error = np.std(sample, ddof=1) / np.sqrt(sample_size)

    # Compute margin of error for the confidence level
    margin_of_error = 1.96 * sample_std_error  # Approximation for 95% confidence level
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    # Store the interval and check if it contains the true mean
    intervals.append((lower_bound, upper_bound))
    contains_mean = lower_bound <= true_mean <= upper_bound
    contains_mean_count += int(contains_mean)

    # Plot the confidence interval
    color = 'green' if contains_mean else 'red'
    plt.plot([i, i], [lower_bound, upper_bound], color=color, marker='o')

# Plot the true population mean
plt.axhline(y=true_mean, color='blue', linestyle='--', label='True Population Mean')
plt.title('Confidence Intervals (95%) and the True Mean')
plt.xlabel('Sample Index')
plt.ylabel('Interval Range')
plt.legend()

plt.show()

# Display result summary
print(f"Out of {num_intervals} intervals, {contains_mean_count} contained the true mean.")
print("This demonstrates that confidence intervals contain the true mean approximately 95% of the time.")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic population data (e.g., heights)
population = np.random.normal(loc=170, scale=10, size=10000)
population_mean = np.mean(population)

# Function to generate confidence intervals
def compute_confidence_interval(sample, confidence=0.95):
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    margin_of_error = 1.96 * (sample_std / np.sqrt(len(sample)))  # 95% confidence interval
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    return sample_mean, lower_bound, upper_bound

# Generate multiple samples and store confidence intervals
num_samples = 30
sample_size = 50
samples = [np.random.choice(population, size=sample_size, replace=False) for _ in range(num_samples)]
intervals = [compute_confidence_interval(sample) for sample in samples]

means = [interval[0] for interval in intervals]
lower_bounds = [interval[1] for interval in intervals]
upper_bounds = [interval[2] for interval in intervals]

# --- 3D Visualization ---
fig_3d = plt.figure(figsize=(10, 6))
ax = fig_3d.add_subplot(111, projection='3d')
x_vals = np.arange(num_samples)
y_vals = means
z_vals = np.zeros(num_samples)

ax.bar3d(x_vals, z_vals, lower_bounds, dx=0.5, dy=0.5, dz=np.array(upper_bounds) - np.array(lower_bounds), color='skyblue', alpha=0.6)
ax.axhline(population_mean, color='red', linestyle='--', label='True Population Mean')

ax.set_xlabel('Sample Index')
ax.set_ylabel('Sample Mean')
ax.set_zlabel('Confidence Interval')
ax.set_title('3D Confidence Intervals for Sample Means')
plt.legend()
plt.show()

# --- Animated Visualization ---
fig_anim, ax_anim = plt.subplots(figsize=(10, 5))

ax_anim.axhline(population_mean, color='red', linestyle='--', label='True Population Mean')
ax_anim.set_xlim(0, num_samples)
ax_anim.set_ylim(150, 190)
ax_anim.set_title('Confidence Interval Animation')
ax_anim.set_xlabel('Sample Index')
ax_anim.set_ylabel('Value')

lines = []

def init():
    ax_anim.legend()
    return lines

def update(frame):
    sample_mean = means[frame]
    lower = lower_bounds[frame]
    upper = upper_bounds[frame]

    # Plot the confidence interval for the current frame
    line = ax_anim.plot([frame, frame], [lower, upper], color='blue')[0]
    point = ax_anim.plot(frame, sample_mean, 'bo')[0]
    lines.append(line)
    lines.append(point)
    return lines

ani = FuncAnimation(fig_anim, update, frames=num_samples, init_func=init, blit=True, repeat=False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Given parameters
sigma = 25  # standard deviation
z_alpha_2 = 1.96  # critical value for 95% confidence

# Function to calculate the required sample size
def required_sample_size(moe, sigma, z_alpha_2):
    return (z_alpha_2 * sigma / moe) ** 2

# Values for margin of error (moe) and calculated sample sizes
moe_values = np.linspace(1, 10, 50)  # Margin of error from 1cm to 10cm
sample_sizes = required_sample_size(moe_values, sigma, z_alpha_2)

# 3D Visualization of Margin of Error vs Sample Size
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create data for the 3D plot
X, Y = np.meshgrid(moe_values, z_alpha_2)
Z = required_sample_size(X, sigma, Y)

# Scatter Plot
ax.scatter(moe_values, sample_sizes, z_alpha_2, color='teal', marker='o')

ax.set_title('Sample Size vs Margin of Error (3D Visualization)')
ax.set_xlabel('Margin of Error (cm)')
ax.set_ylabel('Sample Size (n)')
ax.set_zlabel('Z-value')

plt.show()

# Animated Visualization
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(1, 10)
ax.set_ylim(0, 800)

line, = ax.plot([], [], lw=2)

ax.set_title("Sample Size vs Margin of Error (Animated Plot)")
ax.set_xlabel("Margin of Error (cm)")
ax.set_ylabel("Sample Size (n)")

# Initialization function for animation
def init():
    line.set_data([], [])
    return line,

# Animation function
def animate(i):
    x = moe_values[:i]
    y = sample_sizes[:i]
    line.set_data(x, y)
    return line,

ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=len(moe_values), interval=100, blit=True
)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Given problem values
mean_height = 170  # Mean sample height in cm
std_dev = 25  # Population standard deviation in cm
confidence_level = 1.96  # For 95% confidence level

# Function to calculate margin of error
def calculate_margin_of_error(std_dev, sample_size, z_value):
    return z_value * (std_dev / np.sqrt(sample_size))

# --- 3D Plot: Margin of Error vs. Sample Size ---
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')

# Generate sample sizes and corresponding margin of errors
sample_sizes = np.arange(10, 500, 10)
margin_errors = calculate_margin_of_error(std_dev, sample_sizes, confidence_level)

# Plotting the 3D graph
ax.plot(sample_sizes, margin_errors, zs=mean_height, zdir='z', label='Margin of Error Curve', color='purple')
ax.set_xlabel('Sample Size (n)')
ax.set_ylabel('Margin of Error (cm)')
ax.set_zlabel('Mean Height (cm)')
ax.set_title('3D Visualization: Margin of Error vs Sample Size')
plt.show()

# --- Animated Confidence Interval Plot ---
fig, ax = plt.subplots(figsize=(8, 5))

# Sample sizes for the animation
sample_sizes_anim = np.arange(10, 500, 10)
mean_values = [mean_height] * len(sample_sizes_anim)

# Setup the plot
line, = ax.plot([], [], 'o-', lw=2, color='orange')
ax.set_xlim(0, 500)
ax.set_ylim(150, 190)
ax.set_title("Dynamic Confidence Interval Plot")
ax.set_xlabel("Sample Size (n)")
ax.set_ylabel("Height Confidence Interval (cm)")

# Initialization function for the animation
def init():
    line.set_data([], [])
    return line,

# Update function for each frame
def update(frame):
    n = sample_sizes_anim[frame]
    margin_error = calculate_margin_of_error(std_dev, n, confidence_level)
    lower_bound = mean_height - margin_error
    upper_bound = mean_height + margin_error

    # Update plot points
    line.set_data([n, n], [lower_bound, upper_bound])
    ax.set_title(f"Sample Size = {n}, Margin of Error = {margin_error:.2f} cm")

    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=len(sample_sizes_anim), init_func=init, blit=True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# True population mean (fixed)
true_mean = 170

# Generate 10 random sample means around the true mean
np.random.seed(42)
sample_means = np.random.normal(loc=true_mean, scale=5, size=10)

# Confidence interval margin of error (example value)
margin_of_error = 7

# Plot setup
plt.figure(figsize=(10, 6))
plt.axvline(true_mean, color='red', linestyle='--', label='True Mean (170 cm)')

# Plot confidence intervals for each sample mean
for i, sample_mean in enumerate(sample_means):
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    plt.plot([lower_bound, upper_bound], [i, i], color='blue', marker='|', markersize=10)
    plt.scatter(sample_mean, i, color='blue', label='Sample Mean' if i == 0 else "")

# Annotate plot
plt.title('Confidence Intervals for Different Sample Means')
plt.xlabel('Height (cm)')
plt.ylabel('Sample Index')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Given data for illustration
sample_mean = 170
confidence_level = 0.95
critical_value = 1.96  # for 95% confidence level
std_dev = 25
sample_size = 49

# Step 1: Calculate standard error
standard_error = std_dev / np.sqrt(sample_size)

# Step 2: Calculate margin of error
margin_of_error = critical_value * standard_error

# Step 3: Calculate confidence interval
lower_bound = sample_mean - margin_of_error
upper_bound = sample_mean + margin_of_error

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))

# Plot sample mean
ax.axvline(sample_mean, color='blue', linestyle='--', label=f"Sample Mean = {sample_mean} cm")

# Plot confidence interval
ax.axvspan(lower_bound, upper_bound, color='lightgreen', alpha=0.3, 
            label=f"95% Confidence Interval ({lower_bound:.2f} cm, {upper_bound:.2f} cm)")

# Plot lower and upper bounds
ax.axvline(lower_bound, color='green', linestyle=':', label=f"Lower Bound = {lower_bound:.2f} cm")
ax.axvline(upper_bound, color='green', linestyle=':', label=f"Upper Bound = {upper_bound:.2f} cm")

# Annotate important points
ax.annotate("Sample Mean", xy=(sample_mean, 0.1), xytext=(sample_mean + 2, 0.2),
             arrowprops=dict(facecolor='blue', arrowstyle='->'), fontsize=10)

ax.annotate("Lower Bound", xy=(lower_bound, 0.05), xytext=(lower_bound - 15, 0.1),
             arrowprops=dict(facecolor='green', arrowstyle='->'), fontsize=10)

ax.annotate("Upper Bound", xy=(upper_bound, 0.05), xytext=(upper_bound + 5, 0.1),
             arrowprops=dict(facecolor='green', arrowstyle='->'), fontsize=10)

# Labels and legend
ax.set_title("Confidence Interval Visualization")
ax.set_xlabel("Height (cm)")
ax.set_yticks([])  # Hide y-axis
ax.legend()
plt.show()
