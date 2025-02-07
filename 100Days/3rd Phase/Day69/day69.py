import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Sample data
us_heights = np.random.normal(68.442, 3.113, 10)
argentina_heights = np.random.normal(65.949, 3.106, 9)

# 2D Plot
plt.figure(figsize=(10, 5))
plt.scatter(range(1, 11), us_heights, color='blue', label='US Heights')
plt.scatter(range(1, 10), argentina_heights, color='orange', label='Argentina Heights')
plt.axhline(np.mean(us_heights), color='blue', linestyle='--', label='US Mean')
plt.axhline(np.mean(argentina_heights), color='orange', linestyle='--', label='Argentina Mean')
plt.title('2D Plot of US vs Argentina Heights')
plt.xlabel('Sample Index')
plt.ylabel('Height (inches)')
plt.legend()
plt.show()

# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x_us = np.arange(1, 11)
y_us = np.ones_like(x_us) * 1
z_us = us_heights

x_arg = np.arange(1, 10)
y_arg = np.ones_like(x_arg) * 2
z_arg = argentina_heights

ax.scatter(x_us, y_us, z_us, color='blue', label='US Heights')
ax.scatter(x_arg, y_arg, z_arg, color='orange', label='Argentina Heights')

ax.set_title('3D Plot of US vs Argentina Heights')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Country Group (1: US, 2: Argentina)')
ax.set_zlabel('Height (inches)')
ax.legend()
plt.show()

# Animation Plot
fig, ax = plt.subplots(figsize=(10, 5))

x_data, y_data = [], []
line, = ax.plot([], [], color='purple')
ax.set_xlim(0, 10)
ax.set_ylim(60, 75)
ax.set_title('Animated Plot: Height Data Accumulation')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Height (inches)')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    x_data.append(frame)
    y_data.append(us_heights[frame - 1] if frame <= len(us_heights) else argentina_heights[frame - len(us_heights) - 1])
    line.set_data(x_data, y_data)
    return line,

ani = FuncAnimation(fig, update, frames=list(range(1, 20)), init_func=init, blit=True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
np.random.seed(42)
us_heights = np.random.normal(68.442, 3.113, 10)
argentina_heights = np.random.normal(65.949, 3.106, 9)

# Plot distributions
plt.figure(figsize=(12, 6))
sns.histplot(us_heights, kde=True, color='blue', label='US Heights', binwidth=0.5)
sns.histplot(argentina_heights, kde=True, color='orange', label='Argentina Heights', binwidth=0.5)

# Plot sample means
plt.axvline(np.mean(us_heights), color='blue', linestyle='--', label=f'US Mean: {np.mean(us_heights):.2f}')
plt.axvline(np.mean(argentina_heights), color='orange', linestyle='--', label=f'Argentina Mean: {np.mean(argentina_heights):.2f}')

plt.title('Height Distributions: US vs Argentina')
plt.xlabel('Height (inches)')
plt.ylabel('Frequency')
plt.legend()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Data for US and Argentina heights
x_us = np.arange(1, 11)
y_us = np.ones_like(x_us) * 1
z_us = us_heights

x_arg = np.arange(1, 10)
y_arg = np.ones_like(x_arg) * 2
z_arg = argentina_heights

# Scatter plot in 3D
ax.scatter(x_us, y_us, z_us, color='blue', label='US Heights')
ax.scatter(x_arg, y_arg, z_arg, color='orange', label='Argentina Heights')

# Axis labels
ax.set_title('3D Plot of Height Samples')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Country Group (1: US, 2: Argentina)')
ax.set_zlabel('Height (inches)')
ax.legend()
plt.show()


from matplotlib.animation import FuncAnimation

# Initialize plot
fig, ax = plt.subplots(figsize=(12, 6))
x_data, y_data = [], []
line, = ax.plot([], [], color='purple', linewidth=2)
sample_indices = list(range(1, len(us_heights) + len(argentina_heights) + 1))
all_samples = np.concatenate((us_heights, argentina_heights))

# Plot settings
ax.set_xlim(0, len(sample_indices) + 1)
ax.set_ylim(60, 75)
ax.set_title('Sample Mean Convergence Animation')
ax.set_xlabel('Sample Count')
ax.set_ylabel('Sample Mean (inches)')

# Initialize plot elements
def init():
    line.set_data([], [])
    return line,

# Update function for animation
def update(frame):
    x_data.append(frame)
    y_data.append(np.mean(all_samples[:frame]))
    line.set_data(x_data, y_data)
    return line,

# Animate
ani = FuncAnimation(fig, update, frames=sample_indices, init_func=init, blit=True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
np.random.seed(42)
us_heights = np.random.normal(68.442, 3.113, 10)
argentina_heights = np.random.normal(65.949, 3.106, 9)

# Plot distributions
plt.figure(figsize=(12, 6))
sns.histplot(us_heights, kde=True, color='blue', label='US Heights', binwidth=0.5)
sns.histplot(argentina_heights, kde=True, color='orange', label='Argentina Heights', binwidth=0.5)

# Plot sample means
us_mean = np.mean(us_heights)
arg_mean = np.mean(argentina_heights)

plt.axvline(us_mean, color='blue', linestyle='--', label=f'US Mean: {us_mean:.2f}')
plt.axvline(arg_mean, color='orange', linestyle='--', label=f'Argentina Mean: {arg_mean:.2f}')

# Annotate the means
plt.annotate(f'US Mean: {us_mean:.2f}', xy=(us_mean, 1), xytext=(us_mean + 1, 2),
             arrowprops=dict(facecolor='blue', arrowstyle='->'))
plt.annotate(f'Argentina Mean: {arg_mean:.2f}', xy=(arg_mean, 1), xytext=(arg_mean - 3, 2),
             arrowprops=dict(facecolor='orange', arrowstyle='->'))

plt.title('Height Distributions: US vs Argentina with Sample Means')
plt.xlabel('Height (inches)')
plt.ylabel('Frequency')
plt.legend()
plt.show()



from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

# Data for US and Argentina heights
x_us = np.arange(1, 11)
y_us = np.ones_like(x_us) * 1
z_us = us_heights

x_arg = np.arange(1, 10)
y_arg = np.ones_like(x_arg) * 2
z_arg = argentina_heights

# Scatter plot in 3D
us_points = ax.scatter(x_us, y_us, z_us, color='blue', label='US Heights')
arg_points = ax.scatter(x_arg, y_arg, z_arg, color='orange', label='Argentina Heights')

# Annotate a few key points
ax.text(x_us[0], y_us[0], z_us[0], f'{z_us[0]:.2f}', color='blue')
ax.text(x_arg[0], y_arg[0], z_arg[0], f'{z_arg[0]:.2f}', color='orange')

# Set axis labels
ax.set_title('3D Plot of Height Samples by Country Group')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Country Group (1: US, 2: Argentina)')
ax.set_zlabel('Height (inches)')
ax.legend()
plt.show()


from matplotlib.animation import FuncAnimation

# Initialize plot
fig, ax = plt.subplots(figsize=(12, 6))
x_data, y_data = [], []
line, = ax.plot([], [], color='purple', linewidth=2)
sample_indices = list(range(1, len(us_heights) + len(argentina_heights) + 1))
all_samples = np.concatenate((us_heights, argentina_heights))

# Plot settings
ax.set_xlim(0, len(sample_indices) + 1)
ax.set_ylim(60, 75)
ax.set_title('Sample Mean Convergence Animation')
ax.set_xlabel('Sample Count')
ax.set_ylabel('Sample Mean (inches)')
mean_text = ax.text(1, 72, '', fontsize=12, color='purple')

# Initialize plot elements
def init():
    line.set_data([], [])
    mean_text.set_text('')
    return line, mean_text

# Update function for animation
def update(frame):
    x_data.append(frame)
    current_mean = np.mean(all_samples[:frame])
    y_data.append(current_mean)
    line.set_data(x_data, y_data)
    mean_text.set_text(f'Sample {frame}: Mean = {current_mean:.2f}')
    return line, mean_text

# Animate
ani = FuncAnimation(fig, update, frames=sample_indices, init_func=init, blit=True)
plt.show()




import matplotlib.pyplot as plt
import numpy as np

# Data
participants = list(range(1, 11))
before_weights = [78, 82, 75, 85, 90, 88, 92, 76, 80, 89]
after_weights = [76, 80, 73, 83, 87, 86, 89, 74, 78, 87]

# Bar width and positioning
bar_width = 0.35
index = np.arange(len(participants))

plt.figure(figsize=(12, 6))

# Bar plots for before and after weights
plt.bar(index, before_weights, bar_width, label='Before Weight', color='skyblue')
plt.bar(index + bar_width, after_weights, bar_width, label='After Weight', color='salmon')

# Labels and title
plt.title('Before vs After Weight Comparison')
plt.xlabel('Participant Number')
plt.ylabel('Weight (kg)')
plt.xticks(index + bar_width / 2, participants)

# Annotating each bar for clarity
for i in range(len(participants)):
    plt.text(i, before_weights[i] + 0.5, f"{before_weights[i]} kg", ha='center', color='blue')
    plt.text(i + bar_width, after_weights[i] + 0.5, f"{after_weights[i]} kg", ha='center', color='red')

plt.legend()
plt.show()


import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, len(participants) + 1)
ax.set_ylim(0, max(before_weights) + 5)
ax.set_title('Weight Loss Tracking Animation')
ax.set_xlabel('Participant Number')
ax.set_ylabel('Weight (kg)')

# Plot lines for before and after weights
before_line, = ax.plot([], [], marker='o', color='blue', label='Before Weight')
after_line, = ax.plot([], [], marker='^', color='red', label='After Weight')

def init():
    before_line.set_data([], [])
    after_line.set_data([], [])
    return before_line, after_line

def update(frame):
    before_line.set_data(participants[:frame], before_weights[:frame])
    after_line.set_data(participants[:frame], after_weights[:frame])
    return before_line, after_line

ani = animation.FuncAnimation(fig, update, frames=len(participants) + 1,
                              init_func=init, blit=True, interval=500)

plt.legend()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

# Data
participants = np.arange(1, 11)
before_weights = [78, 82, 75, 85, 90, 88, 92, 76, 80, 89]
after_weights = [76, 80, 73, 83, 87, 86, 89, 74, 78, 87]

# Bar properties
bar_width = 0.4

# Plotting the bars for before and after weights
ax.bar(participants, before_weights, zs=0, zdir='y', color='skyblue', width=bar_width, label='Before Weight')
ax.bar(participants, after_weights, zs=1, zdir='y', color='salmon', width=bar_width, label='After Weight')

# Labels and titles
ax.set_title('3D Comparison of Before vs After Weights')
ax.set_xlabel('Participant Number')
ax.set_ylabel('Weight Group (Before/After)')
ax.set_zlabel('Weight (kg)')

# Set ticks for clarity
ax.set_yticks([0, 1])
ax.set_yticklabels(['Before', 'After'])

plt.legend()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
purchase_A = [50, 55, 45, 60, 53, 48, 52, 56, 49, 51]  # Purchase data for A
purchase_B = [55, 58, 60, 61, 59, 57, 62, 65, 63, 61]  # Purchase data for B

# Create a DataFrame for easy plotting
import pandas as pd
data = pd.DataFrame({
    'Purchase Amount': purchase_A + purchase_B,
    'Design': ['A'] * len(purchase_A) + ['B'] * len(purchase_B)
})

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Design', y='Purchase Amount', data=data)
plt.title('Purchase Amount Distribution for Design A and B')
plt.show()

import matplotlib.pyplot as plt

# Example data
time = [1, 2, 3, 4, 5]
conversion_rate_A = [0.25, 0.28, 0.30, 0.35, 0.38]
conversion_rate_B = [0.20, 0.22, 0.25, 0.29, 0.32]

# Create a 2D plot
plt.figure(figsize=(8, 6))
plt.plot(time, conversion_rate_A, label="Strategy A", color='blue', marker='o')
plt.plot(time, conversion_rate_B, label="Strategy B", color='green', marker='s')

plt.title('Conversion Rate Comparison Over Time')
plt.xlabel('Time')
plt.ylabel('Conversion Rate')
plt.legend()
plt.grid(True)
plt.show()



from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create data for 3D plot
time = np.array([1, 2, 3, 4, 5])
groups = np.array([1, 2])  # 1: Design A, 2: Design B
X, Y = np.meshgrid(time, groups)
Z = np.array([[0.25, 0.28, 0.30, 0.35, 0.38],  # Design A conversion rates
              [0.20, 0.22, 0.25, 0.29, 0.32]])  # Design B conversion rates

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plotting the surface
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title('Conversion Rates for Designs A and B')
ax.set_xlabel('Time')
ax.set_ylabel('Design')
ax.set_zlabel('Conversion Rate')

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm
import matplotlib.animation as animation

# Step 1: Define the data
# Group A: Original design
n_A = 80  # Number of users in Group A
mean_A = 50  # Mean purchase amount for Group A
std_A = 10  # Standard deviation for Group A

# Group B: New design
n_B = 20  # Number of users in Group B
mean_B = 55  # Mean purchase amount for Group B
std_B = 15  # Standard deviation for Group B

# Significance level
alpha = 0.05

# Step 2: Calculate the t-statistic and p-value
# Pooled standard deviation
pooled_std = np.sqrt(((n_A - 1) * std_A**2 + (n_B - 1) * std_B**2) / (n_A + n_B - 2))
# t-statistic
t_stat = (mean_B - mean_A) / (pooled_std * np.sqrt(1/n_A + 1/n_B))
# Degrees of freedom
df = n_A + n_B - 2
# p-value (one-tailed test)
p_value = 1 - t.cdf(t_stat, df)

# Step 3: Create the animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("A-B Testing: Hypothesis Testing Visualization", fontsize=16)
ax.set_xlabel("Test Statistic (t)", fontsize=14)
ax.set_ylabel("Probability Density", fontsize=14)
ax.set_xlim(-4, 4)
ax.set_ylim(0, 0.5)

# Plot the t-distribution
x = np.linspace(-4, 4, 1000)
y = t.pdf(x, df)
ax.plot(x, y, label="t-Distribution (df={})".format(df), color="blue")

# Add critical region
critical_value = t.ppf(1 - alpha, df)
ax.fill_between(x, 0, y, where=(x >= critical_value), color="red", alpha=0.5, label="Critical Region (α=0.05)")

# Add observed t-statistic
ax.axvline(t_stat, color="green", linestyle="--", label="Observed t-statistic = {:.2f}".format(t_stat))

# Add p-value annotation
p_value_text = "p-value = {:.3f}".format(p_value)
ax.text(2, 0.45, p_value_text, fontsize=12, color="green")

# Add legend
ax.legend(loc="upper right", fontsize=12)

# Animation function
def animate(frame):
    ax.set_title("Step 1: Define Groups and Collect Data", fontsize=16)
    ax.set_title("Step 2: Calculate t-Statistic and p-value", fontsize=16)
    ax.set_title("Step 3: Compare p-value to α (Decision Making)", fontsize=16)
    return ax

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=3, interval=2000, repeat=False)

# Save or display the animation
plt.show()
# ani.save("ab_testing_animation.gif", writer="pillow")  # Uncomment to save as GIF