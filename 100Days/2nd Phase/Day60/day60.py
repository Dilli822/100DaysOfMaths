import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Data Setup
ages = np.array([7, 7, 7, 8, 8, 9, 9, 9, 9, 10])
heights = np.array([45, 46, 46, 47, 47, 49, 49, 49, 50, 50])

# Unique values and counts
unique_ages, age_counts = np.unique(ages, return_counts=True)
unique_heights, height_counts = np.unique(heights, return_counts=True)

# Joint Count Table
joint_counts = np.zeros((len(unique_ages), len(unique_heights)))

# Fill the joint count table
for age, height in zip(ages, heights):
    row = np.where(unique_ages == age)[0][0]
    col = np.where(unique_heights == height)[0][0]
    joint_counts[row, col] += 1

# Normalize to get probabilities
joint_probs = joint_counts / len(ages)

# 2D Plot of the Joint Distribution
plt.figure(figsize=(8, 6))
plt.imshow(joint_probs, cmap='Blues', extent=[unique_heights.min() - 0.5, unique_heights.max() + 0.5, unique_ages.min() - 0.5, unique_ages.max() + 0.5], aspect='auto')
plt.colorbar(label='Probability')
plt.xticks(unique_heights)
plt.yticks(unique_ages)
plt.title('2D Heatmap of Joint Probability Distribution')
plt.xlabel('Height (inches)')
plt.ylabel('Age (years)')

for i in range(len(unique_ages)):
    for j in range(len(unique_heights)):
        prob = joint_probs[i, j]
        if prob > 0:
            plt.text(unique_heights[j], unique_ages[i], f'{prob:.2f}', ha='center', va='center', color='black')

plt.show()

# 3D Plot of the Joint Distribution
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(unique_heights, unique_ages)
Z = joint_probs

ax.bar3d(X.flatten(), Y.flatten(), np.zeros_like(Z.flatten()), 0.5, 0.5, Z.flatten(), shade=True, color='skyblue')
ax.set_xlabel('Height (inches)')
ax.set_ylabel('Age (years)')
ax.set_zlabel('Probability')
ax.set_title('3D Plot of Joint Probability Distribution')

plt.show()

# Animated Plot
fig, ax = plt.subplots(figsize=(8, 6))

cax = ax.imshow(joint_probs, cmap='Blues', extent=[unique_heights.min() - 0.5, unique_heights.max() + 0.5, unique_ages.min() - 0.5, unique_ages.max() + 0.5], aspect='auto')
plt.colorbar(cax, label='Probability')
plt.xticks(unique_heights)
plt.yticks(unique_ages)
plt.title('Animated Visualization of Joint Distribution')
plt.xlabel('Height (inches)')
plt.ylabel('Age (years)')

# Function to update animation
def update(frame):
    ax.clear()
    prob = joint_probs * (np.sin(frame / 10) + 1.5)
    cax = ax.imshow(prob, cmap='Blues', extent=[unique_heights.min() - 0.5, unique_heights.max() + 0.5, unique_ages.min() - 0.5, unique_ages.max() + 0.5], aspect='auto')
    ax.set_title(f'Animated Visualization (Frame {frame})')
    ax.set_xlabel('Height (inches)')
    ax.set_ylabel('Age (years)')
    for i in range(len(unique_ages)):
        for j in range(len(unique_heights)):
            if prob[i, j] > 0:
                ax.text(unique_heights[j], unique_ages[i], f'{prob[i, j]:.2f}', ha='center', va='center', color='black')

ani = FuncAnimation(fig, update, frames=100, interval=100)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Simulated dataset
np.random.seed(42)
n = 1000

# Simulating waiting time (X) and satisfaction rating (Y)
X = np.random.uniform(0, 10, n)
Y = 10 - X + np.random.normal(0, 1.5, n)  # Negative correlation with some noise
Y = np.clip(Y, 0, 10)  # Keeping Y within 0-10 range

# 2D Heatmap
plt.figure(figsize=(8, 6))
kde = sns.kdeplot(x=X, y=Y, fill=True, cmap="coolwarm", levels=30)
plt.colorbar(kde.collections[-1], label="Density")  # Fix: Use the last collection from kdeplot
plt.xlabel("Waiting Time (X) in minutes")
plt.ylabel("Customer Satisfaction Rating (Y)")
plt.title("Heatmap of Joint Distribution")
plt.show()


# 3D Surface Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Creating 2D histogram bins
hist, xedges, yedges = np.histogram2d(X, Y, bins=30, density=True)

# Constructing the grid for plotting
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

# Bar heights
dz = hist.ravel()

# Plot the 3D histogram
ax.bar3d(xpos, ypos, zpos, 0.3, 0.3, dz, shade=True, cmap="coolwarm")
ax.set_xlabel("Waiting Time (X)")
ax.set_ylabel("Customer Satisfaction Rating (Y)")
ax.set_zlabel("Density")
ax.set_title("3D Histogram of Joint Distribution")
plt.show()







import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Generate sample data for age and height (example)
np.random.seed(0)
ages = np.random.randint(5, 18, 100)  # Ages between 5 and 18
heights = np.random.normal(loc=50, scale=10, size=100)  # Heights centered around 50 with SD = 10

# 2D Plot: Scatter plot of Age vs Height with Marginal Distributions
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
ax[0].scatter(ages, heights, color='blue', alpha=0.7)
ax[0].set_title('Age vs Height')
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Height')

# Marginal Distribution (height)
height_bins = np.linspace(30, 80, 10)
ax[1].hist(heights, bins=height_bins, color='green', alpha=0.7, edgecolor='black')
ax[1].set_title('Marginal Distribution of Heights')
ax[1].set_xlabel('Height')
ax[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# 3D Plot: Age, Height, and Frequency
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot in 3D space (Age, Height, Frequency)
x = ages
y = heights
z = np.zeros_like(x)

# Plot points in 3D
ax.scatter(x, y, z, c='r', marker='o', alpha=0.6)

ax.set_title('3D Plot: Age vs Height')
ax.set_xlabel('Age')
ax.set_ylabel('Height')
ax.set_zlabel('Frequency')

plt.show()





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Generate synthetic age and height data
np.random.seed(42)
ages = np.random.randint(5, 15, 200)  # Ages between 5 and 15
heights = np.random.normal(50 + (ages - 5) * 2.5, 3, 200)  # Heights with some variability

# Plot Joint Distribution (2D)
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(ages, heights, c='skyblue', edgecolor='black', alpha=0.6)
ax.set_title("2D Scatter Plot: Joint Distribution (Age vs Height)", fontsize=14)
ax.set_xlabel("Age (Years)")
ax.set_ylabel("Height (Inches)")
plt.show()

# Plot Marginal Distributions (2D Histogram)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Marginal for Age
ax[0].hist(ages, bins=10, color='salmon', edgecolor='black')
ax[0].set_title("Marginal Distribution of Age")
ax[0].set_xlabel("Age (Years)")
ax[0].set_ylabel("Frequency")

# Marginal for Height
ax[1].hist(heights, bins=10, color='lightgreen', edgecolor='black')
ax[1].set_title("Marginal Distribution of Height")
ax[1].set_xlabel("Height (Inches)")
ax[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# 3D Plot for Joint Distribution
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(ages, heights, bins=(10, 10))

# Construct arrays for the anchor positions of the bars
xpos, ypos = np.meshgrid(xedges[:-1] + 0.5, yedges[:-1] + 0.5, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the bars
dx = dy = 0.8 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='teal', alpha=0.7)
ax.set_title("3D Visualization of Joint Distribution")
ax.set_xlabel("Age (Years)")
ax.set_ylabel("Height (Inches)")
ax.set_zlabel("Frequency")
plt.show()

# Animation for Conditional Distributions
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(5, 15)
ax.set_ylim(40, 80)
scatter = ax.scatter([], [], c='purple', edgecolor='black', alpha=0.6)
ax.set_title("Conditional Distribution Animation (Height for Specific Ages)", fontsize=14)
ax.set_xlabel("Age (Years)")
ax.set_ylabel("Height (Inches)")

def update(frame):
    ax.clear()
    ax.set_xlim(5, 15)
    ax.set_ylim(40, 80)
    selected_age = frame
    conditional_heights = heights[ages == selected_age]
    ax.scatter([selected_age] * len(conditional_heights), conditional_heights,
               c='purple', edgecolor='black', alpha=0.6)
    ax.set_title(f"Conditional Distribution for Age {selected_age}", fontsize=14)
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Height (Inches)")

ani = FuncAnimation(fig, update, frames=range(5, 15), repeat=False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate sample data for age and height (example)
np.random.seed(0)
ages = np.random.randint(5, 18, 100)  # Ages between 5 and 18
heights = np.random.normal(loc=50, scale=10, size=100)  # Heights centered around 50 with SD = 10

# Set up figure and axis for animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(30, 80)  # Height axis range
ax.set_ylim(0, 15)  # Frequency axis range
ax.set_xlabel('Height')
ax.set_ylabel('Frequency')
ax.set_title('Conditional Distribution of Height for Different Ages')

# Create a histogram container
hist_bins = np.linspace(30, 80, 10)
bar = ax.bar(hist_bins[:-1], np.zeros_like(hist_bins[:-1]), width=np.diff(hist_bins), color='green', alpha=0.7)

# Update function for animation
def update(frame):
    age = frame + 5  # Starting from age 5 to 17
    # Filter data for current age
    filtered_heights = heights[ages == age]
    # Calculate the histogram for current age
    hist_values, _ = np.histogram(filtered_heights, bins=hist_bins)
    
    # Update bar heights
    for rect, height in zip(bar, hist_values):
        rect.set_height(height)
    
    ax.set_title(f'Conditional Distribution of Height for Age {age}')

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(5, 18), interval=500)

# Show the animation
plt.show()
