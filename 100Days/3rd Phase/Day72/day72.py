import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================
# 1. Probability Distribution of a Single Die (2D)
# ==============================================
plt.figure(figsize=(12, 8))

# Outcomes of a single die
outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6  # Equal probability for each outcome

# Plot
plt.subplot(2, 2, 1)
plt.bar(outcomes, probabilities, color='skyblue')
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.title('Probability Distribution of a Single Die')
plt.ylim(0, 0.2)  # Set y-axis limit

# ==============================================
# 2. Probability Distribution of the Sum of Two Dice (2D)
# ==============================================

# Possible sums of two dice
sums = range(2, 13)
# Probabilities for each sum
probabilities_sum = [1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36]

# Plot
plt.subplot(2, 2, 2)
plt.bar(sums, probabilities_sum, color='lightgreen', tick_label=sums)
plt.xlabel('Sum of Two Dice')
plt.ylabel('Probability')
plt.title('Probability Distribution of the Sum of Two Dice')
plt.ylim(0, 0.2)  # Set y-axis limit

# ==============================================
# 3. Joint Probability Distribution of Two Dice (3D)
# ==============================================

# Create a 6x6 grid for two dice
x = np.arange(1, 7)  # Die 1 outcomes
y = np.arange(1, 7)  # Die 2 outcomes
x, y = np.meshgrid(x, y)
z = np.ones_like(x) / 36  # Equal probability for each combination

# 3D Plot
plt.subplot(2, 2, 3, projection='3d')
ax = plt.gca()
ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z).ravel(), 1, 1, z.ravel(), shade=True, color='orange')
ax.set_xlabel('Die 1')
ax.set_ylabel('Die 2')
ax.set_zlabel('Probability')
ax.set_title('Joint Probability Distribution of Two Dice')

# ==============================================
# 4. Bayes' Theorem Visualization (Conditional Probability)
# ==============================================

# Given data
P_D = 0.01  # Prevalence of the disease
P_Pos_given_D = 0.99  # Probability of testing positive given disease
P_Pos_given_no_D = 0.01  # Probability of testing positive given no disease

# Calculate P(Positive)
P_Pos = P_Pos_given_D * P_D + P_Pos_given_no_D * (1 - P_D)

# Calculate P(Disease | Positive) using Bayes' Theorem
P_D_given_Pos = (P_Pos_given_D * P_D) / P_Pos

# Plot
plt.subplot(2, 2, 4)
labels = ['P(Disease | Positive)', 'P(No Disease | Positive)']
sizes = [P_D_given_Pos, 1 - P_D_given_Pos]
colors = ['lightcoral', 'lightskyblue']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title("Bayes' Theorem: Disease Testing")

# Adjust layout and display
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to draw a single die face
def draw_die(ax, value, x, y, size=1):
    """Draw a die face with the given value at position (x, y)."""
    ax.clear()
    ax.set_xlim(0, 7 * size)
    ax.set_ylim(0, 2 * size)
    ax.set_aspect('equal')
    ax.axis('off')  # Hide axes

    # Draw the die square
    die_face = plt.Rectangle((x, y), size, size, fc='white', ec='black', lw=2)
    ax.add_patch(die_face)

    # Draw dots based on the die value
    dot_positions = {
        1: [(x + size / 2, y + size / 2)],
        2: [(x + size / 4, y + 3 * size / 4), (x + 3 * size / 4, y + size / 4)],
        3: [(x + size / 4, y + 3 * size / 4), (x + size / 2, y + size / 2), (x + 3 * size / 4, y + size / 4)],
        4: [(x + size / 4, y + 3 * size / 4), (x + 3 * size / 4, y + 3 * size / 4),
            (x + size / 4, y + size / 4), (x + 3 * size / 4, y + size / 4)],
        5: [(x + size / 4, y + 3 * size / 4), (x + 3 * size / 4, y + 3 * size / 4),
            (x + size / 2, y + size / 2),
            (x + size / 4, y + size / 4), (x + 3 * size / 4, y + size / 4)],
        6: [(x + size / 4, y + 3 * size / 4), (x + 3 * size / 4, y + 3 * size / 4),
            (x + size / 4, y + size / 2), (x + 3 * size / 4, y + size / 2),
            (x + size / 4, y + size / 4), (x + 3 * size / 4, y + size / 4)]
    }

    for pos in dot_positions[value]:
        dot = plt.Circle(pos, size / 12, fc='black')
        ax.add_patch(dot)

# Function to simulate rolling two dice
def roll_dice():
    """Simulate rolling two dice and return their values."""
    return np.random.randint(1, 7), np.random.randint(1, 7)

# Animation function
def animate(frame):
    """Update the animation frame."""
    die1_value, die2_value = roll_dice()
    draw_die(ax1, die1_value, x=1, y=0.5, size=1)
    draw_die(ax2, die2_value, x=4, y=0.5, size=1)
    plt.suptitle(f"Rolling Dice: {die1_value} and {die2_value}", fontsize=14)

# Set up the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.axis('off')
ax2.axis('off')

# Create the animation
ani = FuncAnimation(fig, animate, frames=range(20), interval=500, repeat=True)

# Display the animation
plt.show()