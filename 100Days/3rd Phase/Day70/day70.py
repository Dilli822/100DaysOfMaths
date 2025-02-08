import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Number of cards in a standard deck
total_cards = 52
hearts = 13
spades = 13
diamonds = 13
clubs = 13

# Basic probabilities for suits
P_heart = hearts / total_cards
P_spade = spades / total_cards
P_diamond = diamonds / total_cards
P_club = clubs / total_cards

# Conditional probability (Example: P(Heart | Red card)) 
# Probability of drawing a heart given the card is red (red cards = hearts + diamonds)
red_cards = hearts + diamonds
P_heart_given_red = hearts / red_cards

# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(7, 5))

# Initialize plot elements
bar_width = 0.3
x_pos = np.arange(4)

bars = ax.bar(x_pos, [P_heart, P_spade, P_diamond, P_club], width=bar_width, color='lightblue')
ax.set_ylim(0, 1)
ax.set_xticks(x_pos)
ax.set_xticklabels(['Hearts', 'Spades', 'Diamonds', 'Clubs'])
ax.set_ylabel('Probability')
ax.set_title('Card Probability Distribution')

# Text annotations for conditional probability
text = ax.text(0.5, 0.9, f'P(Heart | Red) = {P_heart_given_red:.2f}', transform=ax.transAxes,
               fontsize=12, color='red', ha='center')

# Animation function
def update(frame):
    # Varying the height of the bars (e.g., simulating change over time or with conditional events)
    new_hearts_prob = np.clip(np.random.uniform(0, P_heart), 0, 1)
    new_spades_prob = np.clip(np.random.uniform(0, P_spade), 0, 1)
    new_diamonds_prob = np.clip(np.random.uniform(0, P_diamond), 0, 1)
    new_clubs_prob = np.clip(np.random.uniform(0, P_club), 0, 1)
    
    # Update bar heights
    bars[0].set_height(new_hearts_prob)
    bars[1].set_height(new_spades_prob)
    bars[2].set_height(new_diamonds_prob)
    bars[3].set_height(new_clubs_prob)
    
    # Update conditional probability text
    P_heart_given_red = new_hearts_prob / (new_hearts_prob + new_diamonds_prob)
    text.set_text(f'P(Heart | Red) = {P_heart_given_red:.2f}')
    
    return bars, text

# Create animation
ani = FuncAnimation(fig, update, frames=100, interval=500, blit=False)

plt.tight_layout()
plt.show()



# Red cards (hearts + diamonds)
red_cards = hearts + diamonds

# Conditional probability P(Heart | Red)
P_heart_given_red = hearts / red_cards

# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(7, 5))

# Initialize plot elements
bars = ax.bar([0], [P_heart_given_red], color='lightgreen')
ax.set_ylim(0, 1)
ax.set_ylabel('Conditional Probability')
ax.set_title('Conditional Probability P(Heart | Red)')

# Animation function
def update(frame):
    # Update conditional probability (simulate variation)
    new_P_heart_given_red = np.clip(np.random.uniform(0, P_heart_given_red), 0, 1)
    bars[0].set_height(new_P_heart_given_red)
    
    return bars

# Create animation
ani = FuncAnimation(fig, update, frames=100, interval=200, blit=False)

plt.tight_layout()
plt.show()


import scipy.stats as stats

# Define sample size and population parameters
n = 30  # Sample size
population_mean = 50
population_std = 10

# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(7, 5))

# Initialize plot elements
sample_means = np.random.normal(loc=population_mean, scale=population_std / np.sqrt(n), size=1000)

# Calculate the confidence interval
conf_int = stats.norm.interval(0.95, loc=population_mean, scale=population_std / np.sqrt(n))

# Plot histogram for sample means and confidence interval
ax.hist(sample_means, bins=30, color='lightblue', edgecolor='black')
ax.axvline(conf_int[0], color='red', linestyle='--', label=f'Confidence Interval Lower: {conf_int[0]:.2f}')
ax.axvline(conf_int[1], color='red', linestyle='--', label=f'Confidence Interval Upper: {conf_int[1]:.2f}')
ax.legend()
ax.set_title('Confidence Interval for Sample Mean')
ax.set_xlabel('Sample Mean')
ax.set_ylabel('Frequency')

# Animation function (showing change in confidence interval over iterations)
def update(frame):
    # Sample new data points
    new_sample_means = np.random.normal(loc=population_mean, scale=population_std / np.sqrt(n), size=1000)
    
    # Update the histogram
    ax.clear()
    ax.hist(new_sample_means, bins=30, color='lightblue', edgecolor='black')
    
    # Recalculate confidence interval
    conf_int = stats.norm.interval(0.95, loc=population_mean, scale=population_std / np.sqrt(n))
    ax.axvline(conf_int[0], color='red', linestyle='--', label=f'Confidence Interval Lower: {conf_int[0]:.2f}')
    ax.axvline(conf_int[1], color='red', linestyle='--', label=f'Confidence Interval Upper: {conf_int[1]:.2f}')
    ax.legend()
    
    return ax.patches, ax.lines

# Create animation
ani = FuncAnimation(fig, update, frames=100, interval=500, blit=False)

plt.tight_layout()
plt.show()


# Define probabilities
P_A = 0.5  # P(A)
P_B = 0.4  # P(B)
P_B_given_A = 0.7  # P(B | A)

# Bayes' Theorem: P(A | B) = (P(B | A) * P(A)) / P(B)
P_A_given_B = (P_B_given_A * P_A) / P_B

# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(7, 5))

# Initialize plot elements
bars = ax.bar([0, 1], [P_A, P_B], color=['lightblue', 'lightgreen'], label=["P(A)", "P(B)"])
ax.set_ylim(0, 1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["P(A)", "P(B)"])
ax.set_ylabel('Probability')
ax.set_title('Bayes Theorem Visualization')

# Animation function
def update(frame):
    # Simulate changing probabilities
    new_P_A = np.clip(np.random.uniform(0, 1), 0, 1)
    new_P_B = np.clip(np.random.uniform(0, 1), 0, 1)
    new_P_B_given_A = np.clip(np.random.uniform(0, 1), 0, 1)
    
    # Recalculate Bayes' Theorem
    new_P_A_given_B = (new_P_B_given_A * new_P_A) / new_P_B
    
    # Update bars
    bars[0].set_height(new_P_A)
    bars[1].set_height(new_P_B)
    
    ax.set_title(f'Bayes Theorem: P(A | B) = {new_P_A_given_B:.2f}')
    
    return bars

# Create animation
ani = FuncAnimation(fig, update, frames=100, interval=500, blit=False)

plt.tight_layout()
plt.show()
