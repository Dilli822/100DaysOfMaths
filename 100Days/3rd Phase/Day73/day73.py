import numpy as np
import matplotlib.pyplot as plt

# Simulate rolling two dice
np.random.seed(42)
num_rolls = 10000
dice1 = np.random.randint(1, 7, num_rolls)
dice2 = np.random.randint(1, 7, num_rolls)
sums = dice1 + dice2

# Conditional probability: P(Sum=7 | Dice1=4)
condition = dice1 == 4
conditional_sum = sums[condition]
prob = np.mean(conditional_sum == 7)

print(f"P(Sum=7 | Dice1=4) = {prob:.2f}")

# Visualization
plt.hist(sums, bins=range(2, 14), density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title("Probability Distribution of Dice Sums")
plt.xlabel("Sum of Two Dice")
plt.ylabel("Probability")
plt.show()


# Define parameters
prevalence = 0.01  # P(Disease)
sensitivity = 0.95  # P(Test+ | Disease)
specificity = 0.90  # P(Test- | No Disease)

# Calculate P(Disease | Test+)
def bayes_theorem(prevalence, sensitivity, specificity):
    p_test_pos_given_disease = sensitivity
    p_test_pos_given_no_disease = 1 - specificity
    p_disease = prevalence
    p_no_disease = 1 - prevalence

    # P(Test+)
    p_test_pos = (p_test_pos_given_disease * p_disease) + (p_test_pos_given_no_disease * p_no_disease)

    # P(Disease | Test+)
    p_disease_given_test_pos = (p_test_pos_given_disease * p_disease) / p_test_pos
    return p_disease_given_test_pos

prob = bayes_theorem(prevalence, sensitivity, specificity)
print(f"P(Disease | Test+) = {prob:.4f}")


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Example data
y_true = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1]

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix")
plt.show()


def likelihood_ratios(sensitivity, specificity):
    LR_plus = sensitivity / (1 - specificity)  # LR for a positive test
    LR_minus = (1 - sensitivity) / specificity  # LR for a negative test
    return LR_plus, LR_minus

LR_plus, LR_minus = likelihood_ratios(sensitivity, specificity)
print(f"Likelihood Ratio (Positive Test): {LR_plus:.2f}")
print(f"Likelihood Ratio (Negative Test): {LR_minus:.2f}")


from mpl_toolkits.mplot3d import Axes3D

# Create a 3D histogram
dice1_vals = np.arange(1, 7)
dice2_vals = np.arange(1, 7)
counts = np.zeros((6, 6))

for i in range(num_rolls):
    counts[dice1[i]-1, dice2[i]-1] += 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(dice1_vals, dice2_vals)
ax.bar3d(x.ravel(), y.ravel(), np.zeros(36), 1, 1, counts.ravel(), shade=True)
ax.set_xlabel("Dice 1")
ax.set_ylabel("Dice 2")
ax.set_zlabel("Frequency")
plt.title("3D Dice Roll Frequencies")
plt.show()


from matplotlib.animation import FuncAnimation

# Initialize the plot
fig, ax = plt.subplots()
x = np.arange(2, 13)
bars = ax.bar(x, np.zeros(11), color='blue')
ax.set_ylim(0, 0.2)
ax.set_xlabel("Sum of Two Dice")
ax.set_ylabel("Probability")

# Update function for animation
def update(frame):
    dice1 = np.random.randint(1, 7, frame)
    dice2 = np.random.randint(1, 7, frame)
    sums = dice1 + dice2
    hist, _ = np.histogram(sums, bins=range(2, 14), density=True)
    for bar, h in zip(bars, hist):
        bar.set_height(h)
    ax.set_title(f"Rolls: {frame}")
    return bars

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(1, 1000, 10), interval=100, repeat=False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)

# Create the visualization class
class DiagnosticTestViz:
    def __init__(self):
        # Set up the figure with two subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        self.fig.suptitle('Diagnostic Testing Visualization', fontsize=16)
        
        # Distribution parameters
        self.mu_healthy = 5
        self.mu_sick = 8
        self.sigma = 1
        self.threshold = 6.5
        
        # Generate data points
        self.x = np.linspace(0, 13, 200)
        self.healthy_dist = norm.pdf(self.x, self.mu_healthy, self.sigma)
        self.sick_dist = norm.pdf(self.x, self.mu_sick, self.sigma)
        
        # Animation parameters
        self.current_frame = 0
        self.n_points = 100
        self.points_healthy = np.random.normal(self.mu_healthy, self.sigma, self.n_points)
        self.points_sick = np.random.normal(self.mu_sick, self.sigma, self.n_points)
        self.results = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
        
    def setup_static_plot(self):
        # Plot distributions
        self.ax1.plot(self.x, self.healthy_dist, 'b-', label='Healthy Population')
        self.ax1.plot(self.x, self.sick_dist, 'r-', label='Sick Population')
        self.ax1.axvline(self.threshold, color='k', linestyle='--', label='Threshold')
        
        # Fill areas for false positives and negatives
        self.ax1.fill_between(self.x[self.x >= self.threshold], 
                            self.healthy_dist[self.x >= self.threshold],
                            alpha=0.3, color='red', label='False Positives')
        self.ax1.fill_between(self.x[self.x <= self.threshold],
                            self.sick_dist[self.x <= self.threshold],
                            alpha=0.3, color='blue', label='False Negatives')
        
        # Add labels and legend
        self.ax1.set_title('Population Distributions and Error Regions')
        self.ax1.set_xlabel('Test Value')
        self.ax1.set_ylabel('Probability Density')
        self.ax1.legend()
        
        # Setup scatter plot
        self.ax2.set_xlim(0, 13)
        self.ax2.set_ylim(-1, 1)
        self.ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax2.axvline(x=self.threshold, color='k', linestyle='--', alpha=0.3)
        self.ax2.set_title('Testing Animation')
        self.ax2.set_xlabel('Test Value')
        self.text = self.ax2.text(0.02, 0.95, '', transform=self.ax2.transAxes)
        
    def animate(self, frame):
        if frame < self.n_points:
            # Test healthy individuals
            point = self.points_healthy[frame]
            y = -0.2
            if point >= self.threshold:
                color = 'red'  # False Positive
                self.results['FP'] += 1
            else:
                color = 'blue'  # True Negative
                self.results['TN'] += 1
        else:
            # Test sick individuals
            point = self.points_sick[frame - self.n_points]
            y = 0.2
            if point >= self.threshold:
                color = 'red'  # True Positive
                self.results['TP'] += 1
            else:
                color = 'blue'  # False Negative
                self.results['FN'] += 1
                
        self.ax2.scatter(point, y, c=color, alpha=0.6)
        
        # Update results text
        total = sum(self.results.values())
        if total > 0:
            text = f'Results (n={total}):\n'
            text += f'True Positives: {self.results["TP"]}\n'
            text += f'False Positives: {self.results["FP"]}\n'
            text += f'True Negatives: {self.results["TN"]}\n'
            text += f'False Negatives: {self.results["FN"]}'
            self.text.set_text(text)
        
        return self.ax2.collections + [self.text]

    def create_animation(self):
        self.setup_static_plot()
        anim = FuncAnimation(self.fig, self.animate, frames=2*self.n_points,
                           interval=50, blit=True)
        plt.tight_layout()
        return anim

# Create and display the visualization
viz = DiagnosticTestViz()
animation = viz.create_animation()
plt.show()