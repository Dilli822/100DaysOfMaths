import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Set up common parameters
x = np.linspace(0, 20, 500)  # Range of W values (noise power)
degrees_of_freedom = [1, 2, 5, 10, 15, 20, 25, 30]  # Different degrees of freedom for Chi-squared distribution

# 1. 2D Plot of Chi-squared PDFs
plt.figure(figsize=(10, 6))
for k in degrees_of_freedom:
    pdf = chi2.pdf(x, df=k)
    plt.plot(x, pdf, label=f"k = {k}")

plt.title("Chi-squared Distribution PDFs")
plt.xlabel("Noise Power (W)")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

# 2. 3D Surface Plot of Chi-squared PDFs
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
k_values = np.linspace(1, 10, 10)
X, Y = np.meshgrid(x, k_values)
Z = np.array([chi2.pdf(X[i], df=Y[i]) for i in range(len(k_values))])

for i, k in enumerate(k_values):
    ax.plot(x, [k]*len(x), Z[i], label=f"k = {int(k)}")

ax.set_title("Chi-squared Distribution PDFs (3D)")
ax.set_xlabel("Noise Power (W)")
ax.set_ylabel("Degrees of Freedom (k)")
ax.set_zlabel("Probability Density")
plt.show()

# 3. Animated Plot of Chi-squared Distribution
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, 20)
ax.set_ylim(0, 0.5)
ax.set_title("Chi-squared Distribution Animation")
ax.set_xlabel("Noise Power (W)")
ax.set_ylabel("Probability Density")
ax.grid()

def init():
    line.set_data([], [])
    return line,

def update(frame):
    k = frame + 1
    pdf = chi2.pdf(x, df=k)
    line.set_data(x, pdf)
    ax.set_title(f"Chi-squared Distribution (k = {k})")
    return line,

ani = FuncAnimation(fig, update, frames=20, init_func=init, blit=True, interval=300)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 2D Plot for Sampling Discrete Distribution
def plot_discrete_distribution():
    colors = ['Green', 'Blue', 'Orange']
    probabilities = [0.3, 0.5, 0.2]
    
    fig, ax = plt.subplots()
    ax.bar(colors, probabilities, color=['green', 'blue', 'orange'])
    ax.set_title('Discrete Distribution')
    ax.set_ylabel('Probability')
    plt.show()

# CDF for Discrete Distribution
def plot_discrete_cdf():
    probabilities = [0.3, 0.5, 0.2]
    cumulative = np.cumsum(probabilities)
    colors = ['Green', 'Blue', 'Orange']

    fig, ax = plt.subplots()
    ax.step([0, 1, 2, 3], [0] + list(cumulative), where='post', color='red', label='CDF')
    ax.set_xticks(range(3))
    ax.set_xticklabels(colors)
    ax.set_title('CDF for Discrete Distribution')
    ax.set_ylabel('Cumulative Probability')
    plt.legend()
    plt.show()

# Animation for Continuous Distribution Sampling
def animate_continuous_sampling():
    x = np.linspace(-3, 3, 1000)
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)  # Standard Gaussian PDF
    cdf = np.cumsum(pdf) / np.sum(pdf)  # Approximate CDF using discrete values

    fig, ax = plt.subplots()
    ax.plot(x, pdf, label='PDF', color='blue')
    ax.plot(x, cdf, label='CDF', color='red')
    ax.set_title('Sampling from Continuous Distribution')
    ax.legend()

    sampled_points, = ax.plot([], [], 'go', label='Sampled Points')
    uniform_points = []
    samples = []

    def update(frame):
        uniform_sample = np.random.rand()
        idx = np.searchsorted(cdf, uniform_sample)
        sample = x[idx]

        uniform_points.append(uniform_sample)
        samples.append(sample)

        sampled_points.set_data(samples, [0] * len(samples))
        return sampled_points,

    ani = FuncAnimation(fig, update, frames=20, interval=500, blit=True, repeat=False)
    plt.show()

# 3D Plot for Sampling from Continuous Distribution
def plot_3d_continuous_distribution():
    from mpl_toolkits.mplot3d import Axes3D

    x = np.linspace(-3, 3, 1000)
    y = np.linspace(0, 1, 1000)
    X, Y = np.meshgrid(x, y)
    Z = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * X**2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('3D Visualization of Gaussian Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('CDF Sample')
    ax.set_zlabel('PDF')
    plt.show()

# Execute Plots
plot_discrete_distribution()
plot_discrete_cdf()
animate_continuous_sampling()
plot_3d_continuous_distribution()




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 2D Plot for Discrete Distribution
def plot_discrete_distribution():
    categories = ['Green', 'Blue', 'Orange']
    probabilities = [0.3, 0.5, 0.2]

    fig, ax = plt.subplots()
    bars = ax.bar(categories, probabilities, color=['green', 'blue', 'orange'])
    ax.set_title('Discrete Distribution')
    ax.set_ylabel('Probability')
    ax.set_xlabel('Categories')

    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{prob:.2f}', ha='center', va='bottom')

    plt.show()

# CDF for Discrete Distribution
def plot_discrete_cdf():
    probabilities = [0.3, 0.5, 0.2]
    cumulative = np.cumsum(probabilities)
    categories = ['Green', 'Blue', 'Orange']

    fig, ax = plt.subplots()
    ax.step(range(len(cumulative) + 1), [0] + list(cumulative), where='post', color='red', label='CDF')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_title('Discrete CDF')
    ax.set_ylabel('Cumulative Probability')
    ax.set_xlabel('Categories')
    ax.legend()

    for i, cum_prob in enumerate(cumulative):
        ax.text(i + 0.5, cum_prob - 0.05, f'{cum_prob:.2f}', ha='center')

    plt.show()

# Animation for Continuous Distribution Sampling
def animate_continuous_sampling():
    x = np.linspace(-3, 3, 1000)
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    cdf = np.cumsum(pdf) / np.sum(pdf)

    fig, ax = plt.subplots()
    ax.plot(x, pdf, label='PDF', color='blue')
    ax.plot(x, cdf, label='CDF', color='red')
    ax.set_title('Sampling from Continuous Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('Density/Probability')
    ax.legend()

    sampled_points, = ax.plot([], [], 'go', label='Sampled Points')
    samples = []

    def update(frame):
        uniform_sample = np.random.rand()
        idx = np.searchsorted(cdf, uniform_sample)
        sample = x[idx]
        samples.append(sample)
        sampled_points.set_data(samples, [0] * len(samples))
        return sampled_points,

    ani = FuncAnimation(fig, update, frames=20, interval=500, blit=True, repeat=False)
    plt.show()

# 3D Plot for Continuous Distribution
def plot_3d_continuous_distribution():
    x = np.linspace(-3, 3, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * X**2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('3D Gaussian Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('CDF Sample')
    ax.set_zlabel('PDF')

    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label='Density')
    plt.show()

# Execute Plots
plot_discrete_distribution()
plot_discrete_cdf()
animate_continuous_sampling()
plot_3d_continuous_distribution()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Expected Value for Discrete Distribution

def plot_expected_value_discrete():
    outcomes = np.array([1, 2, 3, 4, 5, 6])
    probabilities = np.full(6, 1/6)  # Fair dice probabilities
    expected_value = np.sum(outcomes * probabilities)

    fig, ax = plt.subplots()
    bars = ax.bar(outcomes, probabilities, color='skyblue', alpha=0.7, edgecolor='black')
    ax.axhline(expected_value / 6, color='red', linestyle='--', label=f'Expected Value = {expected_value:.2f}')

    ax.set_title('Expected Value of Dice Rolls')
    ax.set_xlabel('Outcomes')
    ax.set_ylabel('Probability')
    ax.legend()

    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{prob:.2f}', ha='center', va='bottom')

    plt.show()

# Animation for Squared Values of Dice Rolls

def animate_squared_dice_rolls():
    outcomes = np.array([1, 2, 3, 4, 5, 6])
    probabilities = np.full(6, 1/6)  # Fair dice probabilities
    squared_outcomes = outcomes**2
    expected_value = np.sum(squared_outcomes * probabilities)

    fig, ax = plt.subplots()
    bars = ax.bar(outcomes, squared_outcomes, color='lightgreen', alpha=0.7, edgecolor='black')
    ax.axhline(expected_value, color='red', linestyle='--', label=f'Expected Value = {expected_value:.2f}')

    ax.set_title('Expected Value of Squared Dice Rolls')
    ax.set_xlabel('Outcomes')
    ax.set_ylabel('Squared Values')
    ax.legend()

    def update(frame):
        sampled_outcomes = np.random.choice(outcomes, size=frame, p=probabilities)
        cumulative_avg = np.mean(sampled_outcomes**2)
        ax.axhline(cumulative_avg, color='blue', linestyle='--', alpha=0.5, label=f'Frame {frame}')
        return bars

    ani = FuncAnimation(fig, update, frames=20, interval=500, repeat=False)
    plt.show()

# Linear Transformation of Dice Rolls

def plot_linear_transformation():
    outcomes = np.array([1, 2, 3, 4, 5, 6])
    probabilities = np.full(6, 1/6)  # Fair dice probabilities
    transformation = 2 * outcomes - 5
    expected_value = np.sum(transformation * probabilities)

    fig, ax = plt.subplots()
    bars = ax.bar(outcomes, transformation, color='orange', alpha=0.7, edgecolor='black')
    ax.axhline(expected_value, color='purple', linestyle='--', label=f'Expected Value = {expected_value:.2f}')

    ax.set_title('Linear Transformation of Dice Rolls')
    ax.set_xlabel('Outcomes')
    ax.set_ylabel('Transformed Values (2x - 5)')
    ax.legend()

    for bar, value in zip(bars, transformation):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f'{value:.2f}', ha='center', va='bottom')

    plt.show()

# Execute Visualizations
plot_expected_value_discrete()
animate_squared_dice_rolls()
plot_linear_transformation()



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, binom
from matplotlib.animation import FuncAnimation

# Set seaborn style
sns.set(style="whitegrid")

# ==================================================
# 1. Symmetric Distribution (Normal Distribution)
# ==================================================
mu, sigma = 0, 1  # Mean and standard deviation
x = np.linspace(-5, 5, 1000)
y = norm.pdf(x, mu, sigma)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Normal Distribution")
plt.axvline(mu, color='r', linestyle='--', label="Mean = Median = Mode")
plt.title("Symmetric Distribution (Normal Distribution)")
plt.xlabel("X")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

# ==================================================
# 2. Skewed Distribution (Salaries with an Outlier)
# ==================================================
salaries = np.append(np.random.normal(22000, 5000, 100), 250000)  # 100 normal salaries + 1 outlier

plt.figure(figsize=(10, 6))
sns.histplot(salaries, bins=30, kde=True, label="Salary Distribution")
plt.axvline(np.mean(salaries), color='r', linestyle='--', label="Mean")
plt.axvline(np.median(salaries), color='g', linestyle='--', label="Median")
plt.title("Skewed Distribution (Salaries with an Outlier)")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()

# ==================================================
# 3. Binomial Distribution (Symmetric and Asymmetric Cases)
# ==================================================
n = 5  # Number of trials
p_symmetric = 0.5  # Symmetric case
p_asymmetric = 0.3  # Asymmetric case

x = np.arange(0, n+1)
y_symmetric = binom.pmf(x, n, p_symmetric)
y_asymmetric = binom.pmf(x, n, p_asymmetric)

# Symmetric case
plt.figure(figsize=(10, 6))
plt.bar(x, y_symmetric, label="Symmetric Binomial (p=0.5)")
plt.axvline(np.mean(x), color='r', linestyle='--', label="Mean = Median")
plt.title("Symmetric Binomial Distribution")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()

# Asymmetric case
plt.figure(figsize=(10, 6))
plt.bar(x, y_asymmetric, label="Asymmetric Binomial (p=0.3)")
plt.axvline(np.mean(x * y_asymmetric), color='r', linestyle='--', label="Mean")
plt.axvline(np.median(x), color='g', linestyle='--', label="Median")
plt.title("Asymmetric Binomial Distribution")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()

# ==================================================
# 4. 3D Plot (Normal Distribution)
# ==================================================
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = norm.pdf(np.sqrt(X**2 + Y**2), 0, 1)

# Handle invalid values in Z
Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title("3D Normal Distribution")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Probability Density")
plt.show()

# ==================================================
# 5. Animation (Skewed Distribution)
# ==================================================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 300000, 1000)
y = norm.pdf(x, 22000, 5000)

def animate(i):
    ax.clear()
    ax.plot(x, y, label="Salary Distribution")
    ax.axvline(22000 + i * 1000, color='r', linestyle='--', label="Mean")
    ax.axvline(np.median(salaries), color='g', linestyle='--', label="Median")
    ax.set_title(f"Skewed Distribution (Mean Increasing)")
    ax.set_xlabel("Salary")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid()

ani = FuncAnimation(fig, animate, frames=50, interval=200)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 2D Plot: Normal Distribution with Standard Deviation Intervals
def plot_normal_distribution():
    mean = 0
    std_dev = 1
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 500)
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

    fig, ax = plt.subplots()
    ax.plot(x, y, label='Normal Distribution', color='blue')

    # Mark standard deviation intervals
    for i in range(1, 4):
        ax.axvline(mean + i * std_dev, color='orange', linestyle='--', label=f'+{i} SD')
        ax.axvline(mean - i * std_dev, color='green', linestyle='--', label=f'-{i} SD')

    ax.fill_between(x, y, where=((x >= mean - std_dev) & (x <= mean + std_dev)), color='blue', alpha=0.2, label='68.2% Area')
    ax.fill_between(x, y, where=((x >= mean - 2 * std_dev) & (x <= mean + 2 * std_dev)), color='purple', alpha=0.1, label='95% Area')
    ax.fill_between(x, y, where=((x >= mean - 3 * std_dev) & (x <= mean + 3 * std_dev)), color='red', alpha=0.05, label='99.7% Area')

    ax.set_title('Normal Distribution and Standard Deviation Intervals')
    ax.set_xlabel('X')
    ax.set_ylabel('Density')
    ax.legend()
    plt.show()

# 3D Plot: Normal Distribution Surface
def plot_normal_distribution_3d():
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]  # Independent variables with variance 1
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = (1 / (2 * np.pi)) * np.exp(-0.5 * (X**2 + Y**2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    ax.set_title('3D Normal Distribution')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Density')
    plt.show()

# Animation: Standard Deviation Demonstration
def animate_standard_deviation():
    mean = 0
    x = np.linspace(mean - 4, mean + 4, 500)  # Fixed range
    fig, ax = plt.subplots()
    line, = ax.plot(x, np.zeros_like(x), color='blue')
    fill = [ax.fill_between(x, np.zeros_like(x), color='blue', alpha=0.2)]
    ax.set_xlim(mean - 4, mean + 4)
    ax.set_ylim(0, 0.5)  # Adjust to expected density values
    ax.set_title('Dynamic Standard Deviation Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Density')

    def update(frame):
        std_dev = max(frame / 10, 0.1)  # Ensure std_dev stays above 0.1
        y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        line.set_ydata(y)

        # Remove previous fill_between
        if fill[0]:
            fill[0].remove()

        # Add new shaded area
        fill[0] = ax.fill_between(x, y, color='blue', alpha=0.2)
        return line,

    ani = FuncAnimation(fig, update, frames=50, interval=200, blit=False, repeat=False)
    plt.show()

# Execute Visualizations
plot_normal_distribution()
plot_normal_distribution_3d()
animate_standard_deviation()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

# Set seaborn style
sns.set(style="whitegrid")

# ==================================================
# 1. Expected Value of the Coin and Dice Game
# ==================================================
# Expected values
E_coin = 0.5  # Expected value of the coin game
E_dice = 3.5  # Expected value of the dice game
E_total = E_coin + E_dice  # Total expected value

# Plot
plt.figure(figsize=(10, 6))
plt.bar(["Coin Game", "Dice Game", "Total"], [E_coin, E_dice, E_total], color=["blue", "green", "red"])
plt.title("Expected Values of the Coin and Dice Game")
plt.ylabel("Expected Value ($)")
plt.grid(axis="y")
plt.show()

# ==================================================
# 2. Expected Number of Correct Assignments (Matching Problem)
# ==================================================
# Number of people
n = 8_000_000_000  # 8 billion people
E_matches = 1  # Expected number of correct assignments

# Plot
plt.figure(figsize=(10, 6))
plt.bar(["Expected Matches"], [E_matches], color="purple")
plt.title(f"Expected Number of Correct Assignments (n = {n})")
plt.ylabel("Expected Matches")
plt.grid(axis="y")
plt.show()

# ==================================================
# 3. 3D Plot (Matching Problem for Small n)
# ==================================================
# Number of people
n_small = 3  # Small number for visualization
E_matches_small = 1  # Expected number of correct assignments

# Generate data for 3D plot
x = np.arange(1, n_small+1)
y = np.arange(1, n_small+1)
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, 1/n_small)  # Probability of correct assignment for each person

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.set_title("3D Plot: Probability of Correct Assignments")
ax.set_xlabel("Person")
ax.set_ylabel("Assignment")
ax.set_zlabel("Probability")
plt.show()

# ==================================================
# 4. Animation (Matching Problem for Small n)
# ==================================================
# Number of people
n_animation = 3  # Small number for visualization
E_matches_animation = 1  # Expected number of correct assignments

# Generate data for animation
x_animation = np.arange(1, n_animation+1)
y_animation = np.full(n_animation, 1/n_animation)  # Probability of correct assignment for each person

# Animation function
fig, ax = plt.subplots(figsize=(10, 6))
def animate(i):
    ax.clear()
    ax.bar(x_animation[:i+1], y_animation[:i+1], color="orange")
    ax.set_title(f"Expected Matches for {i+1} People")
    ax.set_xlabel("Person")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.grid(axis="y")

ani = FuncAnimation(fig, animate, frames=n_animation, interval=1000)
plt.show()