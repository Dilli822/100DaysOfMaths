import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math

# Parameters for the binomial distribution
n = 10  # number of trials
p = 1/6  # probability of success (rolling a 1)

# Function to calculate the binomial coefficient
def binomial_coefficient(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

# Binomial PMF
def binomial_pmf(k, n, p):
    return binomial_coefficient(n, k) * (p**k) * ((1-p)**(n-k))

# Possible outcomes (0 to n)
x = np.arange(0, n+1)
y = [binomial_pmf(k, n, p) for k in x]

# 2D Visualization
plt.figure(figsize=(8, 6))
plt.bar(x, y, color='skyblue', edgecolor='black')
plt.title("2D Visualization of Binomial Distribution")
plt.xlabel("Number of successes (k)")
plt.ylabel("Probability (P)")
plt.show()

# 3D Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Creating the 3D bar plot
_x = np.arange(len(x))
_y = np.zeros_like(_x)
_z = np.zeros_like(_x)
dx = dy = 0.5
dz = y

ax.bar3d(_x, _y, _z, dx, dy, dz, shade=True, color='lightcoral', edgecolor='black')
ax.set_title("3D Visualization of Binomial Distribution")
ax.set_xlabel("Number of successes (k)")
ax.set_ylabel("Trial index (arbitrary)")
ax.set_zlabel("Probability (P)")
plt.show()

# Animation Visualization
fig, ax = plt.subplots(figsize=(8, 6))
bars = plt.bar(x, [0]*len(x), color='lightgreen', edgecolor='black')
plt.title("Animated Binomial Distribution")
plt.xlabel("Number of successes (k)")
plt.ylabel("Probability (P)")

def update(frame):
    for bar, height in zip(bars, y[:frame + 1]):
        bar.set_height(height)

ani = FuncAnimation(fig, update, frames=len(x), repeat=False, interval=500)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Function to plot Bernoulli Distribution
def plot_bernoulli(p):
    x = np.array([0, 1])  # Possible outcomes: 0 (failure), 1 (success)
    pmf = [1-p, p]  # Probabilities of failure and success

    plt.bar(x, pmf, color=['red', 'green'], width=0.4)
    plt.xticks([0, 1], labels=['Failure (0)', 'Success (1)'])
    plt.title(f'Bernoulli Distribution (p = {p})')
    plt.xlabel('Outcomes')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.show()

# Plot for different values of p
plot_bernoulli(0.5)  # Fair coin
plot_bernoulli(0.7)  # Biased coin with p=0.7

from matplotlib.animation import FuncAnimation

# Initialize the figure
fig, ax = plt.subplots()
x = np.array([0, 1])  # Outcomes
bars = ax.bar(x, [0, 1], color=['red', 'green'], width=0.4)

# Update function for animation
def update(p):
    pmf = [1-p, p]  # Probabilities of failure and success
    for bar, h in zip(bars, pmf):
        bar.set_height(h)
    ax.set_title(f'Bernoulli Distribution (p = {p:.2f})')

# Animate for p values between 0 and 1
ani = FuncAnimation(fig, update, frames=np.linspace(0, 1, 100), repeat=True)

plt.xlabel('Outcomes')
plt.ylabel('Probability')
plt.xticks([0, 1], labels=['Failure (0)', 'Success (1)'])
plt.ylim(0, 1)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Continuous Probability Distribution (Example: Exponential Distribution)
def continuous_distribution(x, rate=1):
    return rate * np.exp(-rate * x)

# 2D Plot of Continuous Distribution
def plot_2d_distribution():
    x = np.linspace(0, 5, 500)
    y = continuous_distribution(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Exponential Distribution (rate=1)", color="blue")
    plt.fill_between(x, y, color="lightblue", alpha=0.5)
    plt.title("2D Continuous Probability Distribution")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid()
    plt.show()

# 3D Visualization
def plot_3d_distribution():
    x = np.linspace(0, 5, 500)
    y = np.linspace(0, 5, 500)
    X, Y = np.meshgrid(x, y)
    Z = continuous_distribution(X + Y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
    ax.set_title("3D Visualization of Continuous Distribution")
    ax.set_xlabel("X (Time 1)")
    ax.set_ylabel("Y (Time 2)")
    ax.set_zlabel("Probability Density")
    plt.show()

# Plot and animate
plot_2d_distribution()
plot_3d_distribution()

# Sample continuous_distribution function for illustration
def continuous_distribution(x):
    return np.exp(-x**2)

# Animation: Building a Continuous Distribution
def animate_continuous_distribution():
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 5, 500)
    y = continuous_distribution(x)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.set_title("Building a Continuous Distribution (Animation)")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Probability Density")
    
    # Create an empty plot for line and fill
    line, = ax.plot([], [], lw=2, color="blue")
    fill = ax.fill_between([], [], color="lightblue", alpha=0.5)

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        # Instead of clearing, we just remove the previous fill and add a new one
        for coll in ax.collections:
            coll.set_visible(False)  # Hide previous fill
        ax.fill_between(x[:frame], y[:frame], color="lightblue", alpha=0.5)
        return line,  # Returning the updated objects (line and fill)

    anim = FuncAnimation(fig, update, frames=len(x), interval=10, blit=False)
    plt.show()

# Plot and animate
animate_continuous_distribution()



import numpy as np
import matplotlib.pyplot as plt

# Define the uniform distribution parameters
x = np.linspace(0, 5, 500)
y = np.ones_like(x) / 5  # uniform distribution between 0 and 5

# Plotting the PDF
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="PDF", color="blue")
plt.fill_between(x, y, color="lightblue", alpha=0.5)
plt.title("Uniform Distribution (PDF)")
plt.xlabel("Time (minutes)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Define a Gaussian distribution
mu, sigma = 2.5, 1  # Mean and standard deviation
x = np.linspace(0, 5, 500)
y = np.linspace(0, 5, 500)
X, Y = np.meshgrid(x, y)
Z = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - mu) ** 2 + (Y - mu) ** 2) / sigma ** 2)

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title("3D Plot of Gaussian Distribution")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Probability Density")
plt.show()

from matplotlib.animation import FuncAnimation

# Define a simple exponential distribution for animation
def continuous_distribution(x):
    return np.exp(-x)  # Exponential decay function

# Create the animation
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 5, 500)
y = continuous_distribution(x)
line, = ax.plot([], [], lw=2, color="blue")
ax.set_xlim(0, 5)
ax.set_ylim(0, 1)
ax.set_title("Animated Continuous Distribution")
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Probability Density")

def update(frame):
    line.set_data(x[:frame], y[:frame])
    return line,

# Create the animation
anim = FuncAnimation(fig, update, frames=len(x), interval=30, blit=False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Discrete distribution data
x = np.array([1, 2, 3, 4, 5])
pmf = np.array([0.1, 0.2, 0.3, 0.2, 0.2])  # Probability mass function

# Compute the CDF (cumulative sum of PMF)
cdf = np.cumsum(pmf)

# Plot CDF
plt.step(x, cdf, where='post', label="CDF", color="red", linewidth=2)
plt.fill_between(x, cdf, color="lightcoral", alpha=0.5)
plt.title("Cumulative Distribution Function (Discrete)")
plt.xlabel("X")
plt.ylabel("Cumulative Probability")
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.show()


# Continuous distribution data (Normal Distribution)
from scipy.stats import norm

x = np.linspace(-5, 5, 500)
pdf = norm.pdf(x, 0, 1)  # Standard normal distribution
cdf = norm.cdf(x, 0, 1)

# Plot CDF and PDF
plt.figure(figsize=(10, 6))
plt.plot(x, cdf, label="CDF", color="blue", linewidth=2)
plt.fill_between(x, cdf, color="lightblue", alpha=0.5)
plt.title("Cumulative Distribution Function (Continuous)")
plt.xlabel("X")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.grid(True)
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define bivariate normal distribution
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]  # Covariance matrix
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = multivariate_normal(mean, cov).pdf(np.dstack((X, Y)))

# 3D CDF surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.set_title("3D Cumulative Distribution Function (Bivariate Normal)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Probability Density")
plt.show()


from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

# Continuous distribution: Exponential Distribution
lambda_param = 1
x = np.linspace(0, 10, 500)
pdf = lambda_param * np.exp(-lambda_param * x)
cdf = np.cumsum(pdf) * (x[1] - x[0])  # Approximate CDF

# Plot setup
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)
ax.set_title("Animated CDF of Exponential Distribution")
ax.set_xlabel("X")
ax.set_ylabel("Cumulative Probability")
line, = ax.plot([], [], color='blue', lw=2)

# Update function for animation
def update(frame):
    line.set_data(x[:frame], cdf[:frame])
    return line,

# Create animation
anim = FuncAnimation(fig, update, frames=len(x), interval=30, blit=True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define the interval [a, b]
a = 0
b = 15

# Define x values for plotting the PDF
x_pdf = np.linspace(a, b, 1000)
pdf = np.ones_like(x_pdf) / (b - a)

# Define x values for plotting the CDF
x_cdf = np.linspace(a, b, 1000)
cdf = (x_cdf - a) / (b - a)

# Create the figure and axes
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot the PDF
ax[0].plot(x_pdf, pdf, label="PDF", color="blue")
ax[0].fill_between(x_pdf, 0, pdf, alpha=0.3, color="blue")
ax[0].set_title("Uniform Distribution - PDF")
ax[0].set_xlabel("x")
ax[0].set_ylabel("Probability Density")
ax[0].grid(True)

# Plot the CDF
ax[1].plot(x_cdf, cdf, label="CDF", color="green")
ax[1].fill_between(x_cdf, 0, cdf, alpha=0.3, color="green")
ax[1].set_title("Uniform Distribution - CDF")
ax[1].set_xlabel("x")
ax[1].set_ylabel("Cumulative Probability")
ax[1].grid(True)

plt.tight_layout()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Define the range for x and y
x = np.linspace(a, b, 100)
y = np.linspace(a, b, 100)

# Create a meshgrid for 3D surface plotting
X, Y = np.meshgrid(x, y)

# Z values are constant, since it's a uniform distribution
Z = np.ones_like(X) / (b - a)

# Create a figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis')

# Labels and title
ax.set_title("3D Surface Plot of Uniform Distribution")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Probability Density")
plt.show()


import matplotlib.animation as animation

# Create a figure and axis
fig, ax = plt.subplots(figsize=(7, 5))

# Plot static CDF for reference
ax.plot(x_cdf, cdf, label="CDF", color="green")
ax.set_xlim([a, b])
ax.set_ylim([0, 1])
ax.set_title("CDF of Uniform Distribution (Animated)")
ax.set_xlabel("x")
ax.set_ylabel("Cumulative Probability")

# Initial plot of the area under the curve
line, = ax.plot([], [], color="red")

# Function to update the plot for the animation
def update(frame):
    # Update the line to show cumulative area up to the frame
    line.set_data(x_cdf[:frame], cdf[:frame])
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(x_cdf), interval=50, blit=True)

# Show the animation
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Define parameters for uniform distribution
a = 0  # Start of the interval
b = 15  # End of the interval

# Generate values for plotting
x = np.linspace(a-1, b+1, 1000)  # x-axis values
pdf = np.ones_like(x) / (b - a)  # PDF: constant value over [a, b]
pdf[x < a] = 0  # PDF is 0 before 'a'
pdf[x > b] = 0  # PDF is 0 after 'b'

# CDF: For x between a and b, it's (x - a) / (b - a)
cdf = np.clip((x - a) / (b - a), 0, 1)

# 2D Plot of PDF and CDF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot PDF
ax1.plot(x, pdf, label='PDF', color='blue')
ax1.fill_between(x, 0, pdf, color='blue', alpha=0.3)
ax1.set_title('Probability Density Function (PDF)')
ax1.set_xlabel('x')
ax1.set_ylabel('Probability Density')
ax1.axvline(x=a, color='black', linestyle='--', label=f'a = {a}')
ax1.axvline(x=b, color='black', linestyle='--', label=f'b = {b}')
ax1.legend()

# Plot CDF
ax2.plot(x, cdf, label='CDF', color='green')
ax2.fill_between(x, 0, cdf, color='green', alpha=0.3)
ax2.set_title('Cumulative Distribution Function (CDF)')
ax2.set_xlabel('x')
ax2.set_ylabel('Cumulative Probability')
ax2.axvline(x=a, color='black', linestyle='--', label=f'a = {a}')
ax2.axvline(x=b, color='black', linestyle='--', label=f'b = {b}')
ax2.legend()

plt.tight_layout()
plt.show()

# 3D Surface Plot
X = np.linspace(a, b, 100)
Y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(X, Y)
Z = np.ones_like(X) * (1 / (b - a))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title('3D Surface Plot of Uniform Distribution PDF')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Probability Density')

plt.show()

# Animation of CDF
fig, ax = plt.subplots(figsize=(8, 6))

def update(frame):
    ax.clear()
    ax.plot(x, cdf, label='CDF', color='green')
    ax.fill_between(x, 0, cdf, color='green', alpha=0.3)
    ax.axvline(x=x[frame], color='red', linestyle='--', label=f'Current x = {x[frame]:.2f}')
    ax.set_title('Animated Cumulative Distribution Function (CDF)')
    ax.set_xlabel('x')
    ax.set_ylabel('Cumulative Probability')
    ax.legend()

ani = animation.FuncAnimation(fig, update, frames=range(0, len(x), 10), interval=100, repeat=False)
plt.show()
