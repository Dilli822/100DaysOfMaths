import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Define parameters
mu_0 = 66.7  # Null hypothesis mean
mu_actual = 68.442  # Sample mean
s = 3  # Sample standard deviation
n = 10  # Sample size
alpha = 0.05  # Significance level
df = n - 1  # Degrees of freedom for t-distribution

# Step 2: Compute t-statistic
t_score = (mu_actual - mu_0) / (s / np.sqrt(n))
p_value = 1 - stats.t.cdf(t_score, df)

# Step 3: 2D Plot of t-Distribution
x = np.linspace(mu_0 - 4 * s, mu_0 + 4 * s, 1000)
y_t = stats.t.pdf(x, df, mu_0, s / np.sqrt(n))
y_norm = stats.norm.pdf(x, mu_0, s / np.sqrt(n))

plt.figure(figsize=(10, 5))
sns.lineplot(x=x, y=y_t, label="t-Distribution (df=9)")
sns.lineplot(x=x, y=y_norm, linestyle='--', label="Normal Approximation")
plt.axvline(mu_actual, color='r', linestyle='--', label="Observed Mean")
plt.axvline(mu_0 + stats.t.ppf(1 - alpha, df) * (s / np.sqrt(n)), color='g', linestyle='--', label="Critical Value")
plt.fill_between(x, y_t, where=(x > mu_0 + stats.t.ppf(1 - alpha, df) * (s / np.sqrt(n))), color='red', alpha=0.3, label="Rejection Region")
plt.text(mu_actual, max(y_t)/2, f"Observed Mean = {mu_actual}\nT-score = {t_score:.2f}\nP-value = {p_value:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel("Sample Mean")
plt.ylabel("Probability Density")
plt.title("Hypothesis Testing with t-Distribution")
plt.legend()
plt.show()

# Step 4: 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, np.linspace(0, 1, 10))
Z = stats.t.pdf(X, df, mu_0, s / np.sqrt(n)) * Y
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.set_xlabel("Sample Mean")
ax.set_ylabel("Probability Scale")
ax.set_zlabel("Density")
ax.set_title("3D View of t-Distribution in Hypothesis Testing")
plt.show()

# Step 5: Animation of Sampling Distribution
fig, ax = plt.subplots(figsize=(10, 5))
def animate(i):
    ax.clear()
    sample_means = np.random.standard_t(df, size=i + 1) * (s / np.sqrt(n)) + mu_0
    sns.histplot(sample_means, kde=True, bins=15, ax=ax, color='b', alpha=0.6)
    ax.axvline(mu_0, color='r', linestyle='--', label="Null Hypothesis Mean")
    ax.axvline(mu_actual, color='g', linestyle='--', label="Observed Mean")
    ax.set_title(f"Sampling Distribution (n={i + 1})")
    ax.legend()

ani = animation.FuncAnimation(fig, animate, frames=30, interval=200)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.special import erf  # Import erf from scipy.special

# Define parameters
sigma = 3  # Standard deviation
n = 10  # Sample size
alpha_values = [0.01, 0.05, 0.1]
mu_0 = 66.7  # Null hypothesis population mean
critical_value = 68.26

# Function to calculate the power of the test
def power_of_test(mu, alpha, critical_value, sigma, n):
    z_critical = (critical_value - mu) / (sigma / np.sqrt(n))
    return 1 - 0.5 * (1 + erf(z_critical / np.sqrt(2)))  # Use erf from scipy.special

# Generate 2D power curve
mu_values = np.linspace(66.7, 75, 100)
power_curves = {alpha: [power_of_test(mu, alpha, critical_value, sigma, n) for mu in mu_values]
                 for alpha in alpha_values}

# Plotting the 2D Power Curve
plt.figure(figsize=(10, 5))
for alpha, powers in power_curves.items():
    plt.plot(mu_values, powers, label=f"α = {alpha}")

# Add annotations explaining the significance of the lines
plt.axhline(y=0.95, color='gray', linestyle='--', linewidth=0.7)
plt.text(70, 0.96, 'Power = 0.95', fontsize=12, color='black', verticalalignment='bottom')

plt.axhline(y=0.05, color='gray', linestyle='--', linewidth=0.7)
plt.text(70, 0.045, 'Type I Error Rate (α)', fontsize=12, color='black', verticalalignment='top')

plt.title("Power of Test for Different α Levels")
plt.xlabel("True Population Mean (μ)")
plt.ylabel("Power (1 - β)")
plt.legend()
plt.grid(True)

# Add general annotation for the plot
plt.text(67.5, 0.7, 'Power curves for different significance levels (α)', fontsize=14, color='red')

plt.show()

# 3D Visualization: Power Surface
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
alpha_mesh, mu_mesh = np.meshgrid(alpha_values, mu_values)
power_mesh = np.array([[power_of_test(mu, alpha, critical_value, sigma, n)
                         for alpha in alpha_values] for mu in mu_values])

surf = ax.plot_surface(mu_mesh, alpha_mesh, power_mesh, cmap='viridis', edgecolor='none')
ax.set_xlabel("Population Mean (μ)")
ax.set_ylabel("Significance Level (α)")
ax.set_zlabel("Power (1 - β)")
ax.set_title("3D Visualization of Power Function")

# Add annotations to the 3D plot
ax.text(67.5, 0.02, 0.8, 'Power increases as the population mean shifts from μ₀ = 66.7', fontsize=12, color='red')

plt.colorbar(surf)
plt.show()

# Animation of Gaussian Distributions for Different μ Values
fig, ax = plt.subplots(figsize=(10, 5))

x = np.linspace(60, 75, 500)
line, = ax.plot([], [], lw=2)

def init():
    ax.set_xlim(60, 75)
    ax.set_ylim(0, 0.25)
    ax.set_title("Distribution Changes for Different μ")
    ax.set_xlabel("Sample Mean (x̄)")
    ax.set_ylabel("Probability Density")

    # Adding the critical line with annotation for Type I error
    ax.axvline(critical_value, color='red', linestyle='--', label="Critical Value")
    ax.text(critical_value + 0.5, 0.2, f"Critical Value: {critical_value}", fontsize=12, color='red')
    return line,

def update(frame):
    mu = 66.7 + frame / 10
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    line.set_data(x, pdf)

    # Adding annotation about Type II error area
    ax.fill_between(x, pdf, where=(x < critical_value), color='red', alpha=0.3, label="Type II Error Region (β)")
    
    # Adding annotation about power
    ax.text(70, 0.18, f"Power (1-β) increases as μ increases", fontsize=12, color='green')

    return line,

ani = FuncAnimation(fig, update, frames=np.arange(0, 60), init_func=init, blit=True)

plt.show()



# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Parameters for the distributions
sample_size = 10  # Sample size (n)
df = sample_size - 1  # Degrees of freedom for t-distribution (df = n-1)
mu = 0  # Hypothesized population mean (mean of normal distribution)
sigma = 1  # Hypothesized population standard deviation
n = 1000  # Number of points to plot for the distributions
x_range = np.linspace(-4, 4, 1000)  # Range of x values for plotting

# Create the probability density functions (PDF) for t-distribution and normal distribution
t_dist = t.pdf(x_range, df)  # t-distribution PDF for df degrees of freedom
normal_dist = norm.pdf(x_range, mu, sigma)  # Normal distribution PDF

# 2D Plot: Comparison of Normal vs t-distribution
plt.figure(figsize=(10, 6))  # Set figure size for the 2D plot
# Plot the t-distribution (in red, dashed line)
plt.plot(x_range, t_dist, label=f't-distribution (df={df})', color='red', linestyle='--')
# Plot the normal distribution (in blue)
plt.plot(x_range, normal_dist, label='Normal distribution', color='blue')

# Fill the area under the curves for visual clarity
plt.fill_between(x_range, t_dist, color='red', alpha=0.3)  # Red shading for t-distribution
plt.fill_between(x_range, normal_dist, color='blue', alpha=0.3)  # Blue shading for normal distribution

# Title and axis labels
plt.title('Comparison of t-distribution and Normal Distribution')
plt.xlabel('x')  # Label for x-axis
plt.ylabel('Density')  # Label for y-axis

# Show legend, grid, and plot the figure
plt.legend()
plt.grid(True)
plt.show()

# 3D Plot: 3D visualization of t-distribution and normal distribution
X = np.linspace(-4, 4, 100)  # X values for the 3D surface plot
Y = np.linspace(-4, 4, 100)  # Y values for the 3D surface plot
X, Y = np.meshgrid(X, Y)  # Create a meshgrid for 3D plotting
Z_t = t.pdf(X, df)  # Compute t-distribution values
Z_norm = norm.pdf(X)  # Compute normal distribution values

# Set up the 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the t-distribution surface (in red with some transparency)
ax.plot_surface(X, Y, Z_t, color='r', alpha=0.6, rstride=8, cstride=8)

# Plot the normal distribution surface (in blue with some transparency)
ax.plot_surface(X, Y, Z_norm, color='b', alpha=0.3, rstride=8, cstride=8)

# Title and axis labels for 3D plot
ax.set_title('3D Visualization of t-distribution and Normal Distribution')
ax.set_xlabel('X axis')  # Label for x-axis
ax.set_ylabel('Y axis')  # Label for y-axis
ax.set_zlabel('Density')  # Label for z-axis

# Show the 3D plot
plt.show()

# Animation of t-distribution with changing degrees of freedom
fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure for the animation

# Function to update the plot for each frame in the animation
def update(frame):
    ax.clear()  # Clear previous plot to avoid overlap
    df = frame + 1  # Degrees of freedom change over time (starting from 1)
    t_dist = t.pdf(x_range, df)  # Compute t-distribution with updated df

    # Plot the updated t-distribution (in red, dashed line)
    ax.plot(x_range, t_dist, label=f't-distribution (df={df})', color='red', linestyle='--')
    # Plot the normal distribution (in blue)
    ax.plot(x_range, normal_dist, label='Normal distribution', color='blue')

    # Fill the area under the curves
    ax.fill_between(x_range, t_dist, color='red', alpha=0.3)  # Red shading for t-distribution
    ax.fill_between(x_range, normal_dist, color='blue', alpha=0.3)  # Blue shading for normal distribution

    # Update the title and axis labels
    ax.set_title(f'Comparison of t-distribution (df={df}) and Normal Distribution')
    ax.set_xlabel('x')  # Label for x-axis
    ax.set_ylabel('Density')  # Label for y-axis
    ax.legend()  # Show the legend
    ax.grid(True)  # Show grid

# Create the animation using FuncAnimation
ani = animation.FuncAnimation(fig, update, frames=np.arange(1, 31), interval=500)

# Show the animation
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf  # Import erf from scipy.special

# Define parameters
sigma = 3  # Standard deviation
n = 10  # Sample size
alpha_values = [0.01, 0.05, 0.1]  # Significance levels
mu_0 = 66.7  # Null hypothesis population mean
critical_value = 68.26  # Critical value for hypothesis testing

# Function to calculate the power of the test
def power_of_test(mu, alpha, critical_value, sigma, n):
    z_critical = (critical_value - mu) / (sigma / np.sqrt(n))
    return 1 - 0.5 * (1 + erf(z_critical / np.sqrt(2)))  # Use erf from scipy.special

# Generate 2D power curve
mu_values = np.linspace(66.7, 75, 100)
power_curves = {alpha: [power_of_test(mu, alpha, critical_value, sigma, n) for mu in mu_values]
                 for alpha in alpha_values}

# Plotting the 2D Power Curve
plt.figure(figsize=(10, 5))
for alpha, powers in power_curves.items():
    plt.plot(mu_values, powers, label=f"α = {alpha}")
plt.axhline(y=0.95, color='gray', linestyle='--', linewidth=0.7)
plt.axhline(y=0.05, color='gray', linestyle='--', linewidth=0.7)
plt.title("Power of Test for Different α Levels")
plt.xlabel("True Population Mean (μ)")
plt.ylabel("Power (1 - β)")
plt.legend()
plt.grid(True)

# Add annotations for key points
plt.annotate('Power of test increases\nas μ moves further from μ0', 
             xy=(71, 0.7), xytext=(73, 0.8),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.annotate('Critical α levels\n(0.05 and 0.95)', 
             xy=(66.7, 0.05), xytext=(67.5, 0.1),
             arrowprops=dict(facecolor='red', arrowstyle='->'))

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.animation import FuncAnimation

# Parameters for t-distribution
n = 10  # Sample size
df = n - 1  # Degrees of freedom (n-1)
alpha = 0.05  # Significance level
critical_value = stats.t.ppf(1 - alpha, df)  # Critical t-value for right-tailed test
mu = 66.7  # Null hypothesis mean
sample_mean = 68.442  # Sample mean
sample_variance = 3.113  # Sample variance

# t-statistic calculation
t_statistic = (sample_mean - mu) / np.sqrt(sample_variance / n)

# 1. 2D Plot of t-distribution with shaded area (right-tailed test)
x = np.linspace(-5, 5, 1000)
y = stats.t.pdf(x, df)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label='t-distribution (df=9)', color='blue')
ax.fill_between(x, y, where=(x > critical_value), color='red', alpha=0.5, label='p-value area')
ax.axvline(critical_value, color='green', linestyle='--', label=f'Critical Value ({critical_value:.2f})')
ax.axvline(t_statistic, color='orange', linestyle='--', label=f'T-statistic ({t_statistic:.2f})')

ax.set_title("Right-Tailed Test: t-Distribution")
ax.set_xlabel('t-value')
ax.set_ylabel('Probability Density')
ax.legend()
plt.show()

# 2. 3D Plot for t-distribution surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = stats.t.pdf(X, df)

ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title("3D Surface Plot of t-distribution")
ax.set_xlabel('t-value')
ax.set_ylabel('t-value')
ax.set_zlabel('Probability Density')
plt.show()

# 3. Animation of the right-tailed test's p-value region
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label='t-distribution (df=9)', color='blue')
line_t_stat, = ax.plot([], [], 'r-', label='T-statistic', linewidth=2)

def init():
    ax.fill_between(x, y, color='blue', alpha=0.5)
    return line_t_stat,

def update(frame):
    line_t_stat.set_data([critical_value, frame], [0, stats.t.pdf(frame, df)])
    ax.fill_between(x, y, where=(x > frame), color='red', alpha=0.5, label='p-value area')
    return line_t_stat,

ani = FuncAnimation(fig, update, frames=np.linspace(critical_value, 5, 200), init_func=init, blit=True, interval=50)

plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Parameters
n_small = 10   # Small sample size
n_large = 100  # Large sample size
alpha = 0.05   # Significance level

# Create x-axis range for plotting
x = np.linspace(-5, 5, 1000)

# Normal distribution for Z-test
z_pdf = stats.norm.pdf(x)

# T-distribution for small sample (10 samples)
t_small_pdf = stats.t.pdf(x, df=n_small-1)

# T-distribution for large sample (100 samples)
t_large_pdf = stats.t.pdf(x, df=n_large-1)

# Plot Z-test vs T-test
plt.figure(figsize=(10, 6))

# Plotting Z-test normal distribution
plt.plot(x, z_pdf, label="Z-Test (Normal Distribution)", color="blue", linestyle='dashed')

# Plotting T-test for small sample
plt.plot(x, t_small_pdf, label="T-Test (Small Sample, n=10)", color="red", linestyle='dotted')

# Plotting T-test for large sample
plt.plot(x, t_large_pdf, label="T-Test (Large Sample, n=100)", color="green", linestyle='solid')

plt.axvline(0, color='black', linewidth=1, linestyle="--")  # Mean line (null hypothesis)
plt.title("Z-Test vs T-Test: Distribution Comparison")
plt.xlabel("Test Statistic (t or Z)")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
