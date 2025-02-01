import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Generate random coin tosses (1 for heads, 0 for tails)
np.random.seed(42)
tosses = np.random.choice([0, 1], size=50, p=[0.2, 0.8])  # Simulate 50 tosses

# Helper function to update Bayesian belief
# We use Beta distribution where alpha is number of heads + 1, beta is number of tails + 1
def bayesian_update(tosses):
    heads = np.cumsum(tosses)
    trials = np.arange(1, len(tosses) + 1)
    tails = trials - heads
    return heads + 1, tails + 1

# Initial Priors
prior_alpha, prior_beta = 1, 1  # Uniform prior (belief of a fair coin)

# Frequentist MLE Plot
freq_prob = np.cumsum(tosses) / (np.arange(1, len(tosses) + 1))

# Bayesian Posterior Update
posterior_alpha, posterior_beta = bayesian_update(tosses)

# Plot 2D visualization
plt.figure(figsize=(12, 6))
plt.plot(freq_prob, label="Frequentist Probability Estimate", color='blue')
plt.plot((posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2), '--', label="Bayesian Posterior Mean", color='green')
plt.title("Frequentist vs Bayesian Coin Toss Estimates")
plt.xlabel("Number of Tosses")
plt.ylabel("Probability of Heads")
plt.legend()
plt.grid(True)
plt.show()

# 3D Visualization of Bayesian Beliefs
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Define grid for priors and posteriors
alpha_values = np.linspace(1, 10, 50)
beta_values = np.linspace(1, 10, 50)
alpha_grid, beta_grid = np.meshgrid(alpha_values, beta_values)
posterior_pdf = beta.pdf(0.8, alpha_grid, beta_grid)

ax.plot_surface(alpha_grid, beta_grid, posterior_pdf, cmap='viridis', edgecolor='none')
ax.set_title("Bayesian Belief Posterior Distribution")
ax.set_xlabel("Alpha (Heads + 1)")
ax.set_ylabel("Beta (Tails + 1)")
ax.set_zlabel("Probability Density")
plt.show()

# Animation: Posterior Updating
fig, ax = plt.subplots(figsize=(10, 5))
x = np.linspace(0, 1, 200)

# Initialize the plot
def init():
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    ax.set_title("Bayesian Posterior Updating")
    ax.set_xlabel("Probability of Heads")
    ax.set_ylabel("Density")
    return ax,

# Update function for each frame
def update(frame):
    ax.clear()
    alpha = posterior_alpha[frame]
    beta_val = posterior_beta[frame]
    y = beta.pdf(x, alpha, beta_val)
    ax.plot(x, y, color='green')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    ax.set_title(f"Toss {frame + 1}: Bayesian Update")
    ax.set_xlabel("Probability of Heads")
    ax.set_ylabel("Density")
    return ax,

ani = FuncAnimation(fig, update, frames=len(tosses), init_func=init, blit=False, repeat=False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Probabilities
P_movies = 0.8  # Probability of watching movies
P_contest = 0.1  # Probability of a popcorn-throwing contest
P_popcorn_given_movies = 0.7  # Probability of popcorn given movies
P_popcorn_given_contest = 0.99  # Probability of popcorn given contest

# Calculate joint probabilities
P_popcorn_and_movies = P_popcorn_given_movies * P_movies
P_popcorn_and_contest = P_popcorn_given_contest * P_contest

# ==================================================
# 1. 2D Bar Plot: Joint Probabilities
# ==================================================
scenarios = ["Movies", "Popcorn Contest"]
joint_probabilities = [P_popcorn_and_movies, P_popcorn_and_contest]

plt.figure(figsize=(8, 6))
plt.bar(scenarios, joint_probabilities, color=["blue", "orange"])
plt.title("Joint Probabilities: Movies vs Popcorn Contest")
plt.xlabel("Scenario")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.show()

# ==================================================
# 2. 3D Surface Plot: Probability as a Function of P(movies) and P(contest)
# ==================================================
P_movies_range = np.linspace(0, 1, 100)  # Range of P(movies)
P_contest_range = np.linspace(0, 1, 100)  # Range of P(contest)
P_movies_grid, P_contest_grid = np.meshgrid(P_movies_range, P_contest_range)

# Calculate joint probabilities for the grid
P_popcorn_and_movies_grid = P_popcorn_given_movies * P_movies_grid
P_popcorn_and_contest_grid = P_popcorn_given_contest * P_contest_grid

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(P_movies_grid, P_contest_grid, P_popcorn_and_movies_grid, cmap="viridis", alpha=0.7, label="Movies")
ax.plot_surface(P_movies_grid, P_contest_grid, P_popcorn_and_contest_grid, cmap="plasma", alpha=0.7, label="Contest")
ax.set_xlabel("P(Movies)")
ax.set_ylabel("P(Contest)")
ax.set_zlabel("Joint Probability")
ax.set_title("3D Surface Plot: Joint Probabilities")
plt.show()

# ==================================================
# 3. Animation: Changing P(movies) and P(contest) Over Time
# ==================================================
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("P(Movies)")
ax.set_ylabel("Joint Probability")
ax.set_title("Animation: Joint Probability of Movies vs Contest")
ax.grid()

# Initialize lines for movies and contest
line_movies, = ax.plot([], [], color="blue", label="Movies")
line_contest, = ax.plot([], [], color="orange", label="Contest")
ax.legend()

# Animation function
def animate(frame):
    P_movies_current = frame / 100  # Vary P(movies) from 0 to 1
    P_contest_current = 1 - P_movies_current  # Vary P(contest) inversely

    # Calculate joint probabilities
    P_popcorn_and_movies_current = P_popcorn_given_movies * P_movies_current
    P_popcorn_and_contest_current = P_popcorn_given_contest * P_contest_current

    # Update lines
    line_movies.set_data(P_movies_range[:frame], P_popcorn_given_movies * P_movies_range[:frame])
    line_contest.set_data(P_movies_range[:frame], P_popcorn_given_contest * (1 - P_movies_range[:frame]))
    return line_movies, line_contest

ani = FuncAnimation(fig, animate, frames=100, interval=100, blit=True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Generate synthetic dataset
np.random.seed(42)
x = np.linspace(-10, 10, 50)
y_true = 2 * x ** 2 - 4 * x + 5
noise = np.random.normal(0, 10, size=x.shape)
y_noisy = y_true + noise

# Define models
def model_linear(x):
    return 4 * x + 3

def model_quadratic(x):
    return 2 * x ** 2 - 4 * x + 5

def model_poly10(x):
    coeffs = [0.01, -0.05, 0.3, -1.5, 2.0, -0.1, 0.01, 0, 0.1, -0.2, 5]
    return sum(c * x ** i for i, c in enumerate(coeffs))

# Compute losses and L2 penalties
loss_linear = np.mean((y_noisy - model_linear(x)) ** 2)
loss_quadratic = np.mean((y_noisy - model_quadratic(x)) ** 2)
loss_poly10 = np.mean((y_noisy - model_poly10(x)) ** 2)

coeffs_linear = [4]
coeffs_quadratic = [2, -4]
coeffs_poly10 = [0.01, -0.05, 0.3, -1.5, 2.0, -0.1, 0.01, 0, 0.1, -0.2]

l2_penalty_linear = sum(np.array(coeffs_linear) ** 2)
l2_penalty_quadratic = sum(np.array(coeffs_quadratic) ** 2)
l2_penalty_poly10 = sum(np.array(coeffs_poly10) ** 2)

# Regularized errors
regularization_param = 1.0
reg_error_linear = loss_linear + regularization_param * l2_penalty_linear
reg_error_quadratic = loss_quadratic + regularization_param * l2_penalty_quadratic
reg_error_poly10 = loss_poly10 + regularization_param * l2_penalty_poly10

# Print losses and penalties
print("Losses:")
print(f"Linear Loss: {loss_linear:.2f}, Regularized Error: {reg_error_linear:.2f}")
print(f"Quadratic Loss: {loss_quadratic:.2f}, Regularized Error: {reg_error_quadratic:.2f}")
print(f"Degree 10 Polynomial Loss: {loss_poly10:.2f}, Regularized Error: {reg_error_poly10:.2f}")

# Plotting 2D visualizations
plt.figure(figsize=(12, 6))
plt.scatter(x, y_noisy, color='gray', label='Noisy Data')
plt.plot(x, model_linear(x), label='Linear Model', color='blue')
plt.plot(x, model_quadratic(x), label='Quadratic Model', color='green')
plt.plot(x, model_poly10(x), label='Degree 10 Polynomial Model', color='red')
plt.legend()
plt.title("Model Fits")
plt.show()

# 3D Plot of Loss Surface
X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = (2 * X ** 2 - 4 * X + 5 - Y) ** 2

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.set_title("Loss Surface")
ax.set_xlabel("X Values")
ax.set_ylabel("Y Values")
ax.set_zlabel("Loss")
plt.show()

# Animation Visualization for Loss Reduction
fig, ax = plt.subplots(figsize=(8, 6))

x_vals = np.linspace(-10, 10, 100)
line, = ax.plot(x_vals, model_linear(x_vals), color='blue', label='Model Fit')
ax.scatter(x, y_noisy, color='gray', label='Data Points')
ax.set_title("Model Fitting Animation")
ax.legend()

# Animation function
def update(frame):
    if frame < 50:
        line.set_ydata(model_linear(x_vals) * (1 - frame / 100) + model_quadratic(x_vals) * (frame / 100))
    else:
        line.set_ydata(model_quadratic(x_vals))
    return line,

ani = FuncAnimation(fig, update, frames=100, blit=True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
np.random.seed(0)
x = np.sort(np.random.rand(30) * 10)  # Features
y = 0.5 * x ** 2 - 2 * x + 3 + np.random.randn(30) * 3  # True quadratic with noise

x_plot = np.linspace(0, 10, 100).reshape(-1, 1)

# Helper function to plot models
def plot_model_fit(models, degrees, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='black', label='Data Points')

    for model, degree in zip(models, degrees):
        # Polynomial transformation
        poly_features = PolynomialFeatures(degree=degree)
        x_poly = poly_features.fit_transform(x.reshape(-1, 1))
        model.fit(x_poly, y)
        y_plot = model.predict(poly_features.fit_transform(x_plot))

        # Labeling based on degree
        label = f"Degree {degree} Model"
        plt.plot(x_plot, y_plot, label=label)

    plt.xlabel("Feature X")
    plt.ylabel("Target Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Underfit (Linear Model), Best Fit (Quadratic), Overfit (Degree 10)
linear_model = LinearRegression()
quadratic_model = LinearRegression()
complex_model = LinearRegression()
models = [linear_model, quadratic_model, complex_model]
degrees = [1, 2, 10]

plot_model_fit(models, degrees, title="Underfit, Best Fit, and Overfit Models")

# Visualization: Regularized vs Unregularized Model
ridge_model = Ridge(alpha=10)  # Regularized model
unregularized_model = LinearRegression()  # Unregularized model

models = [unregularized_model, ridge_model]
labels = ["Unregularized Model", "Regularized Model (Ridge, alpha=10)"]

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Data Points')

for model, label in zip(models, labels):
    # Use a high-degree polynomial for better visualization
    poly_features = PolynomialFeatures(degree=10)
    x_poly = poly_features.fit_transform(x.reshape(-1, 1))
    model.fit(x_poly, y)
    y_plot = model.predict(poly_features.fit_transform(x_plot))

    plt.plot(x_plot, y_plot, label=label)

plt.xlabel("Feature X")
plt.ylabel("Target Y")
plt.title("Regularized vs Unregularized Model")
plt.legend()
plt.grid(True)
plt.show()

# Animation Visualization
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, color='black', label='Data Points')

line, = ax.plot([], [], lw=2)

ax.set_xlim(0, 10)
ax.set_ylim(min(y) - 5, max(y) + 5)
ax.set_xlabel("Feature X")
ax.set_ylabel("Target Y")
ax.set_title("Regularization Animation")
ax.legend()

alphas = np.linspace(0, 50, 100)  # Range of regularization parameters
poly_features = PolynomialFeatures(degree=10)
x_poly = poly_features.fit_transform(x.reshape(-1, 1))

# Animation update function
def update(frame):
    alpha = alphas[frame]
    ridge = Ridge(alpha=alpha)
    ridge.fit(x_poly, y)
    y_plot = ridge.predict(poly_features.fit_transform(x_plot))
    line.set_data(x_plot, y_plot)
    ax.set_title(f"Ridge Regularization (Alpha={alpha:.2f})")
    return line,

ani = FuncAnimation(fig, update, frames=len(alphas), blit=True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.animation as animation

# Generate sample dataset
np.random.seed(0)
x = np.sort(np.random.rand(100) * 10)  # Feature values
y = np.sin(x) + 0.1 * np.random.randn(100)  # Target values with noise
X = x[:, np.newaxis]

# Define polynomial degrees for underfit, best fit, and overfit
degrees = [1, 2, 10]

# Plot underfit, best fit, and overfit models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, degree in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)

    axes[i].scatter(x, y, label="Data", color='blue')
    axes[i].plot(x, y_pred, color='red', lw=2, label=f"Degree {degree}")
    axes[i].legend()
    axes[i].set_title(f"Degree {degree} Model")
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("y")

plt.suptitle("Underfit, Best Fit, and Overfit Models")
plt.show()

# Plot Regularized vs Unregularized Models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Unregularized Polynomial Regression (Degree 10)
unreg_model = make_pipeline(PolynomialFeatures(10), LinearRegression())
unreg_model.fit(X, y)
y_pred_unreg = unreg_model.predict(X)
ax1.scatter(x, y, label="Data", color='blue')
ax1.plot(x, y_pred_unreg, color='orange', lw=2, label="Unregularized")
ax1.legend()
ax1.set_title("Unregularized (Degree 10)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Regularized (Ridge Regression with Degree 10)
ridge_model = make_pipeline(PolynomialFeatures(10), Ridge(alpha=10))
ridge_model.fit(X, y)
y_pred_ridge = ridge_model.predict(X)
ax2.scatter(x, y, label="Data", color='blue')
ax2.plot(x, y_pred_ridge, color='green', lw=2, label="Regularized (Ridge)")
ax2.legend()
ax2.set_title("Regularized (Degree 10, Alpha=10)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.suptitle("Regularized vs Unregularized Models")
plt.show()

# Animation to visualize the effect of regularization (varying alpha)
def animate_ridge_effect():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color='blue', label="Data")
    ax.set_title("Effect of Regularization on Model Fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    line, = ax.plot([], [], color='purple', lw=2, label="Ridge Model")

    def init():
        line.set_data([], [])
        return line,

    def update(alpha):
        ridge_model = make_pipeline(PolynomialFeatures(10), Ridge(alpha=alpha))
        ridge_model.fit(X, y)
        y_pred = ridge_model.predict(X)
        line.set_data(x, y_pred)
        ax.legend([f"Alpha: {alpha:.2f}"])
        return line,

    ani = animation.FuncAnimation(
        fig, update, frames=np.linspace(0.01, 50, 100), init_func=init,
        blit=True, repeat=False)
    plt.legend()
    plt.show()

animate_ridge_effect()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

# Generate synthetic data points
np.random.seed(42)
x = np.linspace(0, 10, 5)
y_true = 2 * x + 1  # True line: y = 2x + 1
y_observed = y_true + np.random.normal(0, 1, size=len(x))  # Add Gaussian noise

# ==================================================
# 1. 2D Plot: Data Points and Fitted Lines
# ==================================================
# Define three candidate lines (models)
def line1(x):
    return 1.5 * x + 0.5  # Model 1

def line2(x):
    return 2 * x + 1  # Model 2 (true line)

def line3(x):
    return 2.5 * x + 1.5  # Model 3

# Plot data points and candidate lines
plt.figure(figsize=(10, 6))
plt.scatter(x, y_observed, color="blue", label="Observed Data")
plt.plot(x, line1(x), color="red", label="Model 1")
plt.plot(x, line2(x), color="green", label="Model 2 (True Line)")
plt.plot(x, line3(x), color="orange", label="Model 3")
plt.title("2D Plot: Data Points and Candidate Lines")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# ==================================================
# 2. 3D Surface Plot: Likelihood as a Function of Slope and Intercept
# ==================================================
# Define a function to calculate the likelihood of a line given the data
def likelihood(slope, intercept):
    y_pred = slope * x + intercept
    residuals = y_observed - y_pred
    return np.prod(norm.pdf(residuals, loc=0, scale=1))  # Product of Gaussian likelihoods

# Create a grid of slopes and intercepts
slope_range = np.linspace(1, 3, 100)
intercept_range = np.linspace(0, 2, 100)
slope_grid, intercept_grid = np.meshgrid(slope_range, intercept_range)
likelihood_grid = np.zeros_like(slope_grid)

# Calculate likelihood for each combination of slope and intercept
for i in range(len(slope_range)):
    for j in range(len(intercept_range)):
        likelihood_grid[j, i] = likelihood(slope_grid[j, i], intercept_grid[j, i])

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(slope_grid, intercept_grid, likelihood_grid, cmap="viridis")
ax.set_xlabel("Slope (m)")
ax.set_ylabel("Intercept (b)")
ax.set_zlabel("Likelihood")
ax.set_title("3D Surface Plot: Likelihood as a Function of Slope and Intercept")
plt.show()

# ==================================================
# 3. Animation: Likelihood of Different Lines
# ==================================================
# Define candidate lines for animation
candidate_lines = [
    {"slope": 1.5, "intercept": 0.5, "color": "red", "label": "Model 1"},
    {"slope": 2.0, "intercept": 1.0, "color": "green", "label": "Model 2 (True Line)"},
    {"slope": 2.5, "intercept": 1.5, "color": "orange", "label": "Model 3"}
]

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 25)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Animation: Likelihood of Different Lines")
ax.grid()

# Plot observed data
ax.scatter(x, y_observed, color="blue", label="Observed Data")

# Initialize lines for animation
lines = []
for model in candidate_lines:
    line, = ax.plot([], [], color=model["color"], label=model["label"])
    lines.append(line)

# Animation function
def animate(frame):
    model = candidate_lines[frame]
    y_pred = model["slope"] * x + model["intercept"]
    lines[frame].set_data(x, y_pred)
    ax.legend()
    return lines

ani = FuncAnimation(fig, animate, frames=len(candidate_lines), interval=1000, blit=True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Data setup
num_flips = 10
num_heads = 8
num_tails = num_flips - num_heads

# Function to compute likelihood
p_values = np.linspace(0.01, 0.99, 200)  # Avoid 0 and 1 for numerical stability
likelihood_values = (p_values ** num_heads) * ((1 - p_values) ** num_tails)
log_likelihood_values = num_heads * np.log(p_values) + num_tails * np.log(1 - p_values)

# 2D Plot of likelihood and log-likelihood
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Likelihood Plot
axes[0].plot(p_values, likelihood_values, color='blue', label='Likelihood')
axes[0].set_title('Likelihood Function')
axes[0].set_xlabel('P (Head Probability)')
axes[0].set_ylabel('Likelihood')
axes[0].legend()

# Log-Likelihood Plot
axes[1].plot(p_values, log_likelihood_values, color='orange', label='Log-Likelihood')
axes[1].set_title('Log-Likelihood Function')
axes[1].set_xlabel('P (Head Probability)')
axes[1].set_ylabel('Log-Likelihood')
axes[1].legend()

plt.tight_layout()
plt.show()

# 3D Plot of the likelihood function
def plot_3d_likelihood():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid for flips and head probabilities
    P, K = np.meshgrid(p_values, np.arange(0, num_flips + 1, 1))
    likelihood_grid = (P ** K) * ((1 - P) ** (num_flips - K))

    ax.plot_surface(P, K, likelihood_grid, cmap='viridis', edgecolor='none')
    ax.set_title('3D Likelihood Function')
    ax.set_xlabel('P (Head Probability)')
    ax.set_ylabel('Number of Heads')
    ax.set_zlabel('Likelihood')
    
    plt.show()

plot_3d_likelihood()

# Animation of likelihood changes with respect to number of flips
def animate_likelihood():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    line, = ax.plot([], [], lw=2)
    ax.set_title('Likelihood vs Head Probability Over Flips')
    ax.set_xlabel('P (Head Probability)')
    ax.set_ylabel('Likelihood')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        num_heads_frame = frame
        num_tails_frame = num_flips - frame
        likelihood_values_frame = (p_values ** num_heads_frame) * ((1 - p_values) ** num_tails_frame)
        line.set_data(p_values, likelihood_values_frame)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, num_flips + 1), init_func=init, blit=True)
    plt.show()

animate_likelihood()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

# Observed data points
data_points = np.array([1, -1])

# ==================================================
# 1. 2D Plot: Gaussian Distributions and Data Points
# ==================================================
# Define Gaussian distributions
def gaussian(x, mean, std):
    return norm.pdf(x, loc=mean, scale=std)

# Plot observed data points and Gaussian distributions
x = np.linspace(-5, 5, 1000)
means = [10, 2, -1, 0, 1]  # Means of the Gaussian distributions
stds = [1, 1, 1, 1, 1]  # Standard deviations of the Gaussian distributions

plt.figure(figsize=(10, 6))
for mean, std in zip(means, stds):
    plt.plot(x, gaussian(x, mean, std), label=f"Mean = {mean}, Std = {std}")

# Plot observed data points
plt.scatter(data_points, [0, 0], color="red", label="Observed Data")
plt.title("2D Plot: Gaussian Distributions and Data Points")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

# ==================================================
# 2. 3D Surface Plot: Likelihood as a Function of Mean and Standard Deviation
# ==================================================
# Define a function to calculate the likelihood of a Gaussian given the data
def likelihood(mean, std):
    return np.prod(norm.pdf(data_points, loc=mean, scale=std))  # Product of likelihoods

# Create a grid of means and standard deviations
mean_range = np.linspace(-5, 5, 100)
std_range = np.linspace(0.1, 5, 100)
mean_grid, std_grid = np.meshgrid(mean_range, std_range)
likelihood_grid = np.zeros_like(mean_grid)

# Calculate likelihood for each combination of mean and standard deviation
for i in range(len(mean_range)):
    for j in range(len(std_range)):
        likelihood_grid[j, i] = likelihood(mean_grid[j, i], std_grid[j, i])

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(mean_grid, std_grid, likelihood_grid, cmap="viridis")
ax.set_xlabel("Mean (μ)")
ax.set_ylabel("Standard Deviation (σ)")
ax.set_zlabel("Likelihood")
ax.set_title("3D Surface Plot: Likelihood as a Function of Mean and Standard Deviation")
plt.show()

# ==================================================
# 3. Animation: Likelihood of Different Gaussian Distributions
# ==================================================
# Define candidate Gaussian distributions for animation
candidate_gaussians = [
    {"mean": 10, "std": 1, "color": "blue", "label": "Mean = 10, Std = 1"},
    {"mean": 2, "std": 1, "color": "green", "label": "Mean = 2, Std = 1"},
    {"mean": -1, "std": 1, "color": "orange", "label": "Mean = -1, Std = 1"},
    {"mean": 0, "std": 1, "color": "red", "label": "Mean = 0, Std = 1"},
    {"mean": 1, "std": 1, "color": "purple", "label": "Mean = 1, Std = 1"}
]

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.5)
ax.set_xlabel("x")
ax.set_ylabel("Probability Density")
ax.set_title("Animation: Likelihood of Different Gaussian Distributions")
ax.grid()

# Plot observed data points
ax.scatter(data_points, [0, 0], color="red", label="Observed Data")

# Initialize lines for animation
lines = []
for model in candidate_gaussians:
    line, = ax.plot([], [], color=model["color"], label=model["label"])
    lines.append(line)

# Animation function
def animate(frame):
    model = candidate_gaussians[frame]
    y = gaussian(x, model["mean"], model["std"])
    lines[frame].set_data(x, y)
    ax.legend()
    return lines

ani = FuncAnimation(fig, animate, frames=len(candidate_gaussians), interval=1000, blit=True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Example Data
scenarios = ["Movies", "Board Games", "Nap"]
probabilities = [0.9, 0.5, 0.1]  # P(popcorn|scenario)

# Bar Plot for Scenario Probabilities (2D Visualization)
def plot_2d_probabilities():
    plt.figure(figsize=(8, 5))
    plt.bar(scenarios, probabilities, color=['blue', 'orange', 'green'])
    plt.title('Conditional Probability of Popcorn Given Scenarios')
    plt.ylabel('Probability')
    plt.show()

# 3D Visualization: Surface plot for likelihood function
# Define P(Data|Model) surface
p_data_given_model = lambda x, y: np.exp(-((x - 0.6)**2 + (y - 0.4)**2))  # Hypothetical likelihood

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = p_data_given_model(X, Y)

# 3D plot function
def plot_3d_likelihood():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('3D Likelihood Surface')
    ax.set_xlabel('Model Parameter 1')
    ax.set_ylabel('Model Parameter 2')
    ax.set_zlabel('Likelihood')
    plt.show()

# Animation for Linear Regression Model Fit
def plot_regression_animation():
    # Generate random data points for regression
    np.random.seed(42)
    x_points = np.random.uniform(0, 10, 20)
    noise = np.random.normal(0, 0.5, len(x_points))
    y_points = 2 * x_points + 1 + noise  # True model: y = 2x + 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_points, y_points, color='purple', label='Data Points')
    ax.set_title('Linear Regression Model Fitting')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend()

    # Plot regression lines iteratively
    line, = ax.plot([], [], color='red', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        slope = frame / 50  # Gradually adjust the slope
        intercept = 1  # Fixed intercept
        y_fit = slope * x_points + intercept
        line.set_data(x_points, y_fit)
        return line,

    ani = FuncAnimation(fig, update, frames=np.arange(1, 101), init_func=init,
                         blit=True, interval=100)

    plt.show()

# Plot the visualizations
plot_2d_probabilities()  # 2D Conditional Probability Plot
plot_3d_likelihood()     # 3D Likelihood Surface
plot_regression_animation()  # Linear Regression Animation
