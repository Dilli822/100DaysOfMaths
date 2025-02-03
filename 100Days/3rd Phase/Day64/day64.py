import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Prior beliefs (Beta distributions)
prior_conservative = beta(1, 1)  # Very strong belief in a fair coin
prior_mild = beta(2, 2)  # Willing to believe in slight bias
prior_non_informative = beta(1, 1)  # Non-informative prior, same weight for all outcomes

# Coin toss results
heads = 8
tails = 2

# Update beliefs using the Beta distribution (posterior = Beta(alpha + heads, beta + tails))
def update_belief(prior, heads, tails):
    return beta(prior.args[0] + heads, prior.args[1] + tails)

# Posterior beliefs after 8 heads and 2 tails
posterior_conservative = update_belief(prior_conservative, heads, tails)
posterior_mild = update_belief(prior_mild, heads, tails)
posterior_non_informative = update_belief(prior_non_informative, heads, tails)

# Values for the x-axis (probabilities of heads)
x = np.linspace(0, 1, 100)

# Plot the prior and posterior distributions
plt.figure(figsize=(10, 6))

# Prior distributions
plt.plot(x, prior_conservative.pdf(x), label="Conservative Prior", linestyle='--', color='blue')
plt.plot(x, prior_mild.pdf(x), label="Mild Prior", linestyle='--', color='green')
plt.plot(x, prior_non_informative.pdf(x), label="Non-informative Prior", linestyle='--', color='red')

# Posterior distributions
plt.plot(x, posterior_conservative.pdf(x), label="Conservative Posterior", color='blue')
plt.plot(x, posterior_mild.pdf(x), label="Mild Posterior", color='green')
plt.plot(x, posterior_non_informative.pdf(x), label="Non-informative Posterior", color='red')

plt.title("Bayesian Updates with Coin Tosses (8 Heads, 2 Tails)")
plt.xlabel("Probability of Heads")
plt.ylabel("Density")
plt.legend()
plt.show()


import plotly.graph_objects as go
from scipy.stats import beta

# Define the range of probabilities
x = np.linspace(0, 1, 100)

# Create traces for prior and posterior distributions
prior_conservative = beta(1, 1)
prior_mild = beta(2, 2)
prior_non_informative = beta(1, 1)

# Posterior distributions after 8 heads and 2 tails
posterior_conservative = beta(1 + 8, 1 + 2)
posterior_mild = beta(2 + 8, 2 + 2)
posterior_non_informative = beta(1 + 8, 1 + 2)

# Create traces for each distribution (prior and posterior)
trace_conservative_prior = go.Scatter3d(
    x=x, y=prior_conservative.pdf(x), z=np.zeros_like(x),
    mode='lines', name='Conservative Prior', line=dict(color='blue')
)

trace_mild_prior = go.Scatter3d(
    x=x, y=prior_mild.pdf(x), z=np.ones_like(x)*1, 
    mode='lines', name='Mild Prior', line=dict(color='green')
)

trace_non_informative_prior = go.Scatter3d(
    x=x, y=prior_non_informative.pdf(x), z=np.ones_like(x)*2, 
    mode='lines', name='Non-informative Prior', line=dict(color='red')
)

trace_conservative_posterior = go.Scatter3d(
    x=x, y=posterior_conservative.pdf(x), z=np.ones_like(x)*3, 
    mode='lines', name='Conservative Posterior', line=dict(color='blue', dash='dash')
)

trace_mild_posterior = go.Scatter3d(
    x=x, y=posterior_mild.pdf(x), z=np.ones_like(x)*4, 
    mode='lines', name='Mild Posterior', line=dict(color='green', dash='dash')
)

trace_non_informative_posterior = go.Scatter3d(
    x=x, y=posterior_non_informative.pdf(x), z=np.ones_like(x)*5, 
    mode='lines', name='Non-informative Posterior', line=dict(color='red', dash='dash')
)

# Set up the layout
layout = go.Layout(
    title="Bayesian Updates (Prior vs Posterior)",
    scene=dict(
        xaxis_title="Probability of Heads",
        yaxis_title="Density",
        zaxis_title="Distributions"
    ),
    showlegend=True
)

# Create the figure
fig = go.Figure(data=[trace_conservative_prior, trace_mild_prior, trace_non_informative_prior,
                     trace_conservative_posterior, trace_mild_posterior, trace_non_informative_posterior],
                layout=layout)

# Show the plot
fig.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import beta

# Define the initial prior (non-informative)
prior = beta(1, 1)

# Number of tosses and results (e.g., heads and tails)
tosses = [(1, 0)] * 8 + [(0, 1)] * 2  # 8 heads, 2 tails

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 1, 100)
line, = ax.plot(x, prior.pdf(x), label="Prior", color='red')
ax.set_ylim(0, 5)
ax.set_title("Updating Beliefs with Coin Tosses")
ax.set_xlabel("Probability of Heads")
ax.set_ylabel("Density")
ax.legend()

# Function to update the plot at each frame
def update(frame):
    heads, tails = tosses[frame]
    posterior = beta(prior.args[0] + heads, prior.args[1] + tails)
    line.set_ydata(posterior.pdf(x))  # Update the curve
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(tosses), repeat=False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 2D Visualization of Bayesian Priors and Posterior
def plot_2d():
    # Parameters for Bayesian priors
    p = np.linspace(0, 1, 1000)
    
    # Three different priors
    prior_1 = beta.pdf(p, 50, 50)  # narrow peak around 0.5
    prior_2 = beta.pdf(p, 2, 2)    # wider peak around 0.5
    prior_3 = np.ones_like(p)      # non-informative prior (uniform)

    # Observation of 8 heads and 2 tails
    posterior_1 = beta.pdf(p, 58, 52)
    posterior_2 = beta.pdf(p, 10, 4)
    posterior_3 = beta.pdf(p, 8 + 1, 2 + 1)  # MAP Estimation (non-informative)

    # Plot the priors and posteriors
    plt.figure(figsize=(10, 6))
    plt.plot(p, prior_1, label='Prior 1: Narrow (50,50)', linestyle='-', color='blue')
    plt.plot(p, prior_2, label='Prior 2: Wider (2,2)', linestyle='-', color='green')
    plt.plot(p, prior_3, label='Prior 3: Non-informative', linestyle='-', color='red')

    plt.plot(p, posterior_1, label='Posterior 1 (58, 52)', linestyle='--', color='blue')
    plt.plot(p, posterior_2, label='Posterior 2 (10, 4)', linestyle='--', color='green')
    plt.plot(p, posterior_3, label='Posterior 3 (8+1, 2+1)', linestyle='--', color='red')

    plt.title("2D Visualization of Bayesian Priors and Posteriors")
    plt.xlabel("Probability of Heads (p)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

# 3D Visualization of Priors and Posterior
from mpl_toolkits.mplot3d import Axes3D

def plot_3d():
    p = np.linspace(0, 1, 100)
    prior_1 = beta.pdf(p, 50, 50)
    prior_2 = beta.pdf(p, 2, 2)
    prior_3 = np.ones_like(p)

    # Observation: 8 heads, 2 tails
    posterior_1 = beta.pdf(p, 58, 52)
    posterior_2 = beta.pdf(p, 10, 4)
    posterior_3 = beta.pdf(p, 9, 3)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(p, prior_1, zs=0, zdir='y', label='Prior 1: Narrow (50,50)', color='blue')
    ax.plot(p, prior_2, zs=1, zdir='y', label='Prior 2: Wider (2,2)', color='green')
    ax.plot(p, prior_3, zs=2, zdir='y', label='Prior 3: Non-informative', color='red')

    ax.plot(p, posterior_1, zs=0, zdir='y', linestyle='--', color='blue')
    ax.plot(p, posterior_2, zs=1, zdir='y', linestyle='--', color='green')
    ax.plot(p, posterior_3, zs=2, zdir='y', linestyle='--', color='red')

    ax.set_title("3D Visualization of Priors and Posteriors")
    ax.set_xlabel("Probability of Heads (p)")
    ax.set_ylabel("Prior Types")
    ax.set_zlabel("Density")
    ax.legend()
    plt.show()

# Animation of updating belief with more coin tosses (using 8 heads, 2 tails scenario)
from matplotlib.animation import FuncAnimation

def animate_update():
    p = np.linspace(0, 1, 100)
    
    # Priors
    prior_1 = beta.pdf(p, 50, 50)
    prior_2 = beta.pdf(p, 2, 2)
    prior_3 = np.ones_like(p)
    
    # Observations
    posterior_1 = beta.pdf(p, 58, 52)  # After 8 heads, 2 tails
    posterior_2 = beta.pdf(p, 10, 4)   # After 8 heads, 2 tails
    posterior_3 = beta.pdf(p, 9, 3)    # After 8 heads, 2 tails

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 8)
    ax.set_title("Bayesian Update on Beliefs (8 heads, 2 tails)")
    ax.set_xlabel("Probability of Heads (p)")
    ax.set_ylabel("Density")

    line_1, = ax.plot([], [], label="Prior 1", color='blue')
    line_2, = ax.plot([], [], label="Prior 2", color='green')
    line_3, = ax.plot([], [], label="Prior 3", color='red')

    line_post_1, = ax.plot([], [], label="Posterior 1", linestyle='--', color='blue')
    line_post_2, = ax.plot([], [], label="Posterior 2", linestyle='--', color='green')
    line_post_3, = ax.plot([], [], label="Posterior 3", linestyle='--', color='red')

    def init():
        line_1.set_data([], [])
        line_2.set_data([], [])
        line_3.set_data([], [])
        line_post_1.set_data([], [])
        line_post_2.set_data([], [])
        line_post_3.set_data([], [])
        return line_1, line_2, line_3, line_post_1, line_post_2, line_post_3

    def animate(i):
        # Initially show priors and then posteriors
        if i < 50:
            line_1.set_data(p, prior_1)
            line_2.set_data(p, prior_2)
            line_3.set_data(p, prior_3)
        else:
            line_1.set_data(p, posterior_1)
            line_2.set_data(p, posterior_2)
            line_3.set_data(p, posterior_3)
        
        return line_1, line_2, line_3, line_post_1, line_post_2, line_post_3

    ani = FuncAnimation(fig, animate, frames=100, init_func=init, blit=True)
    ax.legend()
    plt.show()

# Call the functions to generate the plots
plot_2d()   # 2D plot visualization
plot_3d()   # 3D plot visualization
animate_update()  # Animation of the updating belief

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Data for the coin tosses
heads = 8
tails = 2

# Prior distributions
x = np.linspace(0, 1, 1000)

# Conservative Prior (Very Narrow around 0.5)
prior_conservative = beta(50, 50).pdf(x)

# Non-Conservative Prior (Wider around 0.5)
prior_non_conservative = beta(5, 5).pdf(x)

# Non-Informative Prior (Uniform distribution)
prior_non_informative = np.ones_like(x)

# Likelihood function - Likelihood of observing the data (8 heads, 2 tails)
# Posterior = Likelihood * Prior (normalized)
posterior_conservative = beta(50 + heads, 50 + tails).pdf(x)
posterior_non_conservative = beta(5 + heads, 5 + tails).pdf(x)
posterior_non_informative = beta(1 + heads, 1 + tails).pdf(x)

# Plotting the priors and posteriors
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plotting Prior and Posterior for Conservative
ax[0].plot(x, prior_conservative, label="Prior: Conservative", color="blue", lw=2)
ax[0].plot(x, posterior_conservative, label="Posterior: Conservative", color="darkblue", lw=2, linestyle='--')
ax[0].fill_between(x, prior_conservative, alpha=0.2, color="blue")
ax[0].fill_between(x, posterior_conservative, alpha=0.2, color="darkblue")
ax[0].set_title("Conservative Prior and Posterior", fontsize=14)
ax[0].set_xlabel("Probability of Heads (p)", fontsize=12)
ax[0].set_ylabel("Density", fontsize=12)
ax[0].legend()

# Plotting Prior and Posterior for Non-Conservative
ax[1].plot(x, prior_non_conservative, label="Prior: Non-Conservative", color="green", lw=2)
ax[1].plot(x, posterior_non_conservative, label="Posterior: Non-Conservative", color="darkgreen", lw=2, linestyle='--')
ax[1].fill_between(x, prior_non_conservative, alpha=0.2, color="green")
ax[1].fill_between(x, posterior_non_conservative, alpha=0.2, color="darkgreen")
ax[1].set_title("Non-Conservative Prior and Posterior", fontsize=14)
ax[1].set_xlabel("Probability of Heads (p)", fontsize=12)
ax[1].set_ylabel("Density", fontsize=12)
ax[1].legend()

# Display the plot
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Setting up the figure
plt.figure(figsize=(10, 8))

# Defining x range
x = np.linspace(0, 1, 1000)

# Conservative Prior (Narrow peak around 0.5)
conservative_prior = norm.pdf(x, 0.5, 0.1)

# Non-Conservative Prior (Wider peak)
non_conservative_prior = norm.pdf(x, 0.5, 0.3)

# Non-Informative Prior (Uniform distribution)
non_informative_prior = np.ones_like(x) / len(x)

# Posterior after observing 8 heads and 2 tails
# Assuming Beta distribution update with the observation of 8 heads and 2 tails
posterior_conservative = (x**8) * ((1-x)**2) * 10  # Beta(9,3)
posterior_non_conservative = (x**8) * ((1-x)**2) * 10  # Beta(9,3)
posterior_non_informative = (x**8) * ((1-x)**2) * 10  # Beta(9,3)

# Plotting
plt.plot(x, conservative_prior, label="Conservative Prior (Narrow Peak)", color='blue', linewidth=2)
plt.plot(x, non_conservative_prior, label="Non-Conservative Prior (Wider Peak)", color='green', linewidth=2)
plt.plot(x, non_informative_prior, label="Non-Informative Prior (Flat)", color='red', linewidth=2)

# Plotting Posterior Distributions
plt.plot(x, posterior_conservative, label="Posterior after 8 Heads, 2 Tails (Conservative)", linestyle='--', color='blue', linewidth=2)
plt.plot(x, posterior_non_conservative, label="Posterior after 8 Heads, 2 Tails (Non-Conservative)", linestyle='--', color='green', linewidth=2)
plt.plot(x, posterior_non_informative, label="Posterior after 8 Heads, 2 Tails (Non-Informative)", linestyle='--', color='red', linewidth=2)

# Adding labels and title
plt.title("Bayesian Priors and Posteriors with Coin Toss Data", fontsize=14)
plt.xlabel("Probability of Heads", fontsize=12)
plt.ylabel("Density", fontsize=12)

# Adding Text Annotations
plt.text(0.25, 6, 'Conservative Prior\nStrong belief\nNarrow Peak', fontsize=10, color='blue')
plt.text(0.25, 3, 'Non-Conservative Prior\nFlexible belief\nWider Peak', fontsize=10, color='green')
plt.text(0.25, 0.5, 'Non-Informative Prior\nNo belief\nFlat', fontsize=10, color='red')

plt.text(0.7, 2, 'Posterior\nConservative: Minimal change', fontsize=10, color='blue')
plt.text(0.7, 1.5, 'Posterior\nNon-Conservative: Significant shift', fontsize=10, color='green')
plt.text(0.7, 0.7, 'Posterior\nNon-Informative: Strong influence of data', fontsize=10, color='red')

# Adding Legend
plt.legend(loc='upper left', fontsize=10)

# Display plot
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Define prior and likelihood values for visualization
prior_fair = 0.75  # Prior probability of fair coin
prior_biased = 1 - prior_fair  # Prior probability of biased coin

likelihood_fair = 0.5  # Probability of heads given fair coin
likelihood_biased = 0.8  # Probability of heads given biased coin

# Compute marginal probability of heads
p_heads = (likelihood_fair * prior_fair) + (likelihood_biased * prior_biased)

# Compute posterior probabilities
posterior_fair = (likelihood_fair * prior_fair) / p_heads
posterior_biased = (likelihood_biased * prior_biased) / p_heads

# 2D Plot: Prior vs. Posterior
plt.figure(figsize=(8, 5))
labels = ['Fair Coin', 'Biased Coin']
priors = [prior_fair, prior_biased]
posteriors = [posterior_fair, posterior_biased]

x = np.arange(len(labels))

plt.bar(x - 0.2, priors, 0.4, label='Prior', color='blue', alpha=0.6)
plt.bar(x + 0.2, posteriors, 0.4, label='Posterior', color='red', alpha=0.6)

plt.xticks(x, labels)
plt.ylabel('Probability')
plt.title('Bayesian Updating: Prior vs. Posterior')
plt.legend()
plt.show()

# 3D Plot: Posterior Probability Surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Generate data
prior_range = np.linspace(0, 1, 50)
likelihood_range = np.linspace(0, 1, 50)
P, L = np.meshgrid(prior_range, likelihood_range)
posterior_surface = (L * P) / ((L * P) + ((1 - L) * (1 - P)))

# Plot the surface
ax.plot_surface(P, L, posterior_surface, cmap='viridis', edgecolor='none')
ax.set_xlabel('Prior Probability')
ax.set_ylabel('Likelihood')
ax.set_zlabel('Posterior Probability')
ax.set_title('Bayesian Posterior Probability Surface')

plt.show()

# Animation: Bayesian Updating Over Multiple Coin Flips
fig, ax = plt.subplots(figsize=(8, 5))
x_vals = np.arange(1, 11)  # 10 coin flips
prior_fair_vals = [0.75]

def update(frame):
    global prior_fair
    
    outcome = np.random.choice([0, 1], p=[1-likelihood_fair, likelihood_fair])  # Simulate flip
    p_heads_dynamic = (likelihood_fair * prior_fair) + (likelihood_biased * (1 - prior_fair))
    posterior_fair = (likelihood_fair * prior_fair) / p_heads_dynamic
    
    prior_fair_vals.append(posterior_fair)
    ax.clear()
    ax.plot(range(len(prior_fair_vals)), prior_fair_vals, marker='o', linestyle='-', color='red')
    ax.set_title('Bayesian Updating Over Coin Flips')
    ax.set_xlabel('Flip Number')
    ax.set_ylabel('Probability Coin is Fair')
    ax.set_ylim(0, 1)

ani = animation.FuncAnimation(fig, update, frames=10, repeat=False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.animation import FuncAnimation

# Function to calculate Beta distribution posterior given prior parameters and data
def bayesian_update(alpha, beta, heads, tails):
    # Update parameters of Beta distribution
    return alpha + heads, beta + tails

# Set initial prior parameters (uniform prior, Beta(1, 1))
alpha_prior, beta_prior = 1, 1

# Data: 8 heads and 2 tails (from the first 10 flips)
heads_1, tails_1 = 8, 2

# Update posterior after first data (10 flips)
alpha_post_1, beta_post_1 = bayesian_update(alpha_prior, beta_prior, heads_1, tails_1)

# Data: 6 heads and 4 tails (from the second 10 flips)
heads_2, tails_2 = 6, 4

# Update posterior after second data (additional 10 flips)
alpha_post_2, beta_post_2 = bayesian_update(alpha_post_1, beta_post_1, heads_2, tails_2)

# Define theta values for plotting the posterior distributions
theta = np.linspace(0, 1, 1000)

# 2D Plot: Plot the prior and posterior distributions
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta, beta.pdf(theta, alpha_prior, beta_prior), label='Prior (Beta(1, 1))', linestyle='--', color='gray')
ax.plot(theta, beta.pdf(theta, alpha_post_1, beta_post_1), label='Posterior after 8 heads, 2 tails (Beta(9, 3))', color='blue')
ax.plot(theta, beta.pdf(theta, alpha_post_2, beta_post_2), label='Posterior after 6 heads, 4 tails (Beta(15, 7))', color='red')
ax.set_title('Posterior Distributions of Theta (Coin Flip)')
ax.set_xlabel('Theta (Probability of heads)')
ax.set_ylabel('Density')
ax.legend()
plt.grid(True)
plt.show()

# 3D Plot: Visualize the posterior distribution as a 3D plot of the updating process
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create a mesh grid of theta values
theta_3d = np.outer(theta, np.ones(100))

# Calculate the Beta distribution for the updated parameters
posterior_1_3d = beta.pdf(theta_3d, alpha_post_1, beta_post_1)
posterior_2_3d = beta.pdf(theta_3d, alpha_post_2, beta_post_2)

# Plot the first posterior distribution (after 10 flips)
ax.plot_surface(theta_3d, np.zeros_like(theta_3d), posterior_1_3d, color='blue', alpha=0.6)
# Plot the second posterior distribution (after 20 flips)
ax.plot_surface(theta_3d, np.ones_like(theta_3d), posterior_2_3d, color='red', alpha=0.6)

ax.set_title('3D Visualization of Posterior Distributions')
ax.set_xlabel('Theta (Probability of heads)')
ax.set_ylabel('Data Index')
ax.set_zlabel('Density')
plt.show()

# Animation: Show how the posterior evolves as more data is introduced
fig, ax = plt.subplots(figsize=(10, 6))

# Function to update the plot for each frame in the animation
def update(frame):
    ax.clear()
    if frame == 0:
        ax.plot(theta, beta.pdf(theta, alpha_prior, beta_prior), label='Prior (Beta(1, 1))', linestyle='--', color='gray')
    elif frame == 1:
        ax.plot(theta, beta.pdf(theta, alpha_post_1, beta_post_1), label='Posterior after 8 heads, 2 tails (Beta(9, 3))', color='blue')
    elif frame == 2:
        ax.plot(theta, beta.pdf(theta, alpha_post_2, beta_post_2), label='Posterior after 6 heads, 4 tails (Beta(15, 7))', color='red')
    
    ax.set_title('Updating Posterior Distribution')
    ax.set_xlabel('Theta (Probability of heads)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

# Create the animation
ani = FuncAnimation(fig, update, frames=3, interval=1000)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Define prior and likelihood values for visualization
prior_fair = 0.75  # Prior probability of fair coin
prior_biased = 1 - prior_fair  # Prior probability of biased coin

likelihood_fair = 0.5  # Probability of heads given fair coin
likelihood_biased = 0.8  # Probability of heads given biased coin

# Compute marginal probability of heads
p_heads = (likelihood_fair * prior_fair) + (likelihood_biased * prior_biased)

# Compute posterior probabilities
posterior_fair = (likelihood_fair * prior_fair) / p_heads
posterior_biased = (likelihood_biased * prior_biased) / p_heads

# 2D Plot: Prior vs. Posterior
plt.figure(figsize=(8, 5))
labels = ['Fair Coin', 'Biased Coin']
priors = [prior_fair, prior_biased]
posteriors = [posterior_fair, posterior_biased]

x = np.arange(len(labels))

plt.bar(x - 0.2, priors, 0.4, label='Prior', color='blue', alpha=0.6)
plt.bar(x + 0.2, posteriors, 0.4, label='Posterior', color='red', alpha=0.6)

plt.xticks(x, labels)
plt.ylabel('Probability')
plt.title('Bayesian Updating: Prior vs. Posterior')
plt.legend()
plt.show()

# 3D Plot: Posterior Probability Surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Generate data
prior_range = np.linspace(0, 1, 50)
likelihood_range = np.linspace(0, 1, 50)
P, L = np.meshgrid(prior_range, likelihood_range)
posterior_surface = (L * P) / ((L * P) + ((1 - L) * (1 - P)))

# Plot the surface
ax.plot_surface(P, L, posterior_surface, cmap='viridis', edgecolor='none')
ax.set_xlabel('Prior Probability')
ax.set_ylabel('Likelihood')
ax.set_zlabel('Posterior Probability')
ax.set_title('Bayesian Posterior Probability Surface')

plt.show()

# Animation: Bayesian Updating Over Multiple Coin Flips
fig, ax = plt.subplots(figsize=(8, 5))
x_vals = np.arange(1, 11)  # 10 coin flips
prior_fair_vals = [0.75]

def update(frame):
    global prior_fair
    
    outcome = np.random.choice([0, 1], p=[1-likelihood_fair, likelihood_fair])  # Simulate flip
    p_heads_dynamic = (likelihood_fair * prior_fair) + (likelihood_biased * (1 - prior_fair))
    posterior_fair = (likelihood_fair * prior_fair) / p_heads_dynamic
    
    prior_fair_vals.append(posterior_fair)
    ax.clear()
    ax.plot(range(len(prior_fair_vals)), prior_fair_vals, marker='o', linestyle='-', color='red')
    ax.set_title('Bayesian Updating Over Coin Flips')
    ax.set_xlabel('Flip Number')
    ax.set_ylabel('Probability Coin is Fair')
    ax.set_ylim(0, 1)

ani = animation.FuncAnimation(fig, update, frames=10, repeat=False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(42)
X = np.linspace(-5, 5, 100)
y = 3 * X**2 + 2 * X + np.random.normal(0, 3, size=X.shape)

# Define three models
def model_1(x): return 3 * x**2 + 2 * x  # True model
def model_2(x): return 2.5 * x**2 + 1.5 * x  # Underfitting model (simpler)
def model_3(x): return 3.5 * x**2 + 2.5 * x + 0.5 * x**3  # Overfitting model (complex)

# Plot data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X, y=y, color='black', label='Data')

# Plot models
plt.plot(X, model_1(X), label="MLE (Best Fit Model)", color='blue')
plt.plot(X, model_2(X), label="MAP (Regularized Simpler Model)", color='green')
plt.plot(X, model_3(X), label="Overfitting Model", color='red', linestyle='dashed')

# Labels and Legend
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("MLE vs MAP vs Regularization in Model Selection")
plt.legend()
plt.grid()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(42)
X = np.linspace(-5, 5, 100)
y = 3 * X**2 + 2 * X + np.random.normal(0, 3, size=X.shape)

# Define three models
def model_1(x): return 3 * x**2 + 2 * x  # True model
def model_2(x): return 2.5 * x**2 + 1.5 * x  # Underfitting model (simpler)
def model_3(x): return 3.5 * x**2 + 2.5 * x + 0.5 * x**3  # Overfitting model (complex)

# Plot data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X, y=y, color='black', label='Data')

# Plot models
plt.plot(X, model_1(X), label="MLE (Best Fit Model)", color='blue')
plt.plot(X, model_2(X), label="MAP (Regularized Simpler Model)", color='green')
plt.plot(X, model_3(X), label="Overfitting Model", color='red', linestyle='dashed')

# Labels and Legend
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("MLE vs MAP vs Regularization in Model Selection")
plt.legend()
plt.grid()
plt.show()


from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

# Generate parameter space
a_values = np.linspace(-5, 5, 50)
b_values = np.linspace(-5, 5, 50)
A, B = np.meshgrid(a_values, b_values)

# Define likelihood function P(Data | Model)
likelihood = np.exp(-0.5 * (A**2 + B**2))  # Gaussian assumption for simplicity

# Define prior P(Model) as Gaussian
prior = norm.pdf(A, 0, 1) * norm.pdf(B, 0, 1)

# Define posterior P(Model | Data) using Bayes' Theorem
posterior = likelihood * prior

# Plot
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, B, posterior, cmap='viridis')

ax.set_xlabel("Parameter a")
ax.set_ylabel("Parameter b")
ax.set_zlabel("Posterior Probability")
ax.set_title("3D Visualization of Model Probability")

plt.show()
