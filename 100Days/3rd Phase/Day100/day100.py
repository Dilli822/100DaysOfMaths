

# import gymnasium as gym
# import numpy as np
# import matplotlib.pyplot as plt

# # Initialize the Lunar Lander environment
# env = gym.make("LunarLander-v3", render_mode="human")

# # Q-learning parameters
# NUM_EPISODES = 100  # Start with fewer episodes for debugging
# LEARNING_RATE = 0.1
# DISCOUNT_FACTOR = 0.99
# EPSILON = 1.0
# EPSILON_DECAY = 0.995
# MIN_EPSILON = 0.01

# # Discretize the state space (simplified)
# NUM_BINS = 10  # Reduce the number of bins for debugging
# STATE_BINS = [
#     np.linspace(-1, 1, NUM_BINS),  # x position
#     np.linspace(-1, 1, NUM_BINS),  # y position
#     np.linspace(-1, 1, NUM_BINS),  # x velocity
#     np.linspace(-1, 1, NUM_BINS),  # y velocity
#     np.linspace(-1, 1, NUM_BINS),  # angle
#     np.linspace(-1, 1, NUM_BINS),  # angular velocity
#     np.linspace(0, 1, NUM_BINS),   # left leg contact
#     np.linspace(0, 1, NUM_BINS)    # right leg contact
# ]

# # Initialize Q-table
# NUM_ACTIONS = env.action_space.n
# Q_TABLE = np.zeros((NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, NUM_ACTIONS))

# # Helper function to discretize the state
# def discretize_state(state):
#     indices = []
#     for i in range(len(state)):
#         indices.append(np.digitize(state[i], STATE_BINS[i]) - 1)
#     # Ensure indices are within bounds
#     indices = [min(max(idx, 0), NUM_BINS - 1) for idx in indices]
#     return tuple(indices)

# # Training loop
# rewards = []
# for episode in range(NUM_EPISODES):
#     state, _ = env.reset()
#     state = discretize_state(state)
#     total_reward = 0
#     done = False

#     while not done:
#         # Epsilon-greedy policy
#         if np.random.random() < EPSILON:
#             action = env.action_space.sample()  # Explore
#         else:
#             action = np.argmax(Q_TABLE[state])  # Exploit

#         # Take action and observe the result
#         next_state, reward, done, _, _ = env.step(action)
#         next_state = discretize_state(next_state)
#         total_reward += reward

#         # Update Q-table
#         Q_TABLE[state][action] += LEARNING_RATE * (
#             reward + DISCOUNT_FACTOR * np.max(Q_TABLE[next_state]) - Q_TABLE[state][action]
#         )

#         # Move to the next state
#         state = next_state

#     # Decay epsilon
#     EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

#     # Store the total reward for this episode
#     rewards.append(total_reward)

#     # Print progress
#     print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {EPSILON}")

# # Plot the rewards over episodes
# plt.plot(rewards)
# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.title("Q-Learning Training Progress")
# plt.show()

# # Close the environment
# env.close()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style='whitegrid')

# 1. Visualizing Mini-Batch Gradient Descent vs. Batch Gradient Descent
def plot_mini_batch_vs_batch():
    np.random.seed(42)
    x = np.linspace(-10, 10, 100)
    cost = x**2  # Example cost function (convex)
    
    steps_batch = [(-8, 64), (-5, 25), (-3, 9), (-1, 1), (0, 0)]  # Large steps
    steps_mini = [(-8, 64), (-6, 36), (-5, 25), (-4, 16), (-3, 9), (-2, 4), (-1, 1), (0, 0)]  # Noisy but converging
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, cost, label='Cost Function J(w)', linewidth=2)
    plt.scatter(*zip(*steps_batch), color='red', label='Batch Gradient Descent', marker='o')
    plt.scatter(*zip(*steps_mini), color='blue', label='Mini-Batch Gradient Descent', marker='x')
    
    plt.xlabel('w')
    plt.ylabel('J(w)')
    plt.title('Mini-Batch vs. Batch Gradient Descent')
    plt.legend()
    plt.show()

# 2. Visualizing Q-Learning with Experience Replay (Random Sampling)
def plot_experience_replay():
    buffer_size = 100  # Simulated replay buffer size
    mini_batch_size = 10
    buffer = np.random.rand(buffer_size)  # Random experience values
    sample_indices = np.random.choice(buffer_size, mini_batch_size, replace=False)
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(buffer_size), buffer, alpha=0.6, label='Replay Buffer')
    plt.scatter(sample_indices, buffer[sample_indices], color='red', label='Sampled Mini-Batch')
    plt.xlabel('Experience Index')
    plt.ylabel('Experience Value')
    plt.title('Experience Replay: Mini-Batch Sampling')
    plt.legend()
    plt.show()

# 3. Visualizing Soft Updates (Weight Updates)
def plot_soft_updates():
    steps = np.arange(10)
    w_old = 1  # Initial weight
    w_new = 10  # Target weight
    tau = 0.1  # Soft update factor
    
    w_values = [w_old]
    for _ in steps:
        w_old = tau * w_new + (1 - tau) * w_old  # Soft update equation
        w_values.append(w_old)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(w_values)), w_values, marker='o', linestyle='-', label='Soft Updates')
    plt.axhline(y=w_new, color='r', linestyle='--', label='Target Weight')
    plt.xlabel('Update Step')
    plt.ylabel('Weight Value')
    plt.title('Soft Updates for Stabilizing Q-Network')
    plt.legend()
    plt.show()

# Execute all plots
plot_mini_batch_vs_batch()
plot_experience_replay()
plot_soft_updates()


# import gymnasium as gym
# import numpy as np
# import matplotlib.pyplot as plt

# # Initialize the Lunar Lander environment
# env = gym.make("LunarLander-v3", render_mode="human")

# # Q-learning parameters
# NUM_EPISODES = 100  # Number of episodes to train
# LEARNING_RATE = 0.1
# DISCOUNT_FACTOR = 0.99
# EPSILON = 1.0
# EPSILON_DECAY = 0.995
# MIN_EPSILON = 0.01

# # Discretize the state space (simplified)
# NUM_BINS = 10  # Reduce the number of bins for debugging
# STATE_BINS = [
#     np.linspace(-1, 1, NUM_BINS),  # x position
#     np.linspace(-1, 1, NUM_BINS),  # y position
#     np.linspace(-1, 1, NUM_BINS),  # x velocity
#     np.linspace(-1, 1, NUM_BINS),  # y velocity
#     np.linspace(-1, 1, NUM_BINS),  # angle
#     np.linspace(-1, 1, NUM_BINS),  # angular velocity
#     np.linspace(0, 1, NUM_BINS),   # left leg contact
#     np.linspace(0, 1, NUM_BINS)    # right leg contact
# ]

# # Initialize Q-table
# NUM_ACTIONS = env.action_space.n
# Q_TABLE = np.zeros((NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, NUM_ACTIONS))

# # Helper function to discretize the state
# def discretize_state(state):
#     indices = []
#     for i in range(len(state)):
#         indices.append(np.digitize(state[i], STATE_BINS[i]) - 1)
#     # Ensure indices are within bounds
#     indices = [min(max(idx, 0), NUM_BINS - 1) for idx in indices]
#     return tuple(indices)

# # Open a file to save logs
# with open("lunarland.txt", "w") as log_file:
#     # Training loop
#     rewards = []
#     successful_landings = []

#     for episode in range(NUM_EPISODES):
#         state, _ = env.reset()
#         state = discretize_state(state)
#         total_reward = 0
#         done = False

#         while not done:
#             # Epsilon-greedy policy
#             if np.random.random() < EPSILON:
#                 action = env.action_space.sample()  # Explore
#             else:
#                 action = np.argmax(Q_TABLE[state])  # Exploit

#             # Take action and observe the result
#             next_state, reward, done, _, _ = env.step(action)
#             next_state = discretize_state(next_state)
#             total_reward += reward

#             # Update Q-table
#             Q_TABLE[state][action] += LEARNING_RATE * (
#                 reward + DISCOUNT_FACTOR * np.max(Q_TABLE[next_state]) - Q_TABLE[state][action]
#             )

#             # Move to the next state
#             state = next_state

#         # Decay epsilon
#         EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

#         # Store the total reward for this episode
#         rewards.append(total_reward)

#         # Check for successful landing
#         if total_reward > 200:  # Threshold for successful landing
#             successful_landings.append(episode)
#             log_file.write(f"Successful Landing! Episode: {episode}, Total Reward: {total_reward}\n")
#         else:
#             log_file.write(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {EPSILON}\n")

# # Plot the rewards over episodes
# plt.plot(rewards)
# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.title("Q-Learning Training Progress")
# plt.show()

# # Print successful landings
# if successful_landings:
#     print(f"Successful landings occurred in episodes: {successful_landings}")
# else:
#     print("No successful landings occurred.")

# # Close the environment
# env.close()