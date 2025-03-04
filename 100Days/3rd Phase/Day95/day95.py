import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

# Generate sample data
np.random.seed(42)
x_normal = np.random.normal(loc=50, scale=10, size=300)
x_anomaly = np.random.normal(loc=80, scale=5, size=10)
data = np.concatenate([x_normal, x_anomaly])

# Define threshold for anomaly detection
threshold = 70

# 1. Plot Anomaly Detection vs Supervised Learning
plt.figure(figsize=(10, 5))
sns.histplot(data, bins=30, kde=True, color='blue', label='Normal Data')
sns.histplot(x_anomaly, bins=5, kde=True, color='red', label='Anomalies')
plt.axvline(threshold, color='black', linestyle='dashed', label='Threshold')
plt.title("Anomaly Detection vs Supervised Learning")
plt.legend()
plt.show()

# 2. Finding Anomalies in the Dataset
plt.figure(figsize=(10, 5))
plt.scatter(range(len(data)), data, c=['red' if x > threshold else 'blue' for x in data])
plt.axhline(threshold, color='black', linestyle='dashed', label='Threshold')
plt.title("Finding Anomalies in Dataset")
plt.legend()
plt.show()

# 3. Conditional Probability Visualization
x_vals = np.linspace(20, 100, 1000)
p_x = norm.pdf(x_vals, loc=50, scale=10)
p_x_given_anomaly = norm.pdf(x_vals, loc=80, scale=5)
plt.figure(figsize=(10, 5))
plt.plot(x_vals, p_x, label='P(x) Normal', color='blue')
plt.plot(x_vals, p_x_given_anomaly, label='P(x|Anomaly)', color='red')
plt.title("Conditional Probability Visualization")
plt.legend()
plt.show()

# 4. Normal Distribution on Skewed Values
skewed_data = np.random.exponential(scale=10, size=500)
sns.histplot(skewed_data, bins=30, kde=True, color='purple')
plt.title("Normal Distribution on Skewed Values")
plt.show()

# 5. Rejecting/Accepting Features Based on Threshold
accepted = data[data < threshold]
rejected = data[data >= threshold]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(accepted, np.zeros(len(accepted)), np.zeros(len(accepted)), color='blue', label='Accepted')
ax.scatter(rejected, np.ones(len(rejected)), np.ones(len(rejected)), color='red', label='Rejected')
ax.set_xlabel('X Feature')
ax.set_ylabel('Decision')
ax.set_zlabel('Anomaly')
ax.legend()
plt.title("Rejecting and Accepting X Feature")
plt.show()
