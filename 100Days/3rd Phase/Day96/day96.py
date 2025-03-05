import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate synthetic data
np.random.seed(0)
n_movies = 5  # Number of movies
n_users = 4   # Number of users
n_features = 2  # Number of latent features

# Movie features (x^i)
X = np.random.rand(n_movies, n_features)

# User preferences (w^j)
W = np.random.rand(n_users, n_features)

# Step 2: Compute predicted ratings
Y_pred = X @ W.T  # Dot product

# Step 3: Visualize movie features in 2D
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], color='red', label='Movies')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("2D Movie Features")
plt.legend()
plt.grid()
plt.show()

# Step 4: Extend to 3D visualization
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], np.mean(Y_pred, axis=1), c='red', label='Movies')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Avg Predicted Rating")
ax.set_title("3D Visualization of Movie Features & Ratings")
ax.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import KMeans

# Generate random clustered data
np.random.seed(42)
cluster1 = np.random.randn(10, 2) + [2, 2]
cluster2 = np.random.randn(10, 2) + [6, 6]
points = np.vstack([cluster1, cluster2])

# Set up the figure
fig, ax = plt.subplots()
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
sc = ax.scatter(points[:, 0], points[:, 1], s=100, c='gray')

# KMeans clustering animation
kmeans = KMeans(n_clusters=2, init='random', n_init=1, max_iter=1)
centroids = kmeans.fit(points).cluster_centers_
centroid_sc = ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='x')

# Animation update function
def update(frame):
    global centroids
    kmeans = KMeans(n_clusters=2, init=centroids, n_init=1, max_iter=1)
    labels = kmeans.fit_predict(points)
    centroids = kmeans.cluster_centers_
    
    sc.set_color(['blue' if l == 0 else 'green' for l in labels])
    centroid_sc.set_offsets(centroids)
    return sc, centroid_sc

ani = animation.FuncAnimation(fig, update, frames=10, interval=500, repeat=False)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define user feature vectors (weights)
users = {"Alice": np.array([5, 0]), "Bob": np.array([5, 0]),
         "Carol": np.array([0, 5]), "Dave": np.array([0, 5])}

# Define movie feature vectors (unknown, we estimate)
movies = {"Love at Last": np.array([1, 0]),
          "Nonstop Car Chases": np.array([0, 1])}

# 2D Plot - Romance vs Action Preferences
fig, ax = plt.subplots()
for user, vec in users.items():
    ax.scatter(vec[0], vec[1], label=user)
ax.set_xlabel("Romance Feature")
ax.set_ylabel("Action Feature")
ax.legend()
ax.set_title("User Preferences in 2D Feature Space")
plt.grid()
plt.show()

# 3D Plot - Adding Ratings Dimension
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ratings = [5, 5, 0, 0]  # Example ratings for "Love at Last"

for (user, vec), rating in zip(users.items(), ratings):
    ax.scatter(vec[0], vec[1], rating, label=user)

ax.set_xlabel("Romance Feature")
ax.set_ylabel("Action Feature")
ax.set_zlabel("Rating")
ax.legend()
ax.set_title("User Ratings in 3D Feature Space")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulated Cost Function Values
iterations = 100
J_values = np.logspace(2, -1, iterations)  # Simulating cost decreasing over iterations

# Create figure
fig, ax = plt.subplots()
ax.set_xlim(0, iterations)
ax.set_ylim(0, max(J_values))
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost J(X, Θ)")
ax.set_title("Collaborative Filtering Cost Minimization")

line, = ax.plot([], [], 'r-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    xdata = np.arange(frame)
    ydata = J_values[:frame]
    line.set_data(xdata, ydata)
    return line,

ani = animation.FuncAnimation(fig, update, frames=iterations, init_func=init, blit=True, interval=50)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define feature vectors for movies (x) and users (w)
movies = {
    "Love at last": np.array([1, 0]),      # Strong romance, no action
    "Romance forever": np.array([1, 0]),  # Strong romance, no action
    "Cute puppies of love": np.array([0.8, 0.2]),
    "Nonstop car chases": np.array([0, 1]),  # Strong action, no romance
    "Swords vs. karate": np.array([0, 1])   # Strong action, no romance
}

users = {
    "Alice": np.array([5, 0]),  # Prefers romance
    "Bob": np.array([5, 0]),    # Prefers romance
    "Carol": np.array([0, 5]),  # Prefers action
    "Dave": np.array([0, 5])    # Prefers action
}

# Plot settings
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 6)
ax.set_xlabel("Romance Feature (x1)")
ax.set_ylabel("Action Feature (x2)")
ax.set_title("Collaborative Filtering Visualization")

# Plot movies as points
for movie, feature in movies.items():
    ax.scatter(feature[0], feature[1], label=movie, marker='o', s=100)

# Plot user preferences as arrows (vectors)
for user, preference in users.items():
    ax.quiver(0, 0, preference[0]/5, preference[1]/5, angles='xy', scale_units='xy', scale=1.2, label=user)

ax.legend()
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define feature vectors for movies (x) and users (w)
movies = {
    "Love at last": np.array([1, 0]),      
    "Romance forever": np.array([1, 0]),  
    "Cute puppies of love": np.array([0.8, 0.2]),
    "Nonstop car chases": np.array([0, 1]),  
    "Swords vs. karate": np.array([0, 1])   
}

users = {
    "Alice": np.array([5, 0]),  
    "Bob": np.array([5, 0]),    
    "Carol": np.array([0, 5]),  
    "Dave": np.array([0, 5])    
}

# Compute predicted ratings (dot product w · x)
ratings = {}
for movie, x in movies.items():
    ratings[movie] = {user: np.dot(w, x) for user, w in users.items()}

# Compute the average of all predicted ratings
all_ratings = [rating for movie_ratings in ratings.values() for rating in movie_ratings.values()]
avg_rating = np.mean(all_ratings) if all_ratings else 0  # Ensure no division by zero

# 3D Plot setup
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("Romance Feature (x1)")
ax.set_ylabel("Action Feature (x2)")
ax.set_zlabel("Predicted Rating")
ax.set_title("3D Visualization of Collaborative Filtering")

# Plot movies as points in 3D
for movie, x in movies.items():
    movie_avg_rating = np.mean(list(ratings[movie].values()))  # Movie-specific average rating
    ax.scatter(x[0], x[1], movie_avg_rating, label=movie, s=100)

# Plot user preference vectors in 3D
for user, w in users.items():
    ax.quiver(0, 0, 0, w[0]/5, w[1]/5, avg_rating, length=1, label=user)

ax.legend()
plt.show()
