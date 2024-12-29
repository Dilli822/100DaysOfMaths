import numpy as np
import matplotlib.pyplot as plt

# Define matrix A and some vectors
A = np.array([[1, 2], [3, 4]])
vectors = np.array([[1, 0], [0, 1], [1, 1], [2, -1]])

# Perform the linear transformation
transformed_vectors = A.dot(vectors.T).T

# Plot original vectors and transformed vectors
plt.figure(figsize=(8, 6))

# Plot original vectors (blue)
for v in vectors:
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue')

# Plot transformed vectors (red)
for v in transformed_vectors:
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='red')

plt.xlim(-4, 10)
plt.ylim(-4, 10)
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("2D Linear Transformation: Trivial Solution")
plt.grid(True)
plt.show()

# Define a singular matrix A (linearly dependent columns)
A_singular = np.array([[1, 1], [2, 2]])

# Perform the linear transformation for singular matrix
transformed_vectors_singular = A_singular.dot(vectors.T).T

# Plot original vectors and transformed vectors for singular matrix
plt.figure(figsize=(8, 6))

# Plot original vectors (blue)
for v in vectors:
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue')

# Plot transformed vectors (red)
for v in transformed_vectors_singular:
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='red')

plt.xlim(-4, 10)
plt.ylim(-4, 10)
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("2D Linear Transformation: Non-trivial Solutions")
plt.grid(True)
plt.show()

# Define vectors to be transformed
vectors = np.array([[1, 0], [0, 1], [1, 1], [2, -1]])

# One-to-One Linear Transformation (Identity Matrix)
A_one_to_one = np.array([[1, 0], [0, 1]])
transformed_vectors_one_to_one = A_one_to_one.dot(vectors.T).T

# Not One-to-One Linear Transformation (Singular Matrix)
A_not_one_to_one = np.array([[1, 1], [0, 0]])
transformed_vectors_not_one_to_one = A_not_one_to_one.dot(vectors.T).T

# Plotting for One-to-One Transformation
plt.figure(figsize=(8, 6))
for v in vectors:
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue')

for v in transformed_vectors_one_to_one:
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='red')

plt.xlim(-2, 4)
plt.ylim(-2, 4)
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("One-to-One Linear Transformation")
plt.grid(True)
plt.show()

# Plotting for Not One-to-One Transformation
plt.figure(figsize=(8, 6))
for v in vectors:
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue')

for v in transformed_vectors_not_one_to_one:
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='red')

plt.xlim(-2, 4)
plt.ylim(-2, 4)
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Not One-to-One Linear Transformation")
plt.grid(True)
plt.show()
