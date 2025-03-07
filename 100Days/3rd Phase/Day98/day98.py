import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# 1. Efficient Recommender Systems: Large Catalog Selection
fig, ax = plt.subplots(figsize=(8, 6))
catalog_size = 1000
selected_items = np.random.choice(catalog_size, size=20, replace=False)
ax.scatter(range(catalog_size), np.zeros(catalog_size), alpha=0.2, label="All Items")
ax.scatter(selected_items, np.zeros(len(selected_items)), color='red', label="Selected Items")
ax.set_title("Efficient Recommender: Selecting Items from a Large Catalog")
ax.legend()
plt.show()

# 2. Retrieval Step: Dot Product Similarity in 2D & 3D
np.random.seed(42)
user_vector = np.random.rand(2)
item_vectors = np.random.rand(10, 2)
dot_products = np.dot(item_vectors, user_vector)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(dot_products.reshape(1, -1), annot=True, cmap="coolwarm", cbar=True)
ax.set_title("Dot Product Similarity for Item Retrieval")
plt.show()

fig_3d = go.Figure()
fig_3d.add_trace(go.Scatter3d(x=item_vectors[:, 0], y=item_vectors[:, 1], z=dot_products, 
                              mode='markers', marker=dict(size=5, color=dot_products)))
fig_3d.update_layout(title="3D Visualization of Dot Product Similarity")
fig_3d.show()

# 3. Ranking Step: Ranking Items by User Preferences
sorted_items = np.argsort(-dot_products)
plt.figure(figsize=(8, 6))
plt.bar(range(1, 11), dot_products[sorted_items], color='royalblue')
plt.xlabel("Item Rank")
plt.ylabel("Predicted Score")
plt.title("Ranking Items Based on User Preferences")
plt.show()

# 4. Optimization Trade-offs: Retrieval Size vs. Accuracy
retrieval_sizes = np.arange(50, 550, 50)
accuracies = 1 - np.exp(-retrieval_sizes / 200)
plt.figure(figsize=(8, 6))
plt.plot(retrieval_sizes, accuracies, marker='o', linestyle='-', color='green')
plt.xlabel("Retrieval Size")
plt.ylabel("Recommendation Accuracy")
plt.title("Trade-off: Retrieval Size vs. Accuracy")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate synthetic item embeddings (high-dimensional vectors)
np.random.seed(42)
item_embeddings = np.random.rand(100, 10)  # 100 items, 10 features

# Apply PCA to reduce to 2D
pca_2d = PCA(n_components=2)
item_2d = pca_2d.fit_transform(item_embeddings)

# Apply PCA to reduce to 3D
pca_3d = PCA(n_components=3)
item_3d = pca_3d.fit_transform(item_embeddings)

# 2D Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(item_2d[:, 0], item_2d[:, 1], c='blue', alpha=0.6)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("2D Visualization of Item Embeddings")
plt.grid()
plt.show()

# 3D Scatter Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(item_3d[:, 0], item_3d[:, 1], item_3d[:, 2], c='red', alpha=0.6)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
ax.set_title("3D Visualization of Item Embeddings")
plt.show()


import seaborn as sns
from scipy.spatial.distance import pdist, squareform

# Compute similarity (cosine similarity)
similarity_matrix = 1 - squareform(pdist(item_embeddings, metric='cosine'))

# 2D Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, cmap="coolwarm", annot=False)
plt.title("Item Similarity Heatmap")
plt.xlabel("Item Index")
plt.ylabel("Item Index")
plt.show()

# 3D Plot of Item Similarity
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

x, y = np.meshgrid(np.arange(len(similarity_matrix)), np.arange(len(similarity_matrix)))
ax.plot_surface(x, y, similarity_matrix, cmap="viridis")

ax.set_xlabel("Item Index")
ax.set_ylabel("Item Index")
ax.set_zlabel("Similarity Score")
ax.set_title("3D Visualization of Item Similarity")
plt.show()


import plotly.express as px
import pandas as pd

# Generate synthetic relevance scores
users = np.random.rand(10, 10)  # 10 users, 10 movies
user_idx = 0
relevance_scores = users[user_idx]  # Assume one user

# 2D Bar Chart
plt.figure(figsize=(8, 5))
plt.bar(range(10), relevance_scores, color='purple', alpha=0.7)
plt.xlabel("Item Index")
plt.ylabel("Predicted Score")
plt.title("Predicted Relevance Scores for User")
plt.xticks(range(10))
plt.show()

# 3D Bar Chart using Plotly
df = pd.DataFrame({"Item Index": np.arange(10), "Relevance Score": relevance_scores})
fig = px.bar(df, x="Item Index", y="Relevance Score", title="3D Ranking Visualization", color="Relevance Score")
fig.show()


# Generate synthetic retrieval vs accuracy data
retrieval_sizes = np.arange(10, 510, 50)
accuracy = np.log1p(retrieval_sizes)  # Logarithmic accuracy gain

# 2D Line Plot
plt.figure(figsize=(8, 5))
plt.plot(retrieval_sizes, accuracy, marker='o', color='green')
plt.xlabel("Retrieval Size")
plt.ylabel("Accuracy (log scale)")
plt.title("Trade-off Between Retrieval Size and Accuracy")
plt.grid()
plt.show()

# 3D Surface Plot
X, Y = np.meshgrid(retrieval_sizes, retrieval_sizes)
Z = np.log1p(X + Y)  # Simulated performance metric

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap="plasma")
ax.set_xlabel("Retrieval Size 1")
ax.set_ylabel("Retrieval Size 2")
ax.set_zlabel("Accuracy Gain")
ax.set_title("3D Trade-off Visualization")
plt.show()
