import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import time

np.random.seed(time.time())
points = np.random.rand(20, 2)

test_point = np.random.rand(1, 2)

# Number of neighbors to use by default for kneighbors queries.
k = 3

# Create KNN with k neighbors
knn = NearestNeighbors(n_neighbors=k)
knn.fit(points)  # Fit the nearest neighbors estimator from the training dataset.

# Find k-neighbors of a point.
distances, indices = knn.kneighbors(test_point)

# Print the 'indices' of the nearest neighbors
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)

# Show the KNN result
plt.scatter(points[:, 0], points[:, 1], c='b', label='Data')
plt.scatter(test_point[0, 0], test_point[0, 1], c='r', marker='x', label='Random point')
plt.scatter(points[indices, 0], points[indices, 1], c='g', marker='o', label='Nearest neighbors')
plt.legend()
plt.xlabel('Trục X')
plt.ylabel('Trục Y')
plt.title('KNN with k = 3')
plt.show()
