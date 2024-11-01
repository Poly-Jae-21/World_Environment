import numpy as np

# Define x and y
x = [1, 3, 6, 6]
y = [4, 5, 4, 8]

# Create array of coordinates
c = np.array([[x_val, y_val] for x_val, y_val in zip(x, y)])

# Calculate pairwise Euclidean distances
distances = np.sqrt(((c[:, np.newaxis] - c) ** 2).sum(axis=2))

# Get average distance (excluding zero distances on the diagonal)
average_distance = np.sum(distances) / (len(x) * (len(x) - 1))

print("Average Distance:", average_distance)

