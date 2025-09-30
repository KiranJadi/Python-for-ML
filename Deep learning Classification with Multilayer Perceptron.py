import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Generate synthetic data (circles)
X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=8)

# Plot the points with their labels
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlGn, edgecolors='k', marker='o', s=50)
plt.xlabel('Feature 1 (X[:, 0])')
plt.ylabel('Feature 2 (X[:, 1])')
plt.colorbar(label='Class Label')
plt.title('make_circles Dataset: 300 samples, noise=0.1, factor=0.5')
plt.show()