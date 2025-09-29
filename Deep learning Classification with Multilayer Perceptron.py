import numpy as nр

import matplotlib.pyplot as plt from sklearn.datasets import make_circles

A

#Generate synthetic data (circles)

X , y = make_circles(n_samples=300, noise 0.1, factor 0.5, random_state=8)

#Plot the points with their labels

plt.scatter(X:, 0), X[:, 1), c=y, cmap=plt.cm.RdYlGn, edgecolors='k', marker='o', s=50)

plt.figure(figsize=(8, 6))

plt.xlabel('Feature 1 (X[:, 0])') plt.ylabel('Feature 2 (X[:, 1]))

plt.colorbar(label='Class Label')

plt.title('Plot of Points with Labels (make_circles Dataset)')

plt.show()