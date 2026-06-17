import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()

pca = PCA(n_components=2)
reduced = pca.fit_transform(iris.data)

colors = ['r', 'g', 'b']
for i, name in enumerate(iris.target_names):
    mask = iris.target == i
    plt.scatter(reduced[mask, 0],
                reduced[mask, 1],
                label=name,
                color=colors[i])
plt.title("PCA on Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()

print("Variance Explained:",
      pca.explained_variance_ratio_.round(3))
