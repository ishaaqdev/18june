from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)
labels = kmeans.fit_predict(X)

print("Confusion Matrix:")
print(confusion_matrix(data.target, labels))

print("\nClassification Report:")
print(classification_report(data.target, labels))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap='viridis'
)
plt.title("K-Means Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
