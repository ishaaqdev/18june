import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = fetch_olivetti_faces()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = GaussianNB()
model.fit(X_train, y_train)

print("Accuracy:",
      model.score(X_test, y_test) * 100)

pred = model.predict(X_test)

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(64, 64),
              cmap='grey')
    ax.set_title(f"Pred: {pred[i]}")
    ax.axis('off')
plt.show()
