import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Accuracy:",
      model.score(X_test, y_test) * 100)

pred = model.predict([X_test[0]])
print("Predicted Class:",
      "Benign" if pred == 1 else "Malignant")

plt.figure(figsize=(12, 8))
plot_tree(model, filled=True)
plt.show()
