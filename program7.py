from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X[:, [0]],
    y,
    test_size=0.2,
    random_state=42
)
lr = LinearRegression()
lr.fit(X_train, y_train)
plt.scatter(X_train, y_train, s=10)
plt.plot(
    X_train,
    lr.predict(X_train),
    color='red'
)
plt.title("Linear Regression")
plt.show()

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
pr = LinearRegression()
pr.fit(X_poly, y_train)
plt.scatter(X_train, y_train, s=10)
plt.scatter(
    X_train,
    pr.predict(X_poly),
    color='red',
    s=10
)
plt.title("Polynomial Regression")
plt.show()
print("Linear Regression Score:", lr.score(X_test, y_test))
