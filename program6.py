import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x) + 0.1*np.random.randn(100)

def lwr(x0, x, y, tau):
    w = np.exp(-(x - x0)**2 / (2*tau**2))
    return np.sum(w * y) / np.sum(w)

x1 = np.linspace(0, 2*np.pi, 200)
y1 = [lwr(i, x, y, 0.5) for i in x1]

plt.scatter(x, y,
            color='red',
            label='Training Data')
plt.plot(x1, y1,
         color='blue',
         linewidth=2,
         label='LWR Fit (tau=0.5)')
plt.title("Locally Weighted Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
