import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np
train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)  # 300 #0.2
print(train_X.shape)
print(train_Y.shape)
print(train_X)
print(train_Y)
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=10, cmap=plt.cm.Spectral)
plt.show()


