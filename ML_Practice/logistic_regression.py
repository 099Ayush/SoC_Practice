import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression(object):

    def __init__(self, eta=0.0001, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.rgen = np.random.RandomState(random_state)

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        return sigmoid(np.clip(x, -250, 250))

    def fit(self, X, y):
        self.w_ = self.rgen.normal(size=1 + X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * np.dot(X.T, errors)
            self.w_[0] += self.eta * np.sum(errors)
            cost = -np.dot(y, np.log(output)) - np.dot(1 - y, np.log(1 - output))
            self.cost_.append(cost)
        return self

    def predict(self, x):
        return np.where(self.activation(self.net_input(x)) >= 0.5, 1, -1)


def implement(classifier: LogisticRegression, X: np.ndarray, y: np.ndarray):
    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    plt.scatter(X[:50, 0], X[:50, 1])
    plt.scatter(X[50:, 0], X[50:, 1])
    x_ = np.arange(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 0.002)
    y_ = np.arange(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, 0.002)
    x1, y1 = np.meshgrid(x_, y_)
    points = np.array([x1.ravel(), y1.ravel()]).T
    Z = classifier.predict(points)
    Z = Z.reshape(x1.shape)
    plt.contourf(x1, y1, Z, alpha=0.3)
    plt.subplot(122)
    plt.plot(range(1, len(classifier.cost_) + 1), classifier.cost_)
    plt.show()


cls = LogisticRegression()
iris = datasets.load_iris()
X = iris['data'][:100, [0, 2]]
y = iris['target'][:100]
cls.fit(X, y)
implement(cls, X, y)
