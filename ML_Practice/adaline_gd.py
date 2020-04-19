import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=100, seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, y):
        self.w_ = np.random.RandomState(self.seed).normal(loc=0.0, scale=1.0, size=1 + X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            net_input = X.dot(self.w_[1:]) + self.w_[0]
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            self.cost_.append((errors ** 2).sum() / 2.0)

        return self

    def activation(self, X):
        return X

    def predict(self, X):
        net_input = X.dot(self.w_[1:]) + self.w_[0]
        return np.where(net_input >= 0, 1, -1)


plt.figure(figsize=(14, 5))
plt.subplot(121)
df = pd.read_csv('data.csv')
X = df[['0', '2']].values[:100]
plt.scatter(X[:50, 0], X[:50, 1])
plt.scatter(X[50:, 0], X[50:, 1])
y = np.array([1 for t in range(50)] + [-1 for t in range(50)])

i_eta = float(input('Enter the learning rate: '))
i_n_iter = int(input('Enter the number of passes over the training dataset: '))
i_seed = int(input('Enter the seed for the random generator: '))

ad_gd = AdalineGD(eta=i_eta, n_iter=i_n_iter, seed=i_seed).fit(X, y)
rn = np.arange(X[:, 0].min(), X[:, 0].max(), 0.01)


def f(x):   # This function represents the classification hyperplane(line)
    w = ad_gd.w_
    return (-x * w[1] - w[0]) / w[2]


plt.plot(rn, f(rn), 'g')
plt.xlabel('Sepal Length: cm')
plt.ylabel('Petal length: cm')
plt.subplot(122)
plt.xlabel('Epoch Number')
plt.ylabel('Logarithm(10) of Cost Function')
plt.plot(range(1, len(ad_gd.cost_) + 1), np.log10(ad_gd.cost_))
plt.show()
