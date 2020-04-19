import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AdalineSGD(object):

    def __init__(self, eta=0.001, n_iter=100, shuffle=True, seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.seed = seed
        self.w_initialized = False
        self.rgen = np.random.RandomState(self.seed)

    def initialize_w(self, n):
        self.w_ = self.rgen.normal(0.0, 1.0, n)
        self.w_initialized = True

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        return x

    def predict(self, x):
        return np.where(self.activation(self.net_input(x)) >= 0, 1, -1)

    def _shuffle(self, X, y):
        r = self.rgen.permutation(y.shape[0])
        return X[r], y[r]

    def update_w(self, x, target):
        error = target - self.activation(self.net_input(x))
        self.w_[1:] += self.eta * error * x
        self.w_[0] += self.eta * error
        return error ** 2 / 2

    def fit(self, X, y):
        self.cost_ = []
        self.initialize_w(X.shape[1] + 1)
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for x, target in zip(X, y):
                cost.append(self.update_w(x, target))
            avg_cost = np.mean(cost)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        n = y.shape[0]
        if not self.w_initialized:
            self.initialize_w(n + 1)
        if n == 1:
            self.update_w(X, y)
        else:
            for x, target in zip(X, y):
                self.update_w(x, target)
        return self


plt.figure(figsize=(12, 5))
plt.subplot(121)
df = pd.read_csv('data.csv')
X = df[['0', '2']].values[:100]
for i in range(2):  # To standardize the dataset.
    X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
plt.scatter(X[:50, 0], X[:50, 1])
plt.scatter(X[50:, 0], X[50:, 1])
y = np.array([1 for t in range(50)] + [-1 for t in range(50)])

i_eta = float(input('Enter the learning rate: '))
i_n_iter = int(input('Enter the number of passes over the training dataset: '))
i_seed = int(input('Enter the seed for the random generator: '))
i_shuffle = bool(int(input('Whether or not to shuffle the dataset in every epoch(enter 0 or 1)? ')))

ad_sgd = AdalineSGD(eta=i_eta, n_iter=i_n_iter, seed=i_seed, shuffle=True).fit(X, y)
rn = np.arange(X[:, 0].min(), X[:, 0].max(), 0.01)


def f(x):   # This function represents the classification hyperplane(line)
    w = ad_sgd.w_
    return (-x * w[1] - w[0]) / w[2]


plt.plot(rn, f(rn), 'g')
plt.subplot(122)
plt.plot(range(1, len(ad_sgd.cost_) + 1), ad_sgd.cost_)
plt.show()
