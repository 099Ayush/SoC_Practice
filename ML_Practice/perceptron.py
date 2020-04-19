import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perceptron(object):

    def __init__(self, n_iter=1000, eta=0.001, seed=1):
        self.n_iter = n_iter    # Number of passes over training dataset
        self.eta = eta  # The training rate
        self.seed = seed    # Seed to generate random weight vector, can be used to regain same result

    def fit(self, X, y):

        # Generate random weight vector, one extra element as the bias unit.
        self.w_ = np.random.RandomState(self.seed).normal(loc=0.0, scale=1.0, size=1 + X.shape[1])
        self.errors_ = []   # A measure of how well the model performs after enough number of epochs.

        for _ in range(self.n_iter):
            n_errors = 0
            for x, target in zip(X, y):     # Execute the perceptron learning algorithm
                predicted = self.predict(x)
                error = target - predicted
                self.w_[1:] += self.eta * error * x
                self.w_[0] += self.eta * error
                n_errors += int(error != 0)
            self.errors_.append(n_errors)

        return self

    def predict(self, X):  # Function to make prediction about a new dataset,
                           # or one from the training dataset while learning.
        value = X.dot(self.w_[1:]) + self.w_[0]
        return np.where(value >= 0, 1, -1)


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

ppn = Perceptron(eta=i_eta, n_iter=i_n_iter, seed=i_seed).fit(X, y)
rn = np.arange(X[:, 0].min(), X[:, 0].max(), 0.01)


def f(x):   # This function represents the classification hyperplane(line)
    w = ppn.w_
    return (-x * w[1] - w[0]) / w[2]


plt.plot(rn, f(rn), 'g')
plt.xlabel('Sepal Length: cm')
plt.ylabel('Petal length: cm')
plt.subplot(122)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_)
plt.xlabel('Epoch Number')
plt.ylabel('Number of Updates')
plt.show()
