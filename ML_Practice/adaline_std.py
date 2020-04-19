import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class AdalineGD(object):

    def __init__(self, f_eta=0.01, f_n_epochs=100, f_seed=1):
        self.eta = f_eta
        self.n_epochs = f_n_epochs
        self.seed = f_seed

    def fit(self, f_X, f_y):
        self.w_ = np.random.RandomState(self.seed).normal(loc=0.0, scale=1.0, size=1 + f_X.shape[1])
        self.cost_ = []
        for _ in range(self.n_epochs):
            net_input = f_X.dot(self.w_[1:]) + self.w_[0]
            output = self.activation(net_input)
            errors = f_y - output
            self.w_[1:] += self.eta * f_X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            self.cost_.append((errors ** 2).sum() / 2.0)

        return self

    def activation(self, f_X):
        return f_X

    def predict(self, f_X):
        net_input = f_X.dot(self.w_[1:]) + self.w_[0]
        return np.where(net_input >= 0, 1, -1)


n_epochs = int(input('The number of repetitions (epochs): '))
eta = float(input('Learning Rate, eta: '))
seed = int(input('Seed value (enter a random integer): '))

df = pd.read_csv('data.csv')
X = df[['0', '2']][:100].values  # Training examples as points on 2-d plane.

for i in range(2):
    X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()

y = df['4'][:100].replace('Iris-setosa', 1).replace('Iris-versicolor', -1).values    # Target values.

ppn = AdalineGD(f_n_epochs=n_epochs, f_eta=eta, f_seed=seed)
ppn.fit(X, y)   # Train the model.

# To create a filled contour plot consisting of division hyperplane(here a straight line):-
x1 = np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.01)
x2 = np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.01)

xx1, xx2 = np.meshgrid(x1, x2)

points = np.array(list(zip(xx1.ravel(), xx2.ravel())))
z = []
for point in points:
    z.append(ppn.predict(point))
z = np.array(z)
z = z.reshape(xx1.shape)
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:50, 0], X[:50, 1], marker='o') # Scatter plot for Setosa Iris flowers.
plt.scatter(X[50:, 0], X[50:, 1], marker='x') # Scatter plot for Versicolor Iris flowers.
plt.contourf(x1, x2, z, alpha=0.3)
plt.xlabel('Sepal Length: cm')
plt.ylabel('Petal length: cm')
plt.subplot(122)
plt.plot(range(1, len(ppn.cost_) + 1), np.log10(ppn.cost_), marker='o')
plt.xlabel('Epoch number')
plt.ylabel('Logarithm of Cost Function(sum of squared errors)')
plt.show()