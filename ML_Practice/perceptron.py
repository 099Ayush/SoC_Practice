import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perceptron(object):

    def __init__(self, f_n_epochs=1000, f_eta=0.001, f_seed=1):
        self.n_epochs = f_n_epochs    # Number of passes over training dataset
        self.eta = f_eta  # The training rate
        self.seed = f_seed    # Seed to generate random weight vector, can be used to regain same result

    def fit(self, f_X, f_y):

        # Generate random weight vector, one extra element as the bias unit.
        self.w_ = np.random.RandomState(self.seed).normal(loc=0.0, scale=1.0, size=1 + f_X.shape[1])
        self.errors_ = []   # A measure of how well the model performs after enough number of epochs.

        for _ in range(self.n_epochs):
            n_errors = 0
            for x, target in zip(f_X, f_y):     # Execute the perceptron learning algorithm
                predicted = self.predict(x)
                error = target - predicted
                self.w_[1:] += self.eta * error * x
                self.w_[0] += self.eta * error
                n_errors += int(error != 0)
            self.errors_.append(n_errors)

        return self

    def predict(self, f_X):  # Function to make prediction about a new dataset,
                                # or one from the training dataset while learning.
        value = f_X.dot(self.w_[1:]) + self.w_[0]
        return np.where(value >= 0, 1, -1)


n_epochs = int(input('The number of repetitions (epochs): '))
eta = float(input('Learning Rate, eta: '))
seed = int(input('Seed value (enter a random integer): '))

df = pd.read_csv('data.csv')
X = df[['0', '2']][:100].values  # Training examples as points on 2-d plane.

y = df['4'][:100].replace('Iris-setosa', 1).replace('Iris-versicolor', -1).values    # Target values.

ppn = Perceptron(f_n_epochs=n_epochs, f_eta=eta, f_seed=seed)
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
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epoch number')
plt.ylabel('Number of Updates')
plt.show()
