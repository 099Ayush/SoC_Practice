import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perceptron(object):

    def __init__(self, n_epochs=1000, eta=0.001, seed=1):
        self.n_epochs = n_epochs    # Number of passes over training dataset
        self.eta = eta  # The training rate
        self.seed = seed    # Seed to generate random weight vector, can be used to regain same result

    def fit(self, X, y):

        # Generate random weight vector, one extra element as the bias unit.
        self.w_ = np.random.RandomState(self.seed).normal(loc=0.0, scale=1.0, size=1 + X.shape[1])
        self.errors_ = []   # A measure of how well the model performs after enough number of epochs.

        for _ in range(self.n_epochs):
            n_errors = 0
            for x, target in zip(X, y): # Execute the perceptron learning algorithm
                predicted = self.prediction(x)
                error = target - predicted
                self.w_[1:] += self.eta * error * x
                self.w_[0] += self.eta * error
                n_errors += int(error != 0)
            self.errors_.append(n_errors)

        return self

    def prediction(self, X): # Function to make prediction about a new dataset,
                             # or one from the training dataset while learning.
        value = X.dot(self.w_[1:]) + self.w_[0]
        return np.where(value >= 0, 1, -1)


n_epochs = int(input('The number of repetitions (epochs): '))
eta = float(input('Learning Rate, eta: '))
seed = int(input('Seed value (enter a random integer): '))

df = pd.read_csv('data.csv')
df1 = df[df['4'] == 'Iris-setosa']
df2 = df[df['4'] == 'Iris-versicolor']
ls11 = [t[0] for t in df1[['0']].values]
ls12 = [t[0] for t in df1[['2']].values]
ls21 = [t[0] for t in df2[['0']].values]
ls22 = [t[0] for t in df2[['2']].values]

X = np.array(list(zip(ls11 + ls21, ls12 + ls22)))   # Training examples as points on 2-d plane.
y = np.array([1 for t in range(len(ls11))] + [-1 for t in range(len(ls21))])    # Target values.

ppn = Perceptron(n_epochs=n_epochs, eta=eta, seed=seed)
ppn.fit(X, y)   # Train the model.

# To create a filled contour plot consisting of division hyperplane(here a straight line):-
x1 = np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.01)
x2 = np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.01)

xx1, xx2 = np.meshgrid(x1, x2)

points = np.array(list(zip(xx1.ravel(), xx2.ravel())))
z = []
for point in points:
    z.append(ppn.prediction(point))
z = np.array(z)
z = z.reshape(xx1.shape)
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(ls11, ls12, marker='o') # Scatter plot for Setosa Iris flowers.
plt.scatter(ls21, ls22, marker='x') # Scatter plot for Versicolor Iris flowers.
plt.contourf(x1, x2, z, alpha=0.3)
plt.xlabel('Sepal Length: cm')
plt.ylabel('Petal length: cm')
plt.subplot(122)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_)
plt.xlabel('Epoch number')
plt.ylabel('Number of updates')
plt.show()
