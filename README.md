# SoC_Practice

The projects 'add_two_numbers' and 'calc_and_poll' are simple Django implementations coded while going through the tutorials of Django.
The calculator in 'calc_and_poll' inputs an expression which should be valid in Python, though // can be replaced by | and ** by ^, considering the general conventions.

The ML_Practice folder contains a training dataset in 'data.csv' containing data about a number of samples of Iris flowers. The Python files contain classifier-classes which can classify the flowers into two species, namely, Setosa and Versicolor, using the sepal length and the petal length as the deciding parameters.

The hyperparameters, such as the learning rate, the number of passes over the training dataset, the seed of random generation, and also the shuffle parameter in case of SGD-Adaline, are taken as input from the user. Then the model learns a classification relation (linear in these cases), and plots the scatter of the dataset, and the line(hyperplane) of classification on the same subplot, and also the plot of the cost function against the number of epochs in the second subplot.

Each python file implements an algorithm of Machine Learning, as listed below:-

1. perceptron.py: The Perceptron learning rule based on the MCP neuron model.
2. adaline_gd.py: The Adaptive Linear Neuron rule, using Gradient Descent algorithm.
3. adaline_std.py: Applying Adaline Gradient Descent algorithm on statndardized dataset.
4. adaline_sgd.py: Stochastic Gradient Descent in Adaline.
