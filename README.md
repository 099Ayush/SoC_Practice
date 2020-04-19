# SoC_Practice

The projects 'add_two_numbers' and 'calc_and_poll' are simple Django implementations coded while going through the tutorials of Django.
The calculator in 'calc_and_poll' inputs an expression which should be valid in Python, though // can be replaced by | and ** by ^, considering the general conventions.

The ML_Practice folder contains a training dataset in 'data.csv' containing data about a number of samples of Iris flowers. The Python files contain classifier-classes which can classify the flowers into two species, namely, Setosa and Versicolor, using the sepal length and the petal length as the deciding parameters.

The file 'perceptron.csv' contains a Perceptron-based classifier which can be customized for desired number of iterations or epochs, the training rate, etc.

Also, the seed parameter in such classifiers can be used as a knob or hyperparameter to generate different results for different seeds, but the same everytime for any particular seed.

'adaline_gd.csv' is the implementation of the general definition of Adaptive Linear Neuron(Adaline) learning algorithm, without the use of standardization or feature scaling. Also, the activation function in this case is the identity function. The hyperparameters are taken as inputs from the user.

Both these files plot the scatter, and a filled contour, representing the hyperplane(a straight line in 2-D) devised by the model during learning from the training dataset of the Iris flowers.
