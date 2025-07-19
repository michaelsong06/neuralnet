from network import *
from preprocessing import *
from activations import *
from sklearn.datasets import fetch_openml

mnist = fetch_openml(name="mnist_784")
data = mnist.data
labels = mnist.target

train_test_split_no = 60000

X_train = data.values[:train_test_split_no].T
y_train = labels[:train_test_split_no].values.astype(int)
y_train = one_hot_encode(y_train, 10).T

X_test = data.values[train_test_split_no:].T
y_test = labels[train_test_split_no:].values.astype(int)
y_test = one_hot_encode(y_test, 10).T

nn_relu = NeuralNetwork(X_train, y_train, X_test, y_test, "relu", 10, [128, 32])

epochs_relu = 200
lr_relu = 0.003
nn_relu.fit(lr=lr_relu, epochs=epochs_relu)
nn_relu.plot_cost(lr_relu)
nn_relu.plot_accuracies(lr_relu)