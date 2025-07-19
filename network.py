import numpy as np
from preprocessing import *
from activations import *
import matplotlib.pyplot as plt
from tqdm import tqdm # progress bars while training

class NeuralNetwork():

    def __init__(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, activation: str, num_labels: int, architecture: list[int]):
        self.X = normalize(X)
        assert np.all((self.X >= 0) | (self.X <= 1)) # test to see if normalizing succeeded

        self.X, self.X_test = X.copy(), X_test.copy()
        self.y, self.y_test = y.copy(), y_test.copy()

        self.layers = {}

        self.architecture = architecture

        self.activation = activation
        assert self.activation in ["relu", "tanh", "sigmoid", "leaky_relu"]
        
        self.parameters = {}

        self.num_labels = num_labels

        self.m = X.shape[1]

        self.architecture.append(self.num_labels)

        self.num_input_features = X.shape[0]

        self.architecture.insert(0, self.num_input_features)

        self.L = len(architecture)

        assert self.X.shape == (self.num_input_features, self.m)
        assert self.y.shape == (self.num_labels, self.m)

    def initialize_parameters(self):
        for i in range(1, self.L):
            print(f"Initializing parameters for layer: {i}.")
            self.parameters["w"+str(i)] = np.random.randn(self.architecture[i], self.architecture[i-1]) * 0.01
            self.parameters["b"+str(i)] = np.zeros((self.architecture[i], 1))
    
    def forward(self):
        params = self.parameters
        self.layers["a0"] = self.X

        for l in range(1, self.L-1):
            self.layers["z"+str(l)] = np.dot(params["w"+str(l)], self.layers["a"+str(l-1)]) + params["b"+str(l)] # z_l = W_l.a_l-1 + b_l
            self.layers["a"+str(l)] = eval(self.activation)(self.layers["z"+str(l)]) # a_l = activation(z_l)
            assert self.layers["a"+str(l)].shape == (self.architecture[l], self.m)
        # final layer (output)
        self.layers["z"+str(self.L-1)] = np.dot(params["w"+str(self.L-1)], self.layers["a"+str(self.L-2)]) + params["b"+str(self.L-1)]
        self.layers["a"+str(self.L-1)] = softmax(self.layers["z"+str(self.L-1)])
        self.output = self.layers["a"+str(self.L-1)]
        assert self.output.shape == (self.num_labels, self.m)
        assert all([s for s in np.sum(self.output, axis=1)])

        cost = -np.sum(self.y * np.log(self.output + 0.000000001))

        return cost, self.layers
    
    def backpropagate(self):
        derivatives = {}
        dZ = self.output - self.y # dZ = dL/dz
        assert dZ.shape == (self.num_labels, self.m)
        dW = np.dot(dZ, self.layers["a"+str(self.L-2)].T) / self.m # dL/dW_l = dZ.(a_l-1)^T
        db = np.sum(dZ, axis=1, keepdims=True) / self.m # dL/db_l = dZ_l
        dAPrev = np.dot(self.parameters["w"+str(self.L-1)].T, dZ)
        derivatives["dW"+str(self.L-1)] = dW
        derivatives["db"+str(self.L-1)] = db

        for l in range(self.L-2, 0, -1):
            dZ = dAPrev * derivative(self.activation, self.layers["z"+str(l)]) # dZ = 
            dW = 1. / self.m * np.dot(dZ, self.layers["a"+str(l-1)].T)
            db = 1. / self.m * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = np.dot(self.parameters["w"+str(l)].T, (dZ))
            derivatives["dW"+str(l)] = dW
            derivatives["db"+str(l)] = db
        self.derivatives = derivatives

        return self.derivatives
    
    def fit(self, lr=0.01, epochs=1000):
        self.costs = []
        self.initialize_parameters()
        self.accuracies = {"train": [], "test": []}
        for epoch in tqdm(range(epochs), colour="BLUE"):
            cost, cache = self.forward()
            self.costs.append(cost)
            derivatives = self.backpropagate()
            for layer in range(1, self.L):
                self.parameters["w"+str(layer)] = self.parameters["w"+str(layer)] - lr * derivatives["dW" + str(layer)]
                self.parameters["b"+str(layer)] = self.parameters["b"+str(layer)] - lr * derivatives["db" + str(layer)]
            train_accuracy = self.accuracy(self.X, self.y)
            test_accuracy = self.accuracy(self.X_test, self.y_test)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch:3d} | Cost : {cost:.3f} | Accuracy: {train_accuracy:.3f}")
            self.accuracies["train"].append(train_accuracy)
            self.accuracies["test"].append(test_accuracy)
        print("Training terminated")
    
    def predict(self, x):
        params = self.parameters
        n_layers = self.L - 1
        values = [x]
        for l in range(1, n_layers):
            z = np.dot(params["w"+str(l)], values[l-1]) + params["b"+str(l)]
            a = eval(self.activation)(z)
            values.append(a)
        z = np.dot(params["w"+str(n_layers)], values[n_layers-1]) + params["b"+str(n_layers)]
        a = softmax(z)
        if x.shape[1]>1:
            ans = np.argmax(a, axis=0)
        else:
            ans = np.argmax(a)
        return ans
    
    def accuracy(self, X, y):
        P = self.predict(X)
        return sum(np.equal(P, np.argmax(y, axis=0))) / y.shape[1]*100



