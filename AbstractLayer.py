import numpy as np


class AbstractLayer:

    next_layer = None

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def forward(self, x):
        pass

    def backward(self, dc_dy):
        pass

    def get_out(self):
        pass

    def add_next_layer(self, next_layer):
        self.next_layer = next_layer

    def get_next_layer(self):
        return self.next_layer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.max(x, 0)


def relu_prime(x):
    return (x > 0).astype(x.dtype)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - np.tanh(x) ** 2


def softmax(x):
    e = np.exp(x)
    dist = e / np.sum(e)
    return dist


