import numpy as np
from AbstractLayer import sigmoid, sigmoid_prime
from AbstractLayer import AbstractLayer


class FCLayer(AbstractLayer):

    def __init__(self, prev_layer, num_out, learning_rate):
        super().__init__(learning_rate)
        print("New FC layer: " + str(prev_layer.get_out()) + ", " + str(num_out))
        self.prev_layer = prev_layer
        self.out = num_out
        self.W = np.random.randn(prev_layer.get_out(), num_out)
        self.b = np.random.randn(num_out)

    def forward(self, x):
        self.x = x
        self.z = x.dot(self.W) + self.b
        self.y = sigmoid(self.z)
        return self.next_layer.forward(self.y)

    def backward(self, dc_dy):
        dz_dw = self.x.T
        dz_db = 1
        dz_dx = self.W.T

        dy_dz = sigmoid_prime(self.z)
        dc_dz = dc_dy * dy_dz

        dc_dw = dc_dz * dz_dw
        dc_db = dc_dz * dz_db
        dc_dx = dc_dz.dot(dz_dx)

        self.W = self.W - (self.learning_rate * dc_dw)
        self.b = self.b - (self.learning_rate * dc_db)

        self.prev_layer.backward(dc_dx)

    def get_out(self):
        return self.out


