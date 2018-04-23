import numpy as np
from AbstractLayer import AbstractLayer


class InputLayer(AbstractLayer):

    def __init__(self, input_size):
        super().__init__(0)
        print("New input layer: " + str(input_size))
        self.input_size = input_size

    def forward(self, x):
        if x.size != self.input_size:
            print("Error: input size does not match")
        else:
            return self.next_layer.forward(x)

    def backward(self, dc_dy):
        pass

    def get_out(self):
        return self.input_size
