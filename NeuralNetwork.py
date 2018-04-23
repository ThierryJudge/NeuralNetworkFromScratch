from FCLayer import FCLayer
from InputLayer import InputLayer
from OutputLayer import OutputLayer
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import spline


class NeuralNetwork:

    epochs = 100000
    learning_rate = 0.01

    def __init__(self, input_size, output_size, layers=None):
        self.input_layer = InputLayer(input_size)
        self.current_layer = self.input_layer

        if layers is not None:
            for l in layers:
                self.current_layer.add_next_layer(FCLayer(self.current_layer, l, self.learning_rate))
                self.current_layer = self.current_layer.get_next_layer()

        self.current_layer.add_next_layer(OutputLayer(self.current_layer, output_size, self.learning_rate))
        self.current_layer = self.current_layer.get_next_layer()
        self.output_layer = self.current_layer

    def train(self, data, labels, plot=True):
        costs = []

        for i in range(self.epochs):

            r = np.random.randint(len(data) - 1)
            x = data[r]
            target = labels[r]

            # reshape to 2d array (used for mnist data)
            if x.ndim == 1:
                x = x.reshape(1, x.shape[0])

            y = self.input_layer.forward(x)

            cost = cross_entropy(y, target)

            self.output_layer.backward(target)

            if i % 100 == 0:
                print("Step: " + str(i) + ", cost: " + str(cost))
                costs.append(cost)

        if plot:
            plt.plot(costs)
            plt.title("Cost")
            plt.show()

    def batch_train(self, data, labels, batch_size = 1, plot=True):
        costs = []
        for i in range(self.epochs):

            r = np.random.randint(len(data) - batch_size - 1)
            xs = data[r:r+batch_size]
            targets = labels[r:r+batch_size]
            dc_dy = 0
            cost = 0
            for j in range(batch_size):
                x = xs[j]

                target = targets[j]

                # reshape to 2d array (used for mnist data)
                if x.ndim == 1:
                    x = x.reshape(1, x.shape[0])

                y = self.input_layer.forward(x)

                cost = cost + cross_entropy(y, target)

                dc_dy = dc_dy + (y - target)

            dc_dy = dc_dy / batch_size
            cost = cost / batch_size

            self.output_layer.backward_batch(dc_dy)

            if i % 100 == 0:
                print("Step: " + str(i) + ", cost: " + str(cost))
                costs.append(cost)

        if (plot):
            plt.plot(cost)
            plt.title("Cost")
            plt.show()

    def test(self, data, labels):
        total_tests = data.shape[0]
        successes = 0
        failures = 0
        for i in range(total_tests):
            x = data[i]
            y_ = np.argmax(labels[i])

            y = self.predict(x)

            if y == y_:
                successes = successes + 1
            else:
                failures = failures + 1

        print("Accuracy: " + str(successes / total_tests * 100) + "% for " + str(total_tests) + " tests.")

    def predict(self, x):
        return np.argmax(self.input_layer.forward(x))


def cross_entropy(y, target):
    return - np.sum(np.multiply(target, np.log(y)))
