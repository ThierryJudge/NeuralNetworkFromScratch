from NeuralNetwork import NeuralNetwork
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np

data_dir = '/tmp/tensorflow/mnist/input_data'


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt


if __name__ == '__main__':

    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    data = mnist.train.images
    labels = mnist.train.labels
    test_data = mnist.test.images
    test_labels = mnist.test.labels

    nn = NeuralNetwork(784, 10)
    nn.train(data, labels)
    nn.test(test_data, test_labels)

    for i in range(10):
        x, y_ = mnist.test.next_batch(1)

        y = nn.predict(x)

        y_ = np.argmax(y_)

        print("Label: " + str(y_) + ", Prediction: " + str(y))
        plot = gen_image(x)
        plot.title("Test " + str(i + 1) + "\nLabel: " + str(y_) + ", Prediction: " + str(y))
        plot.show()
