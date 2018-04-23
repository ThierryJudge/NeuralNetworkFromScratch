from NeuralNetwork import NeuralNetwork
from matplotlib import pyplot as plt
import numpy as np


def generate_data(n):
    data = []
    labels = []
    for i in range(n):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)

        # if(x < 0):                                    #Straigt line
        # if(x < 0 and y < 0):                          #AND
        # if(x < 0 or y < 0):                           #OR
        if (x < 0 and y < 0) or (x >= 0 and y >= 0):    # XOR
            l = np.array([1, 0])
        else:
            l = np.array([0, 1])

        if abs(x) < 0 or abs(y) < 0:
            i = i - 1
        else:
            p = [[x, y]]
            data.append(p)
            labels.append(l)
    return np.array(data), np.array(labels)


def scatter_plot(data, labels):
    for i in range(len(data)):
        if i % (len(data)/1000) == 0:
            p = data[i]
            c = 'r'
            if np.argmax(labels[i]) == 1:
                c = 'b'
            plt.scatter(p[0][0], p[0][1], color=c)
    plt.title('Generated Data')
    plt.show()


if __name__ == '__main__':

    data, labels = generate_data(10000)
    test_data, test_labels = generate_data(1000)
    scatter_plot(data, labels)

    nn = NeuralNetwork(2, 2, [10])
    nn.train(data, labels)

    # custom test with plot
    total_tests = len(test_data)
    successes = 0
    failures = 0
    for i in range(len(test_data)):
        target = np.argmax(test_labels[i])
        x = np.array(test_data[i])

        y = nn.predict(x)

        c = "b"
        if y > 0.5:
            prediction = 1
        else:
            prediction = 0
            c = 'r'

        plt.scatter(x[0][0], x[0][1], color=c)
        if prediction == target:
            successes = successes + 1
        else:
            failures = failures + 1

    print("Accuracy: " + str(successes / total_tests * 100) + "% for " + str(total_tests) + " tests.")
    plt.title("Test Data")
    plt.show()
