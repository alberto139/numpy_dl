import numpy as numpy

class Conv2D:

    def __init__(self, inputs_channel, num_filters, kernel_size, padding, stride, learning_rate, name):

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.inputs_channel = inputs_channel

        # Initialize weights
        self.weights = np.zeros((self.num_filters, self.inputs_channel, self.kernel_size, self.kernel_size))

class MaxPool:
    pass

class ReLu:
    pass

class FullyConnected:
    pass