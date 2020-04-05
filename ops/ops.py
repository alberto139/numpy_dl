import numpy as np  

class Conv2D:

    def __init__(self, num_filters, kernel_size, stride, name):

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride

        # Initialize weights
        #self.weights = np.zeros((self.num_filters, self.inputs_channel, self.kernel_size, self.kernel_size))
        # Random initialization of weights. In LeNet they should be a normal distribution between +-2.4 / num_filters
        #self.weights = np.random.rand(self.num_filters, self.kernel_size, self.kernel_size)
        self.weights = np.random.normal(loc = 0, scale=(2./(self.num_filters*self.kernel_size*self.kernel_size)), size=(self.num_filters,self.kernel_size,self.kernel_size))
        self.bias = np.zeros(self.num_filters)

    def forward(self, inputs):

        input_size = len(inputs)
        # Feauture map to be returned
        output_size = int(  ((input_size - self.kernel_size) / self.stride ) + 1   )
        output = np.zeros((output_size, output_size, self.num_filters))
        
        # Iterate through all the conv filters
        for w, weight in enumerate(self.weights):
            # Iterate trough every dimension (channel) of the input
            for d in range(inputs.shape[2]):
                # Iterate through every row
                for i, row in enumerate(range(0, len(inputs), self.stride)):
                    # Iterate through every column
                    for j, col in enumerate(range(0, len(inputs[0]), self.stride)):

                        # Sub section of the input to be convolved (cross-correlated)
                        # Essentialy a sliding window accross the input
                        inputs_d = inputs[:,:,d]
                        subimg = inputs_d[row : row + self.kernel_size, col : col + self.kernel_size]

                        # Valid Padding
                        # If the subimg is smaller than the kernel size, then ignore it
                        if subimg.shape[0] < self.kernel_size or subimg.shape[1] < self.kernel_size:
                            continue

                        # Convolution
                        result = np.sum(np.multiply(subimg, self.weights[d])) + self.bias[d]
                        output[i][j][d] = result
        
        # Returning feature map
        return output

class MaxPool:
    def __init__(self, kernel_size, stride, name):

        self.kernel_size = kernel_size
        self.stride = stride

        # The 3rd dimension of the input is perserved

    def forward(self, inputs):

        num_filters = inputs.shape[2]
        input_size = len(inputs)
        # Feauture map to be returned
        output_size = int(  ((input_size - self.kernel_size) / self.stride ) + 1   )
        output = np.zeros((output_size, output_size, num_filters))

        # Iterate trough every dimension (channel) of the input
        for d in range(num_filters):
            # Iterate through every row
            for i, row in enumerate(range(0, len(inputs), self.stride)):
                # Iterate through every column
                for j, col in enumerate(range(0, len(inputs[0]), self.stride)):

                    # Sub section of the input to be convolved (cross-correlated)
                    # Essentialy a sliding window accross the input
                    subimg = inputs[row : row + self.kernel_size, col : col + self.kernel_size]

                    # Valid Padding
                    # If the subimg is smaller than the kernel size, then ignore it
                    if subimg.shape[0] < self.kernel_size or subimg.shape[1] < self.kernel_size:
                        continue

                    # Max Pooling
                    result = max(subimg.ravel())
                    output[i][j][d] = result
        
        # Returning feature map
        return output

class FullyConnected:

    def __init__(self, num_inputs, num_outputs, name):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = np.random.normal(loc = 0, scale=(2./(self.num_inputs)), size=(self.num_inputs,self.num_outputs))
        self.bias = np.zeros(self.num_outputs)

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias.T



class ReLu:
    def __init__(self):
        pass

    def forward(self, inputs):
        inputs[inputs < 0] = 0
        return inputs

class Softmax():
    def __inti__(self):
        pass

    def forward(self, inputs):
        inputs = inputs.ravel()
        exp = np.exp(inputs, dtype=np.float)
        self.out = exp/np.sum(exp)
        return self.out

    def backward(self)
