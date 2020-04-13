import numpy as np  
import cupy as cp
import cv2

class Conv2D:

    def __init__(self, num_channels, num_filters, kernel_size, stride, name):

        self.name = name

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_channels = num_channels
        # Initialize weights
        # Random initialization of weights. In LeNet they should be a normal distribution between +-2.4 / num_filters
        self.weights = np.random.normal(loc = 0, scale=(2./(self.num_filters*self.kernel_size*self.kernel_size)), size=(self.num_filters, self.num_channels, self.kernel_size,self.kernel_size))
        self.bias = np.zeros(self.num_filters)

        self.inputs = None


    def forward(self, inputs):

        print(self.name)
        self.inputs = inputs
        input_size = len(inputs)
        # Feauture map to be returned
        output_size = int(  ((input_size - self.kernel_size) / self.stride ) + 1   )
        output = np.zeros((output_size, output_size, self.num_filters))
        
        
        for f in range(len(self.weights)):
            for r in range(len(self.inputs)):
                for c in range(len(self.inputs[0])):
                    
                    # subimg
                    subimg = self.inputs[r:r+self.kernel_size, c:c+self.kernel_size, :]
                    # Valid Padding
                    # If the subimg is smaller than the kernel size, then ignore it
                    if subimg.shape[0] < self.kernel_size or subimg.shape[1] < self.kernel_size:
                        continue

                    #cv2.imshow('subimg', subimg)
                    #cv2.waitKey(0)
                    print(self.weights[f,:,:,:].shape)
                    output[r][c][f] = np.sum(subimg.T * self.weights[f,:,:,:]) + self.bias[f]
        
        # Returning feature map
        #print(output.shape)
        output_img = output[:,:,0]
        for f in range(output.shape[2]):
            output_img = cv2.vconcat([output_img, output[:,:,f]])
            #print(output[:,:,f])
        cv2.imshow(self.name, output_img)
        cv2.waitKey(0)
        return output

    def backward(self, inputs):
        print("============= " + self.name + " =============")
        print("weights: " + str(self.weights.shape))
        print("inputs: " + str(self.inputs.shape))
        print("dy: " + str(inputs.shape))
        x = self.inputs
        dy = inputs
        dw = np.zeros((self.weights.shape))
        print("dw: " + str(dw.shape))


        # For each filter (or weight) 120 for C5
        for f in range(len(self.weights)):
            # For evey input
            for d in range(len(self.inputs.shape[2])):
                # For every row 5
                for i, row in enumerate(range(len(self.inputs))):
                    # For every col  5
                    for j, col in enumerate(range(len(self.inputs[0]))):
                        pass
        
        

class MaxPool:
    def __init__(self, kernel_size, stride, name):

        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        self.inputs = None

        # The 3rd dimension of the input is perserved

    def forward(self, inputs):

        self.inputs = inputs
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


    def backward(self, dy):
        print("Max Pool backward pass")
        print("weights: " + str(self.weights.shape))
        print("inputs: " + str(self.inputs.shape))
        print("dy: " + str(inputs.shape))
        num_filters = self.inputs.shape[2]
        input_size = len(self.inputs)
        # Feauture map to be returned
        output_size = int(  ((input_size - self.kernel_size) / self.stride ) + 1   )
        output = np.zeros((output_size, output_size, num_filters))

        # Output will be the same shape as the input to the forward pass
        dx = np.zeros(self.inputs.shape)

        # Iterate trough every dimension (channel) of the input
        for d in range(num_filters):
            # Iterate through every row
            for i, row in enumerate(range(0, len(self.inputs), self.stride)):
                # Iterate through every column
                for j, col in enumerate(range(0, len(self.inputs[0]), self.stride)):

                    # Sub section of the input to be convolved (cross-correlated)
                    # Essentialy a sliding window accross the input
                    #subimg = self.inputs[row : row + self.kernel_size, col : col + self.kernel_size]

                    # Valid Padding
                    # If the subimg is smaller than the kernel size, then ignore it
                    #if subimg.shape[0] < self.kernel_size or subimg.shape[1] < self.kernel_size:
                    #    continue

                    # Max Pooling
                    #result = max(subimg.ravel())
                    #output[i][j][d] = result
                    dx[i][j][d] = dy[i / self.kernel_size][ j / self.kernel_size ][d]
        
        # Returning feature map
        print(dx)
        return dx

class FullyConnected:

    def __init__(self, num_inputs, num_outputs, name):
        self.name = name

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = np.random.normal(loc = 0, scale=(2./(self.num_inputs)), size=(self.num_inputs,self.num_outputs))
        self.bias = np.zeros(self.num_outputs)

        self.inputs = None
        self.activation = None

        self.learning_rate = 0.01

    def forward(self, inputs):
        self.inputs = inputs
        self.activation = np.dot(inputs, self.weights) + self.bias.T
        return self.activation.ravel()

    def backward(self, dy):

        self.inputs = self.inputs.ravel()
        self.inputs = np.expand_dims(self.inputs, axis = 1)

        dy = dy.ravel()
        dy = np.expand_dims(dy, axis = 1)
       
        
        # Get the dot product of the error (dy) given the input
        # This will give you the gradients corresponding to each weight
        dw = np.dot(self.inputs, dy.T)
        db = np.sum(dy)
       

        dx = np.dot(dy.T, self.weights.T)
        dx = np.dot(self.weights, dy)

        
        old_weights = np.copy(self.weights)
        self.weights += - self.learning_rate * dw
        self.bias -= self.learning_rate * db

        diff = (old_weights - self.weights)
        print(dx.shape)
        return dx
        

class ReLu:
    def __init__(self):
        self.name = "ReLu"
        self.inputs = None

    def forward(self, inputs):
        inputs[inputs < 0] = 0
        self.inputs = inputs
        return self.inputs

    def backward(self, dy):
        dy = dy.reshape(self.inputs.shape)
        dy[self.inputs < 0] = 0
        print(dy.shape)
        return dy

class Softmax():
    def __init__(self):
        self.name = "Softmax"
        self.activation = None


    def forward(self, inputs):
        inputs = inputs.ravel()
        exp = np.exp(inputs, dtype=np.float)
        self.activation = exp/np.sum(exp)
        return self.activation

    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs), dtype=np.float64)
        self.activation = exp / np.sum(exp)
        return self.activation

    def backward(self, y_probs):
        # Derivative of Softmax
        # activation - labels
        #print("Softmax dy: " + str(self.activation - y_probs))
        #print("Softmax activation: " + str(self.activation))
        #print("Softmax y_probs: " + str(y_probs))
        print((self.activation - y_probs).shape)
        return self.activation - y_probs
        


# Other usefull functions
def cross_entropy(y, y_hat):

    # One hot encode Y to create a distribution
    y_probs = np.zeros(len(y_hat))
    y_probs[y] = 1.0

    return -np.sum(y_probs * np.log(y_hat)), y_probs


# Formula to figure out output of Conv or Pool layer
# (n + 2p - f) / s) + 1
# where n = input_size, p = padding, f = kernel_size, s = stride


def sliding_kernel(img):
    # Kernel with stride
    kernel_size = 5
    stride = 1

    input_size = len(img)
    # Same Padding. Pad input so that output is of the same shape
    # p = f - 1 / 2   
    #pad_amount = int((kernel_size - 1) / 2  ) #+ 1

    # We overwride the pad for LeNet is 2 before even the first layer
    pad_amount = 2
    # LeNet also has valid padding
    padding = 'same'

    if padding == 'same':
        img = np.pad(img, pad_amount, mode='constant')

    input_size = len(img)
    pad_amount = 0
    
    cv2.imshow("padded_img", img)

    # Set padding amount back to zero so that our ouput is the right size
    
    # (n + 2p - f) / s) + 1
    output_size = int(  (input_size - kernel_size / stride ) + 1   )
    output = np.zeros((output_size, output_size))

    print("input shape: " + str(img.shape))
    print("kernel size: " + str(kernel_size))
    print("padding amount: " + str(pad_amount))
    print("output_size: " + str(output.shape))
    for i, row in enumerate(range(0, len(img), stride)):
        for j, col in enumerate(range(0, len(img[0]), stride)):
            #print("subimg: " + str(( (row,row + kernel_size), (col, col + kernel_size))))
            print((i, j))
            subimg = img[row : row + kernel_size, col : col + kernel_size]

            # Valid Padding
            if subimg.shape[0] < kernel_size or subimg.shape[1] < kernel_size:
                continue

            # Max Pooling
            result = max(subimg.ravel())
            #print(result)
            output[i][j] = result
            
            # Median Pooling


            kernel_img = np.copy(img)
            kernel_img[row : row + kernel_size, col : col + kernel_size] = 255
            cv2.imshow("subimg", subimg)
            cv2.imshow("kernel", kernel_img)
            cv2.imshow("output", output)
            cv2.waitKey(0)
    #print(output)
    print("Done reshaping")
