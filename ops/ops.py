import numpy  as np  
import cupy #as np
import cv2

class Conv2D:

    def __init__(self, num_channels, num_filters, kernel_size, stride, learning_rate, name):

        self.name = name
        self.learning_rate = learning_rate
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_channels = num_channels
        # Initialize weights
        # Random initialization of weights. In LeNet they should be a normal distribution between +-2.4 / num_filters
        self.weights = np.random.normal(loc = 0, scale=(1./(self.num_filters*self.kernel_size*self.kernel_size)), size=(self.num_filters, self.num_channels, self.kernel_size,self.kernel_size))
        self.bias = np.zeros((self.num_filters, 1))

        self.inputs = None


    def forward(self, inputs):

        #print(self.name)
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
            
                    # feature_maps[f,w,h]=np.sum(self.inputs[:,w:w+self.K,h:h+self.K]*self.weights[f,:,:,:])+self.bias[f]
                    output[r][c][f] = np.sum(subimg.T * self.weights[f,:,:,:]) + self.bias[f]
        
        # Returning feature map
        #print(output.shape)
        #if self.name == 'C1':
        output_img = output[:,:,0]
        for f in range(output.shape[2]):
            output_img = cv2.vconcat([output_img, output[:,:,f]])
            #print(output[:,:,f])
        output_img = output_img / np.max(output_img)
        output_img = cv2.resize(output_img, (output_img.shape[1] * 2, output_img.shape[0] * 2))
        cv2.imshow(self.name, output_img)
        cv2.waitKey(1)

        #print("====== " + str(self.name) + " ======")
        #print("max weights: " + str(np.max(self.weights)))
        #print("min weights: " + str(np.min(self.weights)))

            #print("max bias: " + str(np.max(self.bias)))
            #print("min bias: " + str(np.min(self.bias)))
            #print("in: " + str(self.inputs.shape))
            #print("out: " + str(output.shape))
        return output

    def backward(self, dy):
        #print("====== " + str(self.name) + " Backward ======")
        dw = np.zeros((self.weights.shape))
        dx = np.zeros(self.inputs.shape)
        db = np.zeros(self.bias.shape)

        rows, cols, filters = dy.shape

        # Perform Convolution to determine dw and dx
        for f in range(filters):
            for r in range(rows):
                for c in range(cols):
                    subimg = self.inputs[r:r+self.kernel_size, c:c+self.kernel_size, :]

                    # Valid Padding
                    # If the subimg is smaller than the kernel size, then ignore it
                    if subimg.shape[0] < self.kernel_size or subimg.shape[1] < self.kernel_size:
                        continue

                    dw[f,:,:,:]  += dy[r,c,f] * subimg.T
                    dx[r:r+self.kernel_size, c:c+self.kernel_size, :] += dy[r,c, f] * self.weights[f,:,:,:].T
        
        # Calculate db
        for f in range(filters):
            db[f] = np.sum(dy[:, :, f])

        # Update weights and biases
        #print("self.weights")
        #print(self.weights[0])
        #print("dw")
        #print(dw[0])
        old_weights = np.copy(self.weights)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        #print("diff: " + str(self.weights[0] - old_weights[0]))
        #print("Updated waself.weights")
        #print(self.weights[0])
            
        # Chain rule for the next (previous) layer
        return dx


    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias

       
class MaxPool:
    def __init__(self, kernel_size, stride, name):
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        self.inputs = None


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
                    subimg = inputs[row : row + self.kernel_size, col : col + self.kernel_size, d]

                    # Valid Padding
                    # If the subimg is smaller than the kernel size, then ignore it
                    if subimg.shape[0] < self.kernel_size or subimg.shape[1] < self.kernel_size:
                        continue

                    # Max Pooling
                    result = np.max(subimg)
                    output[i][j][d] = result
        
        #output_img = output[:,:,0]
        #for f in range(output.shape[2]):
        #    output_img = cv2.vconcat([output_img, output[:,:,f]])
            #print(output[:,:,f])

        #print('diff: ' + str(output[:,:,5] - output[:,:,4]))
        #cv2.imshow(self.name, output_img)
        #cv2.waitKey(0)

        # Returning feature map
        return output

    def backward(self, dy):

        num_filters = self.inputs.shape[2]
        input_size = len(self.inputs)

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
                    dx[i][j][d] = dy[i // self.kernel_size][ j // self.kernel_size ][d]
        
        # Returning feature map
        return dx

    def extract(self):
        return

class FullyConnected:

    def __init__(self, num_inputs, num_outputs, learning_rate, name):
        self.name = name

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = np.random.normal(loc = 0, scale=(2./(self.num_inputs)), size=(self.num_inputs,self.num_outputs))
        self.bias = np.zeros((self.num_outputs,1))

        self.inputs = None
        self.activation = None

        self.learning_rate = learning_rate


    def forward(self, inputs):
        self.inputs = inputs

        #print("====== " + str(self.name) + " ======")
        #print("in: " + str(self.inputs.shape))
        self.activation = np.dot(inputs, self.weights) + self.bias.T

       
        #print("====== " + str(self.name) + " ======")
        #print("max weights: " + str(np.max(self.weights)))
        #print("min weights: " + str(np.min(self.weights)))
        
        #print("max bias: " + str(np.max(self.bias)))
        #print("min bias: " + str(np.min(self.bias)))

        #print(self.bias)
        return self.activation.ravel()

    def backward(self, dy):

        #print("--- " + str(self.name) + "  Backward ---")
        #print(dy)
        self.inputs = self.inputs.ravel()
        self.inputs = np.expand_dims(self.inputs, axis = 1)


        dy = dy.ravel()
        dy = np.expand_dims(dy, axis = 1)
       
        
        # Get the dot product of the error (dy) given the input
        # This will give you the gradients corresponding to each weight
        dw = np.dot(self.inputs, dy.T)
        db = np.sum(dy, axis=1, keepdims=True)
       

        #dx = np.dot(dy.T, self.weights.T)
        dx = np.dot(self.weights, dy)

        
        old_weights = np.copy(self.weights)
        #print("self.weights")
        #print(self.weights[0])
        #print("dw")
        #print(dw[0])
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        #print('updated weights')
        #print(self.weights[0])
        
        return dx

    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias

class Flatten:
    def __init__(self):
        self.name = "Flatten"
        self.inputs = None
    def forward(self, inputs):
        self.inputs = inputs
        self.C, self.W, self.H = inputs.shape
        output = inputs.reshape(1, self.C*self.W*self.H)
        #print("====== " + str(self.name) + " ======")
        #print("in: " + str(self.inputs.shape))
        #print("out: " + str(output.shape))

        return output

    def backward(self, dy):
        return dy.reshape(self.C, self.W, self.H)

    def extract(self):
        return


class ReLu:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.inputs = inputs
        ret = inputs.copy()
        ret[ret < 0] = 0
        return ret
    def backward(self, dy):
        dx = dy.copy()
        dx[self.inputs < 0] = 0
        return dx
    def extract(self):
        return


class Sigmoid:
    def __init__(self):
        self.name = "Sigmoid"
        self.inputs = None

    def sigmoid_backend(x):
        """Applies the sigmoid function elementwise."""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def forward(self, inputs):
        self.inputs = inputs
        x = inputs
        #self.activation = (1.0 / (1 + np.exp(-self.inputs) ))
        self.activation = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        print("Sigmoid Forward max: " + str(np.max(self.activation)))
        print("Sigmoid Forward min: " + str(np.min(self.activation)))
        return self.activation

    def backward(self, dy):
        #dx = dy * (self.inputs * (1 - self.inputs))


        def sigmoid_backend(x):
            """Applies the sigmoid function elementwise."""
            return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

        dx = sigmoid_backend(dy) * (1 - sigmoid_backend(dy))
        
        if dx.shape[1] == 1:
            dx = np.expand_dims(dx, axis = 1)
        #dx = (dy * (1.0 - dy)) # WRONG SHAPE
        #ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
        #dx = (1 - self.activation) * self.activation # WRONG Makes everything the same
        #print("Sigmoid Backward: " + str(dx))
        #print(dx)
        return dx

    def extract(self):
        return

class Softmax():
    def __init__(self):
        self.name = "Softmax"
        self.activation = None


    def forward(self, inputs):

        
        
        # CLIP THE INPUTS?
        # +- 5.0?
        #if np.max(inputs) > 4.0 or np.min(inputs) < -4.0:
        #    inputs = inputs / np.max(inputs)

        #if np.max(inputs) >= 100:
            #print('There might be something wrong: ' + str(inputs))
            #print("inputs: " + str(inputs))
            #print("exp: " + str(exp))
            #print("sum exp: " + str(np.sum(exp)))
            #print("activation: " + str(self.activation))
        inputs = inputs.ravel()
        exp = np.exp(inputs, dtype=np.float)
        self.activation = exp/np.sum(exp)

        #print("====== " + str(self.name) + " ======")
        #print("max weights: " + str(np.max(self.activation)))
        #print("min weights: " + str(np.min(self.activation)))
    

        return self.activation

    def forward_stable(self, inputs):
        inputs = inputs - np.max(inputs)
        exp = np.exp(inputs - np.max(inputs), dtype=np.float128)
        self.activation = exp / np.sum(exp)
        return self.activation

    def backward(self, y_probs):
        # Derivative of Softmax
        # activation - labels
        #print("Softmax Backward: " + str(self.activation - y_probs))
        return self.activation - y_probs

    def extract(self):
        return


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
