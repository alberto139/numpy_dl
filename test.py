import numpy as np  
#import cupy as np
from mnist import mnist
#from ops import ops
from ops import ops
import cv2
from statistics import median

import cupy as cp
import time

x_train, t_train, x_test, t_test = mnist.load()
X_train = []
X_test = []

# reshape training and testing images
#for i, img in enumerate(x_train):
#    img = img.reshape(28, 28)
#    img = np.pad(img, 2, mode='constant')
#    img = img.reshape(32, 32, 1)
    
#    X_train.append(img)

#for i, img in enumerate(x_test):
#    img = img.reshape(28, 28)
#    img = np.pad(img, 2, mode='constant')
#    img = img.reshape(32, 32, 1)
    
#    X_test.append(img)

# Turn datasets into nupy arrays
# TODO: CHANGE THIS BACK TO UPPER CASE X
X_train = np.array(x_train, dtype=np.float64)
X_test = np.array(x_test, dtype=np.float64)

X_train -= int(np.mean(X_train))
X_train /= int(np.std(X_train))

X_test -= int(np.mean(X_test))
X_test /= int(np.std(X_test))


# Formula to figure out output of Conv or Pool layer
# (n + 2p - f) / s) + 1
# where n = input_size, p = padding, f = kernel_size, s = stride

def maxpooling(img, kernel_size, stride, padding='valid'):

    # Determine the ouput dimensions
    print("--------------------------------")
    input_size = len(img)
    pad_amount = 0
    output_size = int(  ((input_size - kernel_size) / stride ) + 1   )
    output = np.zeros((output_size, output_size))

    
    print("input shape: " + str(img.shape))
    print("kernel size: " + str(kernel_size))
    print("stride: " + str(stride))
    print("padding amount: " + str(pad_amount))
    print("output_size: " + str(output.shape))

    # Same Padding. Pad input so that output is of the same shape
    # p = f - 1 / 2   
    #pad_amount = int((kernel_size - 1) / 2)
    #print(pad_amount)
    #if padding == 'same':
    #    #img = np.pad(img, pad_amount, mode='constant')
    #    None

    for i, row in enumerate(range(0, len(img), stride)):
        for j, col in enumerate(range(0, len(img[0]), stride)):

            # Get the sub img corresponding to the kernel
            subimg = img[row : row + kernel_size, col : col + kernel_size]

            # Valid Padding (No Padding)
            if padding == 'valid':
                if subimg.shape[0] < kernel_size or subimg.shape[1] < kernel_size:
                    continue

            # Max Pooling
            result = max(subimg.ravel())
            output[i][j] = result

    return output

def convolution(img, kernel_size, stride, padding='valid'):
    

    # Kernel with stride
    kernel_size = 5
    stride = 1
    filters = 6
    input_size = len(img)

    weights = np.random.rand(filters, kernel_size, kernel_size)
    # np.random.normal(loc=0, scale=np.sqrt(1./(self.C*self.K*self.K)), size=(self.C, self.K, self.K))

    bias = 0
    print("weights: " + str(weights.shape))
    # Same Padding. Pad input so that output is of the same shape
    # p = f - 1 / 2   
    #pad_amount = int((kernel_size - 1) / 2  ) #+ 1

    # We overwride the pad for LeNet is 2 before even the first layer
    pad_amount = 2
    # LeNet also has valid padding
    padding = 'same'

    #if padding == 'same':
    #    img = np.pad(img, pad_amount, mode='constant')

    input_size = len(img)
    pad_amount = 0
    
    #cv2.imshow("padded_img", img)

    # Set padding amount back to zero so that our ouput is the right size
    
    # (n + 2p - f) / s) + 1
    output_size = int(  (input_size - kernel_size / stride ) + 1   )
    output = np.zeros((output_size, output_size, filters))

    print("input shape: " + str(img.shape))
    print("kernel size: " + str(kernel_size))
    print("padding amount: " + str(pad_amount))
    print("output_size: " + str(output.shape))

    for d in range(filters):
        for i, row in enumerate(range(0, len(img), stride)):
            for j, col in enumerate(range(0, len(img[0]), stride)):
                #print("subimg: " + str(( (row,row + kernel_size), (col, col + kernel_size))))
                #print((i, j))
                subimg = img[row : row + kernel_size, col : col + kernel_size]

                # Valid Padding
                if subimg.shape[0] < kernel_size or subimg.shape[1] < kernel_size:
                    continue

                # Convolution
                result = np.sum(np.multiply(subimg, weights[d]))
                output[i][j][d] = result

                test = False
                if test:
                    kernel_img = np.copy(img)
                    kernel_img[row : row + kernel_size, col : col + kernel_size] = 255
                    cv2.imshow("subimg", subimg)
                    cv2.imshow("weight", weights[d])
                    cv2.imshow("kernel", kernel_img)
                    cv2.imshow("output", output[:,:,d])
                    cv2.waitKey(0)
    return output
    
    print("Done reshaping")


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

def cross_entropy(y, y_hat):

    # One hot encode Y to create a distribution
    y_probs = np.zeros(len(y_hat))
    y_probs[y] = 1.0

    return -np.sum(y_probs * np.log(y_hat)), y_probs

class Net():
    def __init__(self):
        self.layers = []
        #self.layers.append(ops.Conv2D(6, 5, 1, "C1"))
        #self.layers.append(ops.ReLu())
        #self.layers.append(ops.MaxPool(2, 2, "S2"))
        #self.layers.append(ops.Conv2D(16, 5, 1, "C3"))
        #self.layers.append(ops.ReLu())
        #self.layers.append(ops.MaxPool(2, 2, "S4"))
        #self.layers.append(ops.Conv2D(120, 5, 1, "C5"))
        #self.layers.append(ops.ReLu())
        self.layers.append(ops.FullyConnected(784, 30, "F6"))
        self.layers.append(ops.FullyConnected(30, 30, "F6"))
        self.layers.append(ops.FullyConnected(30, 10, "F7"))
        self.layers.append(ops.Softmax())

    def forward(self, activation):
        
        # Forward pass through every layer
        for layer in self.layers:
            activation = layer.forward(activation)

        return activation

    def backward(self, dy):
        #print("backward dy: " + str(dy))
        for layer in list(reversed(self.layers))[:2]:
            #print(layer.name)
            #print(sum(dy.ravel()))
            dy = layer.backward(dy)
            #print(layer.name)
            #print(sum(dy.ravel()))
            #print(dy.shape)
        


# Initialize Network
network = Net()
import time
batch_start_time = time.time()

# Training
n = 181
epochs = 20000

for e in range(epochs):
    total_loss = 0
    correct = 0
    for i, img in enumerate(X_train):

        #img = X_train[n]


        ### Forward Pass ###
        y_hat = network.forward(img)
        #print("y_hat: " + str(y_hat))
        #print("y_hat: " + str(list(y_hat).index(max(y_hat))))
       


        ### Cross Entropy Loss ###
        y = t_train[i]
        loss, y = cross_entropy(y, y_hat)
        #print("Coss_entropy y: " + str(y))
        total_loss += loss

        if list(y_hat).index(max(y_hat)) == list(y).index(max(y)):
            correct +=1



        #print("Epoch " + str(e) + " Loss " + str(i) + "/" + str(len(X_train)) + ": " + str(loss)[:9])
    


        ### Backward Pass ###


        network.backward(y)
        #print("Time taken in backprop: " + str(time.time() - start_time))

        #cv2.imshow("img", img.reshape(28, 28))
        #cv2.waitKey(0)
    print("Epoch " + str(e) + " Loss: " + str(total_loss/len(X_train))[:9] + " Acc: " + str(correct/len(X_train)))



        

        
