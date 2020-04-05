import numpy as np  
from mnist import mnist
#from ops import ops
from ops import ops
import cv2
from statistics import median

x_train, t_train, x_test, t_test = mnist.load()
X_train = []
X_test = []

# reshape training and testing images
for i, img in enumerate(x_train):
    img = img.reshape(28, 28)
    img = np.pad(img, 2, mode='constant')
    img = img.reshape(32, 32, 1)
    
    X_train.append(img)

for i, img in enumerate(x_test):
    img = img.reshape(28, 28)
    img = np.pad(img, 2, mode='constant')
    img = img.reshape(32, 32, 1)
    
    X_test.append(img)

# Turn datasets into nupy arrays
X_train = np.array(X_train, dtype=np.float64)
X_test = np.array(X_test, dtype=np.float64)

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
    #print("y: " + str(y_probs))
    #print("y_hat: " + str(y_hat))
    #print("y * log(y_hat) = " + str(-( y_probs * np.log(y_hat))))

    return -np.sum(y_probs * np.log(y_hat))


    
    


for i, img in enumerate(X_train):

    
    print("Input shape: " + str(img.shape))
    cv2.imshow("input", img)

    # Convolution layer. 
    # Input: (32, 32, 1) Output: (28, 28, 6)
    C1 = ops.Conv2D(6, 5, 1, "C1")
    img = C1.forward(img)
    print("C1 shape: " + str(img.shape))
    #cv2.imshow("C1", img[:,:,0])

    # ReLu Activation
    relu = ops.ReLu()
    img = relu.forward(img)

    # Pooling Layer
    # Input: (28, 28, 6) Output: (14, 14, 6)
    S2 = ops.MaxPool(2, 2, "S2")
    img = S2.forward(img)
    print("S2 shape: " + str(img.shape))
    #cv2.imshow("S2", img[:,:,0])

    # Convolution layer
    # Input: (14, 14, 6) Output: (10, 10, 6)
    C3 = ops.Conv2D(16, 5, 1, "C3")
    img = C3.forward(img)
    print("C3 shape: " + str(img.shape))
    #cv2.imshow("C3", img[:,:,0])

    # ReLu Activation
    relu = ops.ReLu()
    img = relu.forward(img)

    S4 = ops.MaxPool(2, 2, "S4")
    img = S4.forward(img)
    print("S4 shape: " + str(img.shape))
    #cv2.imshow("S4", img[:,:,0])

    C5 = ops.Conv2D(120, 5, 1, "C5")
    img = C5.forward(img)
    print("C5 shape: " + str(img.shape))
    #cv2.imshow("C5", img[:,:,0])

    # ReLu Activation
    relu = ops.ReLu()
    img = relu.forward(img)

    F6 = ops.FullyConnected(120, 84, "F6")
    img = F6.forward(img)
    print("F6 shape: " + str(img.shape))
    #cv2.imshow("F6", img.reshape(84, 1))

    F7 = ops.FullyConnected(84, 10, "F6")
    img = F7.forward(img)
    print("F7 shape: " + str(img.shape))

    # Softmax
    softmax = ops.Softmax()
    y_hat = softmax.forward(img)

   

    # Cross Entropy Loss
    y = t_train[i]
    #x = 0.01
    #y_hat = np.array([x, x, x, x, x, .7, x, x, x, x])
    loss = cross_entropy(y, y_hat)
    
    print("Loss: " + str(loss))



    cv2.waitKey(0)
    #break