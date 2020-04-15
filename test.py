import numpy #as np  
import numpy as np
#import cupy as np
from mnist import mnist
#from ops import ops
from ops import ops
import cv2
from statistics import median

#import cupy as np
import time

import pickle

from networks import FC_2Layer, LeNet5, Conv2_Layer


# Test 2 Layer Fully Connected networks
def test_fc():

    # Load data
    x_train, t_train, x_test, t_test = mnist.load()

    # Normalize data
    X_train = np.array(x_train, dtype=np.float64)
    X_test = np.array(x_test, dtype=np.float64)

    X_train -= int(np.mean(X_train))
    X_train /= int(np.std(X_train))

    X_test -= int(np.mean(X_test))
    X_test /= int(np.std(X_test))


    # Initialize network
    network = FC_2Layer()


    # Train Network
    epochs = 20000
    for e in range(epochs):
        total_loss = 0
        correct = 0
        for i, img in enumerate(X_train):

            ### Forward Pass ###
            y_hat = network.forward(img)

            ### Cross Entropy Loss ###
            y = t_train[i]
            loss, y = ops.cross_entropy(y, y_hat)
            total_loss += loss

            # Calculate Accuracy
            if list(y_hat).index(max(y_hat)) == list(y).index(max(y)):
                correct +=1

            ### Backward Pass ###
            network.backward(y)

            if i % 5000 == 0:
                print("Epoch " + str(e) + ", Sample " + str(i) + ", Loss: " + str(loss)[:9] + ", ACC: " + str(correct/(i+1)))
                #print("time elapsed: " + str(time.time() - start_time))


        print("Epoch " + str(e) + " Loss: " + str(total_loss/len(X_train))[:9] + " Acc: " + str(correct/len(X_train)))


def test_lenet():

    # Get MNIST dataset
    x_train, t_train, x_test, t_test = mnist.load()
    X_train = []
    X_test = []

    # Reshape training and testing images from 784 to 28 by 28
    # Pad images by 2 to get 32 x 32 images. This is done to further center the digits in the images.
    # It coul dbe done within the first Conv layer, but it's faster to do it only once during the data prep.
    for i, img in enumerate(x_train):
        img = img.reshape(28, 28)
        img = numpy.pad(img, 2, mode='constant')
        img = img.reshape(32, 32, 1)
        
        X_train.append(img)

    for i, img in enumerate(x_test):
        img = img.reshape(28, 28)
        img = numpy.pad(img, 2, mode='constant')
        img = img.reshape(32, 32, 1)
        
        X_test.append(img)

    # Turn datasets into nupy arrays
    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)


    # Normalize dataset to have zero mean
    X_train -= int(np.mean(X_train))
    X_train /= int(np.std(X_train))

    X_test -= int(np.mean(X_test))
    X_test /= int(np.std(X_test))

    
    # Initilize network
    # network = LeNet5()
    network = Conv2_Layer()
   


    # Train Network
    epochs = 10
    start_time = time.time()
    for e in range(epochs):
        total_loss = 0
        correct = 0
        for i, img in enumerate(X_train):

            #n = 1 
            #img = X_train[n]
            y = t_train[i]
            #y = t_train[n]

            ### Forward Pass ###
            y_hat = network.forward(img)

            ### Cross Entropy Loss ###
            
            loss, y = ops.cross_entropy(y, y_hat)
            total_loss += loss

            # Calculate Accuracy
            #try:
            if list(y_hat).index(max(y_hat)) == list(y).index(max(y)):
                correct +=1
            

            ### Backward Pass ###
            network.backward(y)

            
            if i % 100 == 0:
                print("Epoch " + str(e) + ", Sample " + str(i) + ", Loss: " + str(loss)[:9] + ", ACC: " + str(correct/(i+1)))
                print("time elapsed: " + str(time.time() - start_time))

                obj = []
                for layer in network.layers:
                    obj.append(layer.extract())

                #with open('weights_file.pkl', 'wb') as handle:
                    #pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    #print("Saved weights...")




        print("Epoch " + str(e) + " Loss: " + str(total_loss/len(X_train))[:9] + " Acc: " + str(correct/len(X_train)))

#test_fc()
test_lenet()