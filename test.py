import numpy #as np  
import numpy as np
#import cupy as np
from mnist import mnist
#from ops import ops
from ops import ops
import cv2
from statistics import median
import utils

import visualizations as vis

#import cupy as np
import time
import pickle

from networks import FC_2Layer, LeNet5, Conv2_Layer, Conv3_Layer, Conv4_Layer


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
    epochs = 20
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


def train_lenet():

    X_train, X_test, t_train, t_test = utils.load_mnist('square')
    
    # Initilize network
    #network = LeNet5()
    #network = Conv2_Layer()
    network = Conv3_Layer()
    #return
    #network = Conv4_Layer()
   
    # Train Network
    epochs = 20
    batch_size = 100

    start_time = time.time()
    for e in range(epochs):

        total_loss = 0
        correct = 0
        correct_this_batch = 0

        for i, img in enumerate(X_train):

            y = t_train[i]
  

            ### Forward Pass ###
            y_hat = network.forward(img)

            ### Cross Entropy Loss ###
            
            loss, y = ops.cross_entropy(y, y_hat)
            total_loss += loss

            # Calculate Accuracy
            if list(y_hat).index(max(y_hat)) == list(y).index(max(y)):
                correct +=1
                correct_this_batch += 1
            

            ### Backward Pass ###
            network.backward(y)

            
            if i % batch_size == 0:
                print("Epoch " + str(e) + ", Sample " + str(i) + ", Loss: " + str(loss)[:6] + ", ACC: " + str(correct/(i+1))[0:6] + ", Batch ACC: " + str(correct_this_batch/batch_size))
                print("Learning Rate: " + str(network.learning_rate))
                print("time elapsed: " + str(time.time() - start_time))
                correct_this_batch = 0
                
                #if i % 1000 == 0:    
                #    network.learning_rate = network.learning_rate * 0.98
                    

                obj = []
                for layer in network.layers:
                    obj.append(layer.extract())

                with open('weights_file_lenet_sigmoid.pkl', 'wb') as handle:
                    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print("Saved weights...")




        print("Epoch " + str(e) + " Loss: " + str(total_loss/len(X_train))[:9] + " Acc: " + str(correct/len(X_train)))

def load_weights(weights_file, network):
    weights = open(weights_file, 'rb')
    weights = pickle.load(weights)

    w_dict = {}
    for weight in weights:
        if weight:
            print('---')
            for key, val in weight.items():
                print(key)
                ln, wn = key.split(".")
                if ln in w_dict:
                    w_dict[ln][wn] = val
                else:
                    w_dict[ln] = {wn : val}

    for layer in network.layers:
        if layer.name in w_dict:
            layer.feed(w_dict[layer.name]['weights'] , w_dict[layer.name]['bias'] )



    return network


def test_pretrained():
    

    x_train, x_test, t_train, t_test = load_mnist()

    #weights_file = 'weights_file_conv3.pkl'
    weights_file = 'weights_file_lenet.pkl'
    #network = Conv3_Layer()
    network = LeNet5()

    network = load_weights(weights_file, network)

    correct = 0
    for i, img in enumerate(x_test):
        #print(img.shape)
        y_hat = network.forward(img)
        y = t_test[i]
        #print(list(y_hat).index(max(y_hat)))
        #cv2.imshow("img", img)
        #cv2.waitKey(0)

        if list(y_hat).index(max(y_hat)) == y:
            correct +=1
            print(correct / (i+1))




def vis_demo():
    x_train, x_test, t_train, t_test = load_mnist()
    vis.maxpool(x_train[0])


#vis_demo()
#test_fc()
#test_lenet()

#test_pretrained()
train_lenet()