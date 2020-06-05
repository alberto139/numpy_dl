import numpy as np
from mnist import mnist

def load_mnist(shape = 'square'):
    # Get MNIST dataset
    x_train, t_train, x_test, t_test = mnist.load()
    X_train = []
    X_test = []

    # Reshape training and testing images from 784 to 28 by 28
    # Pad images by 2 to get 32 x 32 images. This is done to further center the digits in the images.
    # It coul dbe done within the first Conv layer, but it's faster to do it only once during the data prep.
    
    
    if shape == 'square':
        for i, img in enumerate(x_train):

            if shape == 'square':
                img = img.reshape(28, 28)
                img = np.pad(img, 2, mode='constant')
                img = img.reshape(32, 32, 1)
            
            X_train.append(img)

        for i, img in enumerate(x_test):
            
            img = img.reshape(28, 28)
            img = np.pad(img, 2, mode='constant')
            img = img.reshape(32, 32, 1)
            
            X_test.append(img)

    else:
        X_train = x_train
        X_test = x_test

    # Turn datasets into nupy arrays
    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)


    # Normalize dataset to have zero mean
    X_train -= int(np.mean(X_train))
    X_train /= int(np.std(X_train))

    X_test -= int(np.mean(X_test))
    X_test /= int(np.std(X_test))

    return X_train, X_test, t_train, t_test