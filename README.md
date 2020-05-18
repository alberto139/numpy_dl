# numpy_dl

This is an attempt at implementing a Convolutional Neural Network using onlyu Python and Numpy. The goal is to create a neural network similar to LeNet-5 as described in [Yan LeCun's paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) and with a similar performance on the MNIST dataset (around 1% error).


![LeNet](/imgs/LeNet_architecture.png)

# Known Issues

1. Network does not learn with Sigmoid activation
2. Exploding Gradients with ReLu activation

