# numpy_dl

This is an attempt at implementing a Convolutional Neural Network using onlyu Python and Numpy. The goal is to create a neural network similar to LeNet-5 as described in [Yan LeCun's paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) and with a similar performance on the MNIST dataset (around 1% error).

<p align="center">
<img src="/imgs/LeNet_architecture.png"/>
</p>

## Operations

A number of nueral network operations have been implemented in the `ops.py` folder to be used as a library for building neural networks. These include:
- 2D Convolution
- Max Pooling
- Fully Connected 
- Flattening 
- ReLu Activation
- Sigmoid Activation
- Softmax
- Cross Entropy Loss

## Networks
While the main goal of this project is to implement a LeNet-5 type architecture, a couple of simpler networks have also been implemented as stepping stones to help test some of the operations before building the full LeNet-5 architecture. These networks are included in the `networks.py` file and include a 2 layer Fully Connected network (with not Convolutions), and a few Convolutions Neural Networks.


## Visualizations
To help understand some of the neural network operations, some visualizations in the form of gifs have been provided. Note that some of the operations had to be simplified in order to make the visualizations, such as only applying certain operations on 2 dimensions instead of 3. 

### Max Pooling
<p align="center">
  <img src="https://media.giphy.com/media/dWBolKR8du17FSjwIt/giphy.gif">
</p>

<p align="center">
Left: Image on which max pooling is being performed with the sliding 3x3 kernel visualized as a white box. <br> Center: Receptive field of the kernel. <br> Right: Result
</p>

## Quirks and Known Issues
This project does not aim to be a faithful re-implementation of the LeNet-5 architecture as described in the paper so there are a few differences:
- The padding of the MNIST images is done in the data pre-processing stage and not by the first layer of the network.
- A cross-entropy loss function is used instead of the one described in the paper.
- ReLu activation is used instead of Sigmoid.
- Weight initialization is different.

This project currently has a few issues that might be caused by the changes mentioned above. 

1. Network does not learn with Sigmoid activation
2. Exploding Gradients with ReLu activation
3. Very different weight sizes between layers (might be normal?)

If you have any recommendations or insight as the the cause of these issues, please feel free to submit a pull request or issue. Thanks!


