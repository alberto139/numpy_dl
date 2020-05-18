# numpy_dl

This is an attempt at implementing a Convolutional Neural Network using onlyu Python and Numpy. The goal is to create a neural network similar to LeNet-5 as described in [Yan LeCun's paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) and with a similar performance on the MNIST dataset (around 1% error).

<p align="center">
<img src="/imgs/LeNet_architecture.png"/>
</p>

# Visualizations

## Max Pooling
<p align="center">
  <img src="https://media.giphy.com/media/dWBolKR8du17FSjwIt/giphy.gif">
</p>

<p align="center">
Left: Image on which max pooling is being performed with the sliding 3x3 kernel visualized as a white box. Center: Receptive field of the kernel. Right: Result
</p>

# Known Issues

1. Network does not learn with Sigmoid activation
2. Exploding Gradients with ReLu activation


