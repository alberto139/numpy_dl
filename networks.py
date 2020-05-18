from ops import ops


class FC_2Layer():
    def __init__(self):
        self.layers = []
        self.learning_rate = 0.01
        #self.layers.append(ops.Conv2D(6, 5, 1, "C1"))
        #self.layers.append(ops.ReLu())
        #self.layers.append(ops.MaxPool(2, 2, "S2"))
        #self.layers.append(ops.Conv2D(16, 5, 1, "C3"))
        #self.layers.append(ops.ReLu())
        #self.layers.append(ops.MaxPool(2, 2, "S4"))
        #self.layers.append(ops.Conv2D(120, 5, 1, "C5")) # Essentially a flatten layer
        #self.layers.append(ops.ReLu())
        self.layers.append(ops.FullyConnected(784, 30, self.learning_rate, "F6"))
        self.layers.append(ops.FullyConnected(30, 30, self.learning_rate, "F6"))
        self.layers.append(ops.FullyConnected(30, 10, self.learning_rate, "F7"))
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



class LeNet5():
    def __init__(self):
        self.layers = []
        self.learning_rate = 0.1
        self.layers.append(ops.Conv2D(1, 6, 5, 1, self.learning_rate, "C1"))
        self.layers.append(ops.ReLu())
        self.layers.append(ops.MaxPool(2, 2, "S2"))
        self.layers.append(ops.Conv2D(6, 16, 5, 1, self.learning_rate, "C3"))
        self.layers.append(ops.ReLu())
        self.layers.append(ops.MaxPool(2, 2, "S4"))
        self.layers.append(ops.Conv2D(16, 120, 5, 1, self.learning_rate, "C5"))
        self.layers.append(ops.ReLu())
        self.layers.append(ops.FullyConnected(120, 84, self.learning_rate, "F6"))
        self.layers.append(ops.FullyConnected(84, 10, self.learning_rate, "F7"))
        self.layers.append(ops.Softmax())

    def forward(self, activation):
        
        # Forward pass through every layer
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backward(self, dy):
        for layer in list(reversed(self.layers)):
            dy = layer.backward(dy)


# Works pretty well 78% after 1 epoch
class Conv2_Layer():
    def __init__(self):
        self.layers = []
        self.learning_rate = 0.0005

        self.layers.append(ops.Conv2D(1, 10, 5, 1, self.learning_rate, "C1"))
        self.layers.append(ops.Sigmoid())
        self.layers.append(ops.MaxPool(2, 2, "S2"))
        self.layers.append(ops.Flatten())
        self.layers.append(ops.FullyConnected(1960, 10, self.learning_rate, "F6"))
        self.layers.append(ops.Softmax())

    def forward(self, activation):
        
        # Forward pass through every layer
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backward(self, dy):
        for layer in list(reversed(self.layers)):
            dy = layer.backward(dy)

# 85.7% ACC after one epoch
class Conv3_Layer():
    def __init__(self):
        self.layers = []
        self.learning_rate = 0.0005

        self.layers.append(ops.Conv2D(1, 10, 5, 1, self.learning_rate, "C1"))
        self.layers.append(ops.ReLu())
        self.layers.append(ops.MaxPool(2, 2, "S2"))
        self.layers.append(ops.Conv2D(10, 20, 5, 1, self.learning_rate, "C3"))
        self.layers.append(ops.ReLu())
        self.layers.append(ops.MaxPool(2, 2, "S4"))
        self.layers.append(ops.Flatten())
        self.layers.append(ops.FullyConnected(500, 10, self.learning_rate, "F5"))
        self.layers.append(ops.Softmax())

    def forward(self, activation):

        # Forward pass through every layer
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backward(self, dy):
        for layer in list(reversed(self.layers)):
            dy = layer.backward(dy)


class Conv4_Layer():
    def __init__(self):
        self.layers = []
        self.learning_rate = 0.0005

        self.layers.append(ops.Conv2D(1, 6, 7, 1, self.learning_rate, "C1"))
        self.layers.append(ops.ReLu())
        self.layers.append(ops.MaxPool(2, 2, "S2"))
        self.layers.append(ops.Conv2D(6, 16, 5, 1, self.learning_rate, "C3"))
        self.layers.append(ops.ReLu())
        self.layers.append(ops.MaxPool(2, 2, "S4"))
        self.layers.append(ops.Conv2D(16, 32, 3, 1, self.learning_rate, "C5"))
        self.layers.append(ops.ReLu())
        self.layers.append(ops.MaxPool(2, 2, "S4"))
        self.layers.append(ops.Flatten())
        self.layers.append(ops.FullyConnected(32, 10, self.learning_rate, "F5"))
        self.layers.append(ops.Softmax())

    def forward(self, activation):
        
        # Forward pass through every layer
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backward(self, dy):
        for layer in list(reversed(self.layers)):
            dy = layer.backward(dy)