from ops import ops


class FC_2Layer():
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



class LeNet5():
    def __init__(self):
        self.layers = []
        self.layers.append(ops.Conv2D(1, 6, 5, 1, "C1"))
        self.layers.append(ops.ReLu())
        self.layers.append(ops.MaxPool(2, 2, "S2"))
        self.layers.append(ops.Conv2D(6, 16, 5, 1, "C3"))
        self.layers.append(ops.ReLu())
        self.layers.append(ops.MaxPool(2, 2, "S4"))
        self.layers.append(ops.Conv2D(16, 120, 5, 1, "C5"))
        self.layers.append(ops.ReLu())
        self.layers.append(ops.FullyConnected(120, 84, "F6"))
        self.layers.append(ops.FullyConnected(84, 10, "F7"))
        self.layers.append(ops.Softmax())

    def forward(self, activation):
        
        # Forward pass through every layer
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backward(self, dy):
        for layer in list(reversed(self.layers)):
            dy = layer.backward(dy)