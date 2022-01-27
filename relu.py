import numpy as np


# ReLU activation
class Activation_ReLU:
    # Forward pass
    def __init__(self):
        self.inputs = None
        self.dinputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
