import numpy as np


# ReLU activation
class Activation_ReLU:
    # Forward pass
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
