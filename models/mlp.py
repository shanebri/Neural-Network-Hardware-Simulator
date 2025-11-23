import torch.nn as nn
import torch.nn.functional as F


#simple 4 layer (2 hidden) multilayer perceptron
class SimpleMLP(nn.Module):
    """
    simple feedforward neural network

    architecture (for MNIST style inputs):
        input: 784-dimensional vector (28x28 flatted image)
        hidden layer 1: 128 neurons
        hidden layer 2: 64 neurons
        output layer: 10 logits (e.g. digits 0-9)

    each layer performs: y= xW^T + b
        x is the input vector (or batch of vectors)
        W is the weight matrix
        b is the bias vector
    followed by rectified linear unit activation function in the hidden layers.
    """

    def __init__(self, input_dim=784, hidden1=128, hidden2=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)


    """
    defines the forward pass of the nn:
    x -> fc1 -> ReLU -> fc2 -> ReLU -> fc3
    """
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x