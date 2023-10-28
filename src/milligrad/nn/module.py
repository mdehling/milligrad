__all__ = ['Module', 'Sequential', 'Linear', 'ReLU', 'Sigmoid']

from milligrad.tensor import Tensor
from itertools import chain

import numpy as np


class Module:

    def __call__(self, input):
        if not isinstance(input, Tensor):
            input = Tensor(input, _label='input')
        return self.forward(input)

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p._zero_grad()


class Sequential(Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

    def parameters(self):
        return list(chain.from_iterable(layer.parameters() for layer in self.layers))


class Linear(Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = Tensor(np.random.randn(in_features,out_features), requires_grad=True, _label='weights')
        self.biases = Tensor(np.zeros((1,out_features)), requires_grad=True, _label='biases')

    def forward(self, input):
        return input @ self.weights + self.biases

    def parameters(self):
        return [self.weights, self.biases]


class ReLU(Module):

    def forward(self, input):
        return input.relu()


class Sigmoid(Module):

    def forward(self, input):
        return input.sigmoid()
