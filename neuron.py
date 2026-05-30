from vectors import *
from activation import *

class Neuron():
    def __init__(self,weights,bias,a_type=sigmoid):
        self.weights = weights
        self.bias = bias
        self.a_type = a_type

    def forward(self,input):
        return self.a_type(dot(input,self.weights) + self.bias)
