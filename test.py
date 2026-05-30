from vectors import dot
from activation import *

from neuron import Neuron

inputs = [1,2]
weights = [0.5,0.3]
bias = 0.1

n = Neuron(weights,bias,a_type=sigmoid_derivative)

print(n.forward(inputs))