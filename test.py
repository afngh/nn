from neuron import NeuralNetwork
from loss import compute_gradients
from vectors import dot
from activation import *
from time import sleep
from neuron import Neuron
from loss import *
from neuron import NeuralLayer

inputs = [2,3,5]

nn = NeuralNetwork()

nn.add(NeuralLayer(num_neurons=3,layer_size=3,a_type=relu))
nn.add(NeuralLayer(num_neurons=5,layer_size=3,a_type=relu))
nn.add(NeuralLayer(num_neurons=1,layer_size=5,a_type=relu))

out = nn.forward(inputs=inputs)
print(out)