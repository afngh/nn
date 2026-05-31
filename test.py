from loss import compute_gradients
from vectors import dot
from activation import *
from time import sleep
from neuron import Neuron
from loss import *
from neuron import NeuralLayer

inputs = [5,5]

y_true = 15

ln = NeuralLayer(num_neurons=3,layer_size=2,a_type=relu)

ln.learn(inputs,y_true,learning_rate=0.1)

for neuron in ln.neurons:
    print(f"pred: {neuron.forward([100,200])}")