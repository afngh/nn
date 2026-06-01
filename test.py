import neuron
from neuron import NeuralNetwork, NeuralLayer
from loss import mse
from activation import relu
import random

data = [
    [(0,0),0],
    [(0,1),1],
    [(1,0),1],
    [(1,1),0]
]
