from time import sleep
import random
from loss import mse
from loss import compute_gradients
from vectors import *
from activation import *
from loss import *

class Neuron():
    def __init__(self,weights,bias,a_type=sigmoid,a_deriv=None):
        self.weights = weights
        self.bias = bias
        self.a_type = a_type
        if a_deriv is None:
            if a_type == sigmoid:
                self.a_deriv = sigmoid_derivative
            elif a_type == relu:
                self.a_deriv = relu_derivative
            else:
                self.a_deriv = lambda x: 1.0
        else:
            self.a_deriv = a_deriv
        self.error = 0

    def forward(self,input):
        self.input = input
        self.z = dot(input,self.weights) + self.bias
        self.output = self.a_type(self.z)
        return self.output

    def update_parameters(self, learning_rate):
        self.weights = [w - learning_rate * self.error * x for w, x in zip(self.weights, self.input)]
        self.bias -= learning_rate * self.error

class NeuralLayer():
    def __init__(self,num_neurons,layer_size,weights=None,a_type=sigmoid):
        self.num_neurons = num_neurons
        self.layer_size = layer_size
        self.neurons = []
        for _ in range(num_neurons):
            if weights is not None:
                w = list(weights)
            else:
                w = [random.uniform(-1.0, 1.0) for _ in range(layer_size)]
            bias = random.uniform(-0.5, 0.5)
            self.neurons.append(Neuron(weights=w, bias=bias, a_type=a_type))

    def forward(self,input):
        # print(f"length of nl: {self.neurons}")
        self.output = [neuron.forward(input) for neuron in self.neurons]
        return self.output

class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        print("Added Layer")

    def forward(self, inputs):
        output = inputs
        # print("Init NN")
        for layer in self.layers:
            output = layer.forward(output)
            # print(f"Output of NN Layer: {output}")

        return output

    def backward(self, y_true):
        output_layer = self.layers[-1]
        for i, neuron in enumerate(output_layer.neurons):
            neuron.error = (neuron.output - y_true[-1]) * neuron.a_deriv(neuron.z)
        for l in range(len(self.layers) - 2, -1, -1):
            curr_layer = self.layers[l]
            next_layer = self.layers[l+1]
            for j, neuron in enumerate(curr_layer.neurons):
                accumulated_error = sum(next_neuron.error * next_neuron.weights[j] for next_neuron in next_layer.neurons)
                neuron.error = accumulated_error * neuron.a_deriv(neuron.z)

    def update(self, learning_rate):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update_parameters(learning_rate)