from loss import mse
from activation import sigmoid_derivative
from vectors import dot
from activation import * 
from loss import *
import random

class Neuron():
    def __init__(self,input_size=2,_label=None):
        self.weights = [random.uniform(.5,1) for _ in range(input_size)]
        self.bias = 0.
        self.output = 0.
        self._label = _label
        self.error = 0.
        self.grad = []
        print(f"Neuron {self._label + 1}")

    def forward(self,inputs):
        self.inputs = inputs
        self.z = dot(inputs,self.weights) + self.bias
        self.output = sigmoid(self.z)

        return self.output

class NeuralLayer():
    def __init__(self,num_neurons,layer_size,weights=None,a_type=sigmoid,_label=0):
        self._label = _label
        print(f"Layer {self._label + 1}")
        self.num_neurons = num_neurons
        self.layer_size = layer_size
        self.a_type = a_type
        self.neurons = [Neuron(layer_size,i) for i in range(num_neurons)]
    def forward(self,inputs):
        self.output = [neuron.forward(inputs) for neuron in self.neurons]
        return self.output

class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add(self,layer):
        self.layers.append(layer)
        self.output_layer = layer

    def forward(self,inputs):
        self.output = inputs
        for layer in self.layers:
            self.output = layer.forward(self.output)
        return self.output

    def backward(self,expected_output):
        for neuron in self.output_layer.neurons:
            neuron.error = expected_output - neuron.output
            neuron.grad = compute_gradients(neuron.inputs,neuron.output,expected_output,neuron.z)

        for i in range(len(self.layers)-1,-1,-1):
            prev_layer = self.layers[i-1].neurons
            current_layer = self.layers[i].neurons
            for prev_neuron,j in zip(prev_layer,range(len(prev_layer))):
                prev_neuron.error = sum([current_neuron.error * current_neuron.weights[j] for current_neuron in current_layer])
                prev_neuron.grad = compute_gradients(prev_neuron.inputs,prev_neuron.output,prev_neuron.error,prev_neuron.z)
    
    def update_weights(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.weights = gradient_descent(neuron.weights,neuron.grad,learning_rate=0.01)
                neuron.bias = neuron.bias - 0.01 * sum([neuron.grad[0] * x for x in neuron.inputs])
    
    def neruon_data(self):
        for layer,i in zip(self.layers,range(len(self.layers))):
            print(f"---------------------Layer(Index={i})------------------------------")
            for neuron,j in zip(layer.neurons,range(len(layer.neurons))):
                print(f"[{i}{j}] Neuron(Error={neuron.error:.4f}, Grad={neuron.grad}, Bias={neuron.bias}, Weights={neuron.weights})")




if __name__ == "__main__":
    data = [
    [(0,0),0],
    [(0,1),1],
    [(1,0),1],
    [(1,1),0]
    ]

    inputs = [1,1]

    target = 0

    nn = NeuralNetwork()
    nn.add(NeuralLayer(num_neurons=4,layer_size=2, _label=0))
    nn.add(NeuralLayer(num_neurons=2,layer_size=4, _label=1))
    nn.add(NeuralLayer(num_neurons=1,layer_size=2, _label=2))


    output = nn.forward(inputs)
    print(output,target)
    prev_loss = mse(target,output[0])
    print(f"loss: {prev_loss}")

    nn.neruon_data()
    nn.backward(target)
    print("After:")
    nn.neruon_data()
    print("After weights update:")
    nn.update_weights()
    nn.neruon_data()

    output = nn.forward(inputs)
    print(f"prev_loss: {prev_loss}")
    print(f"curr_loss: {mse(target,output[0])}")