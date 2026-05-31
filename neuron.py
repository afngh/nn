from time import sleep
from loss import mse
from loss import compute_gradients
from vectors import *
from activation import *
from loss import *

class Neuron():
    def __init__(self,weights,bias,a_type=sigmoid):
        self.weights = weights
        self.bias = bias
        self.a_type = a_type

    def forward(self,input):
        print(f"inputs: {input}")
        print(f"weights: {self.weights}")
        self.z = dot(input,self.weights) + self.bias
        return self.a_type(self.z)

    # def learn(self,inputs,y_pred,y_true,learning_rate=0.1):
    #     gradient = compute_gradients(inputs,y_pred,y_true,self.z)
    #     self.weights = gradient_descent(self.weights,gradient,learning_rate)

class NeuralLayer():
    def __init__(self,num_neurons,layer_size,a_type=sigmoid):
        self.num_neurons = num_neurons
        self.layer_size = layer_size
        self.neurons = [Neuron(weights=list(.5 for _ in range(layer_size)),bias=random.uniform(0,.5),a_type=a_type) for _ in range(num_neurons)]

    def forward(self,input):
        print(f"length of nl: {self.neurons}")
        return [neuron.forward(input) for neuron in self.neurons]

    # def learn(self,inputs,y_true,learning_rate=0.01):
    #     for neuron,idx in zip(self.neurons,range(self.num_neurons)):
    #         y_pred = neuron.forward(inputs)
    #         while(mse(y_pred,y_true) > 0.01):   
    #             gradient = compute_gradients(inputs,y_pred,y_true,neuron.z)
    #             neuron.weights = gradient_descent(neuron.weights,gradient,learning_rate)
    #             y_pred = neuron.forward(inputs)
    #         print(f'Trained Neuron: {idx}')
        
    #     for neuron in self.neurons:
    #         print(neuron.weights)




class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        print("Added Layer")

    def forward(self, inputs):
        output = inputs
        print("Init NN")
        for layer in self.layers:
            output = layer.forward(output)
            print(f"Output of NN Layer: {output}")

        return output