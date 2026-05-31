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
        self.z = dot(input,self.weights) + self.bias
        return self.a_type(self.z)

    def learn(self,inputs,y_pred,y_true,learning_rate=0.1):
        gradient = compute_gradients(inputs,y_pred,y_true,self.z)
        self.weights = gradient_descent(self.weights,gradient,learning_rate)

class NeuralLayer():
    def __init__(self,num_neurons,layer_size,a_type=sigmoid):
        self.num_neurons = num_neurons
        self.layer_size = layer_size
        self.a_type = a_type
        self.neurons = [Neuron([random.uniform(0,1),random.uniform(0,1)],0.01,a_type) for _ in range(num_neurons)]
        for neuron in self.neurons:
            print(neuron.weights)

    def forward(self,input):
        return [neuron.forward(input) for neuron in self.neurons]

    def learn(self,inputs,y_true,learning_rate=0.01):
        for neuron in self.neurons:
            y_pred = neuron.forward(inputs)
            while(mse(y_pred,y_true) > 0.01):   
                gradient = compute_gradients(inputs,y_pred,y_true,neuron.z)
                print(f"gradients: {gradient}")
                neuron.weights = gradient_descent(neuron.weights,gradient,learning_rate)
                y_pred = neuron.forward(inputs)
                print(f"weights: {neuron.weights}")
                # sleep(.3)
        
        for neuron in self.neurons:
            print(neuron.weights)