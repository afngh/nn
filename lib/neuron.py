from time import sleep
import random
from loss import mse
from loss import compute_gradients
from vectors import *
from activation import *
from loss import *
import matplotlib.pyplot as plt

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
            elif a_type == tanh:
                self.a_deriv = tanh_derivative
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
        self.pred_history = []

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

    def data_graph(self, pred, target,title="Data Visualization"):
        self.pred_history.append(pred)
        epoch = len(self.pred_history)
        
        # Throttle updates to maintain high training speed (update every 25 epochs)
        if epoch % 25 != 0 and epoch != 1 and epoch != 5000:
            return
            
        if epoch == 1:
            plt.ion()
            plt.figure(title, figsize=(10, 6))
            
        plt.clf()
        
        epochs_range = list(range(1, epoch + 1))
        p00 = [p[0] for p in self.pred_history]
        p01 = [p[1] for p in self.pred_history]
        p10 = [p[2] for p in self.pred_history]
        p11 = [p[3] for p in self.pred_history]
        
        # Vibrant modern styling
        plt.plot(epochs_range, p00, label="[0,0] => 0", color="#e74c3c", linewidth=2.5)
        plt.plot(epochs_range, p01, label="[0,1] => 1", color="#2ecc71", linewidth=2.5)
        plt.plot(epochs_range, p10, label="[1,0] => 1", color="#3498db", linewidth=2.5)
        plt.plot(epochs_range, p11, label="[1,1] => 1", color="#f1c40f", linewidth=2.5)
        
        # Helper target markers
        plt.axhline(y=0, color="#95a5a6", linestyle="--", alpha=0.7, label="Target 0")
        plt.axhline(y=1, color="#2c3e50", linestyle="--", alpha=0.7, label="Target 1")
        
        plt.title(f"{title} (Epoch {epoch})", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Prediction Value", fontsize=12)
        plt.ylim(-0.1, 1.1)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
        
        plt.draw()
        plt.pause(0.001)
            