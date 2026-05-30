from vectors import *
from activation import *
from loss import *

class Neuron():
    def __init__(self,weights,bias,a_type=sigmoid):
        self.weights = weights
        self.bias = bias
        self.a_type = a_type

    def forward(self,input):
        return self.a_type(dot(input,self.weights) + self.bias)

    def learn(self,inputs,y_pred,y_true,learning_rate=0.1):
        gradient = mse_derivative(inputs,y_pred,y_true)
        self.weights = gradient_descent(self.weights,gradient,learning_rate)
