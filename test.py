from vectors import dot
from activation import *
from time import sleep
from neuron import Neuron
from loss import *

inputs = [5,5]
weights = [1,.5]
bias = 0.01

y_true = 10

n = Neuron(weights,bias,a_type=relu)

y_pred = n.forward(input=inputs)

while(mse(y_pred,y_true) > 0.01):
    y_pred = n.forward(input=inputs)
    print(f"loss: {mse(y_pred,y_true)}, prediction {y_pred}, {n.weights}")
    n.learn(inputs,y_pred,y_true)

print(f"final weight {n.weights}, {sum(weights)}")