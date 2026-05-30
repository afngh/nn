import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0,x)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu_derivative(x):
    return 1 if x > 0 else 0