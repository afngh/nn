from vectors import dot
from activation import *

inputs = [1,2]
weights = [0.5,0.3]
bias = 0.1

output = dot(inputs,weights) + bias

print(sigmoid(output))
print(relu(output))
print(sigmoid_derivative(output))
print(relu_derivative(output))