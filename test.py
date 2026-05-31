import neuron
from neuron import NeuralNetwork, NeuralLayer
from loss import mse
from activation import relu
import random

data = [
    ([5, 10], 0),  # 5 - 10 = -5 -> 0
    ([2, 5],  0),  # 2 - 5  = -3 -> 0
    ([1, 1],  0),  # 1 - 1  = 0  -> 0
    ([6, 4],  2),  # 6 - 4  = 2  -> 2
    ([10, 3], 7),  # 10 - 3 = 7  -> 7
    ([12, 2], 10)  # 12 - 2 = 10 -> 10
]

weights = [random.uniform(.5,1) for _ in range(2)]

nn = NeuralNetwork()
l1 = NeuralLayer(num_neurons=5, layer_size=2, a_type=relu)
l2 = NeuralLayer(num_neurons=3, layer_size=5, a_type=relu)
l3 = NeuralLayer(num_neurons=2, layer_size=3, a_type=relu)
l4 = NeuralLayer(num_neurons=1, layer_size=2, a_type=relu)

nn.add(l1)
nn.add(l2)
nn.add(l3)
nn.add(l4)

l1.neurons[0].bias = 0.0
l2.neurons[0].bias = 0.0
l3.neurons[0].bias = 0.0
l4.neurons[0].bias = 0.0

print("Initial Network State:")
print(f"Neuron A (l1) - weights: {l1.neurons[0].weights}, bias: {l1.neurons[0].bias}")
print(f"Neuron B (l2) - weights: {l2.neurons[0].weights}, bias: {l2.neurons[0].bias}")
print(f"Neuron C (l3) - weights: {l3.neurons[0].weights}, bias: {l3.neurons[0].bias}")
print(f"Neuron D (l3) - weights: {l4.neurons[0].weights}, bias: {l4.neurons[0].bias}")

learning_rate = 0.005
epochs = 1000

for epoch in range(1, epochs + 1):
    for input_value, output_value in data:
        out = nn.forward(input_value)
        y_pred = out[0]
        
        loss = mse(y_pred, output_value)
        
        print(f"Epoch {epoch:02d}: pred = {y_pred:.6f}, loss = {loss:.6f}")
        
        nn.backward([output_value])
        
        nn.update(learning_rate)

print("\nFinal Network State after Training:")
print(f"  Neuron A (l1) - weights: {[round(w, 4) for w in l1.neurons[0].weights]}, bias: {round(l1.neurons[0].bias, 4)}")
print(f"  Neuron B (l2) - weights: {[round(w, 4) for w in l2.neurons[0].weights]}, bias: {round(l2.neurons[0].bias, 4)}")
print(f"  Neuron C (l3) - weights: {[round(w, 4) for w in l3.neurons[0].weights]}, bias: {round(l3.neurons[0].bias, 4)}")
print(f"  Neuron D (l4) - weights: {[round(w, 4) for w in l4.neurons[0].weights]}, bias: {round(l4.neurons[0].bias, 4)}")



n = neuron.Neuron(l4.neurons[0].weights,l4.neurons[0].bias,a_type=relu)


while(True):
    ip = list(map(int,input('Enter Inputs: ').split(' ')))
    print(n.forward(ip))