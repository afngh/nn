import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import NeuralNetwork, NeuralLayer, relu

# 1. Configuration matching the "detective math" example
input_value = 2.0
w1 = 0.5
w2 = 1.5

# 2. Instantiate Network and Layers
nn = NeuralNetwork()
l1 = NeuralLayer(num_neurons=1, layer_size=1, weights=[w1], a_type=relu)
l2 = NeuralLayer(num_neurons=1, layer_size=1, weights=[w2], a_type=relu)

nn.add(l1)
nn.add(l2)

# Set biases to 0 to match the clean math example exactly
l1.neurons[0].bias = 0.0
l2.neurons[0].bias = 0.0

# 3. Forward Pass
out = nn.forward([input_value])
print(f"Test output (expected [1.5]): {out}")
assert abs(out[0] - 1.5) < 1e-6, "Test failed: output was not 1.5"
print("Test passed successfully!")
