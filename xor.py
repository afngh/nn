import matplotlib.pyplot as plt
from neuron import NeuralNetwork, NeuralLayer
from loss import mse
from activation import sigmoid

data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

nn = NeuralNetwork()
nn.add(NeuralLayer(num_neurons=4, layer_size=2, a_type=sigmoid))
nn.add(NeuralLayer(num_neurons=2, layer_size=4, a_type=sigmoid))
nn.add(NeuralLayer(num_neurons=1, layer_size=2, a_type=sigmoid))

print("Initial Predictions:")
for inputs, target in data:
    pred = nn.forward(inputs)[0]
    print(f"  Input: {inputs} -> Target: {target} | Prediction: {pred:.4f}")

yts = [0,1,1,0]
learning_rate = 0.5
epochs = 5000
for epoch in range(epochs):
    total_loss = 0
    xps = []
    for inputs, target in data:
        out = nn.forward(inputs)
        pred = out[0]
        
        total_loss += mse(pred, target)
        xps.append(pred)
        nn.backward([target])
        
        nn.update(learning_rate)
    nn.data_graph(xps, yts)
        
    if epoch % 500 == 0 or epoch == 1:
        print(f"Epoch {epoch:04d}: Mean Loss = {total_loss / 4:.6f}")
print("\nFinal Predictions:")
for inputs, target in data:
    pred = nn.forward(inputs)[0]
    print(f"input: {inputs}->prediction: {pred:.4f}")

# Keep window open after training finishes
plt.ioff()
plt.show()