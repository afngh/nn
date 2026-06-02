# Neuron Class (`neuron.py`)

The `Neuron` class represents a single artificial neuron (perceptron). It encapsulates a set of weights, a bias, and an activation function, and provides a method to perform the forward pass (feedforward operation).

---

## Table of Contents
- [Class: Neuron](#class-neuron)
  - [Initialization: `__init__`](#initialization-__init__)
  - [Method: `forward`](#method-forward)
- [How It Works](#how-it-works)
- [Usage Example](#usage-example)

---

## Class: Neuron

```python
class Neuron()
```

### Initialization: `__init__`
Initializes a new instance of the `Neuron` class.

```python
def __init__(self, weights: list[float], bias: float, a_type=sigmoid)
```
- **Parameters:**
  - `weights`: A list of floats representing the connection weights of the neuron.
  - `bias`: A float representing the bias of the neuron.
  - `a_type`: A callable activation function to apply to the pre-activation sum (defaults to the `sigmoid` function from the `activation` module).

### Method: `forward`
Performs the feedforward calculation for a given input.

```python
def forward(self, input: list[float]) -> float
```
- **Parameters:**
  - `input`: A list of floats representing the inputs to the neuron.
- **Returns:** The activated output of the neuron (a float).
- **Raises:** `ValueError` if the `input` vector is not of the same length as the `weights` vector (inherited from `vectors.dot`).

---

## How It Works

The forward pass of the neuron is calculated as follows:

1. **Pre-activation Sum ($z$):**
   Calculates the dot product of the input vector $\vec{x}$ and the weight vector $\vec{w}$, then adds the bias $b$.
   $$z = (\vec{x} \cdot \vec{w}) + b = \sum_{i=1}^{n} (x_i \cdot w_i) + b$$

2. **Activation ($a$):**
   Applies the chosen activation function $f$ (e.g., Sigmoid or ReLU) to the pre-activation sum.
   $$a = f(z)$$

---

## Usage Example

Below is an example showing how to initialize a neuron and run a forward pass with different activation functions:

```python
from neuron import Neuron
from activation import sigmoid, relu, sigmoid_derivative

# Inputs to the neuron
inputs = [1.0, 2.0]

# Neuron parameters
weights = [0.5, 0.3]
bias = 0.1

# 1. Initialize a Neuron using default Sigmoid activation
neuron_sigmoid = Neuron(weights, bias, a_type=sigmoid)
output_sigmoid = neuron_sigmoid.forward(inputs)
print(f"Sigmoid Output: {output_sigmoid:.4f}")  # Output: ~0.7685

# 2. Initialize a Neuron using ReLU activation
neuron_relu = Neuron(weights, bias, a_type=relu)
output_relu = neuron_relu.forward(inputs)
print(f"ReLU Output: {output_relu:.4f}")        # Output: 1.2000

# 3. Initialize a Neuron using Sigmoid Derivative
neuron_deriv = Neuron(weights, bias, a_type=sigmoid_derivative)
output_deriv = neuron_deriv.forward(inputs)
print(f"Sigmoid Derivative Output: {output_deriv:.4f}")  # Output: ~0.1779
```
