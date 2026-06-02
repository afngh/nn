# 🧠 MiniFramework: Neural Networks from Scratch

A lightweight, dependency-free (except for optional visualization) neural network framework built entirely from scratch in Python. This library demonstrates the core mathematics, forward propagation, backpropagation, and optimization mechanics that power modern deep learning.

---

## 🚀 Key Features

* **Custom Autograd-like Backpropagation**: Mathematically correct, layer-by-layer backpropagation (`backward`) and parameter update (`update`) routines for weights and biases.
* **Modular OOP Design**: Highly clean `Neuron`, `NeuralLayer`, and `NeuralNetwork` classes that permit arbitrary layers, node sizes, and activations.
* **Vector Math Library (`lib/vectors.py`)**: A completely custom, zero-dependency linear algebra module handling dot products, scalar multiplication, and matrix multiplication.
* **Multiple Activations**: Sigmoid, ReLU, and Tanh functions, coupled with their respective dynamic analytical derivatives.
* **Live Training Visualizer**: Real-time rendering of predictions converging to target outputs over training epochs using `matplotlib`.

---

## 📈 Live XOR Convergence Visualization

Solving the XOR logic gate requires non-linear decision boundaries. The live training visualization demonstrates all four XOR coordinates (`[0,0]`, `[0,1]`, `[1,0]`, and `[1,1]`) converging gracefully and completely to their binary targets (`0` and `1`) over 5000 training epochs:

![XOR Epoch Convergence](./sample/XOR_Prediction_Live_Convergence.png)

---

## 🛠️ Repository Architecture

The codebase is organized into highly focused folders:

```text
├── lib/
│   ├── __init__.py    # Exposes core API classes/functions
│   ├── activation.py  # Sigmoid, ReLU & Tanh activations + derivatives
│   ├── loss.py        # Mean Squared Error (MSE) loss and gradients
│   ├── vectors.py     # Zero-dependency Vector & Matrix Math engine
│   └── neuron.py      # Core classes: Neuron, NeuralLayer, and NeuralNetwork
├── main/
│   ├── and.py         # AND gate solver with live matplotlib visualization
│   ├── or.py          # OR gate solver with live matplotlib visualization
│   └── xor.py         # XOR gate solver with live matplotlib visualization
├── test/
│   └── test.py        # Verification and unit testing script
├── sample/            # Output training visualization PNG plots
└── metadata/          # Mathematical details and documentation files
```

### 1. Vector Math (`lib/vectors.py`)

Provides custom mathematical operations on Python lists, bypassing standard external libraries like NumPy:

* `dot(v1, v2)`: Computes $\vec{v}_1 \cdot \vec{v}_2 = \sum x_i y_i$.
* `mat_mul(m1, m2)`: Computes matrix-matrix products.
* `add(v1, v2)` / `sub(v1, v2)`: Element-wise list addition and subtraction.

### 2. Activations (`lib/activation.py`)

Encapsulates activation functions and their derivatives, critical for scaling gradients backward:

* **Sigmoid**: $f(x) = \frac{1}{1 + e^{-x}}$ with derivative $f'(x) = f(x)(1 - f(x))$.
* **ReLU**: $f(x) = \max(0, x)$ with derivative $f'(x) = 1 \text{ if } x > 0 \text{ else } 0$.
* **Tanh**: $f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ with derivative $f'(x) = 1 - \tanh^2(x)$.

### 3. Core Framework (`lib/neuron.py`)

* **`Neuron`**: Manages weights, bias, activation type, stored inputs, pre-activations ($z$), outputs, and errors. Dynamically updates itself using standard gradient descent:
    $$w_i \leftarrow w_i - \eta \cdot \delta \cdot x_i$$
    $$b \leftarrow b - \eta \cdot \delta$$
* **`NeuralLayer`**: Manages collections of neurons, supports optional custom weights, and handles forward parameter copy safety (preventing shared reference bugs).
* **`NeuralNetwork`**: Coordinates forward pass, backward pass (calculating output and backpropagating hidden errors $\delta$), parameter updates, and interactive graphing.

---

## 📖 Getting Started & Usage

### Running the XOR Solver

To run the XOR dataset solver and view the live convergence animation, execute:

```bash
python main/xor.py
```

### Building Your Own Network

You can assemble and train your own sequential network in just a few lines of code:

```python
from lib import NeuralNetwork, NeuralLayer, sigmoid

# 1. Instantiate network
nn = NeuralNetwork()

# 2. Add layers sequentially (Input size 2, Hidden layer 4, Output layer 1)
nn.add(NeuralLayer(num_neurons=4, layer_size=2, a_type=sigmoid))
nn.add(NeuralLayer(num_neurons=1, layer_size=4, a_type=sigmoid))

# 3. Forward Pass
pred = nn.forward([1.0, 0.5])
print("Prediction:", pred)

# 4. Backward Pass (Target: [1.0])
nn.backward([1.0])

# 5. Gradient Step (Learning Rate: 0.1)
nn.update(0.1)
```

---

## 📐 Mathematical Formulation of Backpropagation

Our framework implements the exact multi-layer backpropagation formulas:

1. **Output Layer Error Term ($\delta_i^L$):**
    $$\delta_i^L = (a_i^L - y_i) \cdot f'(z_i^L)$$
2. **Hidden Layer Error Term ($\delta_j^l$):**
    For hidden layer $l$, error propagates back from the downstream layer $l+1$:
    $$\delta_j^l = \left( \sum_k \delta_k^{l+1} \cdot w_{kj}^{l+1} \right) \cdot f'(z_j^l)$$
3. **Weight and Bias Updates:**
    $$\Delta w_{ij}^l = -\eta \cdot \delta_i^l \cdot a_j^{l-1}$$
    $$\Delta b_i^l = -\eta \cdot \delta_i^l$$
