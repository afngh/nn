# Activation Functions Library (`activation.py`)

A lightweight Python module containing implementations of common activation functions and their derivatives, typically used in building and training artificial neural networks.

---

## Table of Contents

- [Sigmoid Function](#sigmoid-function)
  - [sigmoid](#sigmoid)
  - [sigmoid_derivative](#sigmoid_derivative)
- [ReLU Function (Rectified Linear Unit)](#relu-function-rectified-linear-unit)
  - [relu](#relu)
  - [relu_derivative](#relu_derivative)
- [Usage Example](#usage-example)

---

## Sigmoid Function

The Sigmoid function maps any real-valued number into a range between 0 and 1. It is commonly used as an activation function in binary classification models.

### `sigmoid`

Computes the standard logistic sigmoid activation.
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```python
def sigmoid(x: float) -> float
```

- **Parameters:**
  - `x`: Input scalar value.
- **Returns:** A float value in the range $(0, 1)$.

### `sigmoid_derivative`

Computes the derivative of the sigmoid function with respect to its input.
$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

```python
def sigmoid_derivative(x: float) -> float
```

- **Parameters:**
  - `x`: Input scalar value.
- **Returns:** The derivative value, which has a maximum value of $0.25$ at $x = 0$.

---

## ReLU Function (Rectified Linear Unit)

The Rectified Linear Unit (ReLU) is one of the most widely used activation functions in deep learning. It outputs the input directly if it is positive, and zero otherwise.

### `relu`

Computes the rectified linear unit activation.
$$f(x) = \max(0, x)$$

```python
def relu(x: float) -> float
```

- **Parameters:**
  - `x`: Input scalar value.
- **Returns:** $x$ if $x > 0$, otherwise $0$.

### `relu_derivative`

Computes the derivative of the ReLU function.
$$f'(x) = \begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \le 0
\end{cases}$$

```python
def relu_derivative(x: float) -> float
```
- **Parameters:**
  - `x`: Input scalar value.
- **Returns:** `1` if the input is strictly greater than `0`, otherwise `0`.

---

## Usage Example

Below is a simple demonstration of importing and using these activation functions in conjunction with vector dot products (e.g., in a single neuron calculation):

```python
from vectors import dot
from activation import sigmoid, relu, sigmoid_derivative, relu_derivative

# Inputs and weights for a single artificial neuron
inputs = [1.0, 2.0]
weights = [0.5, 0.3]
bias = 0.1

# Compute the weighted sum (pre-activation)
pre_activation = dot(inputs, weights) + bias  # 1.2

# Apply activation functions
print(f"Sigmoid: {sigmoid(pre_activation):.4f}")                 # Output: ~0.7685
print(f"ReLU: {relu(pre_activation):.4f}")                       # Output: 1.2000
print(f"Sigmoid Derivative: {sigmoid_derivative(pre_activation):.4f}") # Output: ~0.1779
print(f"ReLU Derivative: {relu_derivative(pre_activation)}")      # Output: 1
```
