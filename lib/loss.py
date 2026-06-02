from .activation import sigmoid_derivative
from .activation import relu_derivative
    

#returns loss
def mse(y_pred,y_true):
    return ((y_pred-y_true) ** 2)

#returns gradient
def compute_gradients(input,y_pred,y_true,z):
    return [(y_pred - y_true) * x * sigmoid_derivative(z) for x in input]

#updates weights for learning
def gradient_descent(weight,gradient,learning_rate=0.01):
    return [w - learning_rate * g for w,g in zip(weight,gradient)]