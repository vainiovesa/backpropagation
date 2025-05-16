from math import exp


def relu(x):
    return x if x > 0 else 0

def relu_derivative(x):
    return 1 if x > 0 else 0

def lrelu(x):
    return x if x > 0 else x / 1000

def lrelu_derivative(x):
    return 1 if x > 0 else 1 / 1000

def sigmoid(x):
    return 1 / (1 + exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

def tanh_derivative(x):
    return 1 - tanh(x) ** 2

KNOWN = {
    "relu":    [relu, relu_derivative],
    "lrelu":   [lrelu, lrelu_derivative],
    "sigmoid": [sigmoid, sigmoid_derivative],
    "tanh":    [tanh, tanh_derivative]
}