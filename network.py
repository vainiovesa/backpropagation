from random import random

class Network:
    def __init__(self, size:list, activation_functions:list,
                 activation_function_derivatives:list):
        self.layers = []
        for i in range(1, len(size)):
            self.layers.append(Layer(size[i], size[i - 1], activation_functions[i - 1],
                                     activation_function_derivatives[i - 1]))

    def feedforward(self, inputs:list):
        last = inputs
        for layer in self.layers:
            last = layer.activations(last)
        return last

    def __repr__(self):
        return f"Network {self.layers}"

class Layer:
    def __init__(self, size:int, previous_layer_size:int, activation_function:callable,
                 activation_function_derivative:callable):
        self.neurons = []
        for _ in range(size):
            self.neurons.append(Neuron(previous_layer_size, activation_function,
                                       activation_function_derivative))
        self.previous_activations = None

    def activations(self, previous_layer_outputs:list):
        self.previous_activations = []
        for neuron in self.neurons:
            self.previous_activations.append(neuron.activation(previous_layer_outputs))
        return self.previous_activations

    def __repr__(self):
        return f"Layer: {self.neurons}"

class Neuron:
    def __init__(self, previous_layer_neurons, activation_function:callable,
                 activation_function_derivative:callable):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.weights = []
        for _ in range(previous_layer_neurons):
            self.weights.append(random() / 10)
        self.bias = random() / 10
        self.derivative_of_cost = None
        self.previous_weighted = None
        self.previous_activation = None

    def activation(self, previous_layer_output:list):
        self.previous_weighted = 0
        for weight, output in zip(self.weights, previous_layer_output):
            self.previous_weighted += weight * output
        self.previous_weighted += self.bias
        self.previous_activation = self.activation_function(self.previous_weighted)
        return self.previous_activation

    def previous_layer_derivatives(self, expected:float):
        derivatives = []
        part = self.activation_function_derivative(self.previous_weighted)
        part *= 2 * (self.previous_activation - expected)
        for weight in self.weights:
            derivatives.append(weight * part)
        return derivatives

    def weights_derivatives(self, previous_layer_activations:list, expected:float):
        derivatives = []
        part = self.activation_function_derivative(self.previous_weighted)
        part *= 2 * (self.previous_activation - expected)
        for activation in previous_layer_activations:
            derivatives.append(activation * part)
        return derivatives

    def bias_derivative(self, expected:float):
        part = self.activation_function_derivative(self.previous_weighted)
        part *= 2 * (self.previous_activation - expected)
        return part

    def update_derivative_of_cost(self, new:float):
        self.derivative_of_cost = new

    def update_weights_and_bias(self, delta_weights:list, delta_bias:float):
        updated_weights = []
        for old, new in zip(self.weights, delta_weights):
            updated_weights.append(old + new)
        self.weights = updated_weights
        self.bias += delta_bias

    def reset_all(self):
        self.previous_activation = None
        self.derivative_of_cost = None
        self.previous_weighted = None

    def __repr__(self):
        weights = [round(weight, 3) for weight in self.weights]
        bias = [round(self.bias, 3)]
        return f"Neuron {weights + bias}"
