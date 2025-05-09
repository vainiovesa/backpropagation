from random import random

class Network:
    def __init__(self, size:list, activation_functions:list):
        self.layers = []
        for i in range(1, len(size)):
            self.layers.append(Layer(size[i], size[i - 1], activation_functions[i - 1]))

    def feedforward(self, inputs:list):
        last = inputs
        for layer in self.layers:
            last = layer.activations(last)
        return last

    def __repr__(self):
        return f"Network {self.layers}"

class Layer:
    def __init__(self, size:int, previous_layer_size:int, activation_function:callable):
        self.neurons = []
        for _ in range(size):
            self.neurons.append(Neuron(previous_layer_size, activation_function))
        self.previous_activations = None

    def activations(self, previous_layer_outputs:list):
        self.previous_activations = []
        for neuron in self.neurons:
            self.previous_activations.append(neuron.activation(previous_layer_outputs))
        return self.previous_activations

    def __repr__(self):
        return f"Layer: {self.neurons}"

class Neuron:
    def __init__(self, previous_layer_neurons, activation_function:callable):
        self.activation_function = activation_function
        self.weights = []
        for _ in range(previous_layer_neurons):
            self.weights.append(random() / 10)
        self.bias = random() / 10
        self.previous_activation = None

    def activation(self, previous_layer_output:list):
        weighted = 0
        for weight, output in zip(self.weights, previous_layer_output):
            weighted += weight * output
        weighted += self.bias
        self.previous_activation = self.activation_function(weighted)
        return self.previous_activation

    def __repr__(self):
        return f"Neuron {[round(weight, 3) for weight in self.weights] + [round(self.bias, 3)]}"
