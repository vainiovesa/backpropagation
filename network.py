from random import random
from activation_functions import ReLU

class Network:
    def __init__(self, size:list):
        self.layers = []
        for i in range(1, len(size)):
            self.layers.append(Layer(size[i], size[i - 1]))

    def __repr__(self):
        return f"Network {self.layers}"

class Layer:
    def __init__(self, size:int, previous_layer_size:int):
        self.neurons = []
        for _ in range(size):
            self.neurons.append(Neuron(previous_layer_size))
    
    def __repr__(self):
        return f"Layer: {self.neurons}"

class Neuron:
    def __init__(self, previous_layer_neurons):
        self.weights = []
        for _ in range(previous_layer_neurons):
            self.weights.append(random() / 10)
        self.bias = random() / 10

    def __repr__(self):
        return f"Neuron {[round(weight, 3) for weight in self.weights] + [round(self.bias, 3)]}"
