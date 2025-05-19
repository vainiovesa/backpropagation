from math import sqrt
from random import random, uniform
from network import Layer

def basic(layer:Layer):
    for neuron in layer.neurons:
        new = [random() - 0.5 for _ in range(len(neuron.weights))]
        neuron.set_weights_and_bias(new, 0)

def glorot(layer:Layer):
    fan_in = len(layer.neurons[0].weights)
    fan_out = len(layer.neurons)
    boundary = sqrt(6) / sqrt(fan_in + fan_out)
    for neuron in layer.neurons:
        new = [uniform(-boundary, boundary) for _ in range(len(neuron.weights))]
        neuron.set_weights_and_bias(new, 0)
