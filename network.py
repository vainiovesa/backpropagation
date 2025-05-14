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

    def reverse(self, inputs:list, expected_outputs:list, learning_rate:float):
        if len(self.layers) == 1:
            self.one_layer_reverse(inputs, expected_outputs, learning_rate)
        else:
            self.outputlayer_reverse(expected_outputs, learning_rate)
            self.hidden_layers_reverse(learning_rate)
            self.first_layer_reverse(inputs, learning_rate)

    def loss(self, expected_output:list):
        error = 0
        output = self.layers[-1].previous_activations
        for output, expected in zip(output, expected_output):
            error += (output - expected) ** 2
        n = len(expected_output)
        return error / n

    def one_layer_reverse(self, inputs:list, expected_outputs:list, learning_rate:float):
        layer = self.layers[0]
        layer.update_weights_and_bias(inputs, expected_outputs, learning_rate)
        layer.reset_all()

    def outputlayer_reverse(self, expected_outputs:list, learning_rate:float):
        layer = self.layers[-1]
        derivatives = layer.previous_layer_derivatives(expected_outputs)
        last_layer = self.layers[-2]
        last_layer.set_derivatives_of_cost(derivatives)
        last_layer_activations = last_layer.get_previous_activations()
        layer.update_weights_and_bias(last_layer_activations, expected_outputs, learning_rate)
        layer.reset_all()

    def hidden_layers_reverse(self, learning_rate:float):
        for i in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[i]
            expected = layer.get_expected()
            derivatives = layer.previous_layer_derivatives(expected)
            last_layer = self.layers[i - 1]
            last_layer_activations = last_layer.get_previous_activations()
            last_layer.set_derivatives_of_cost(derivatives)
            layer.update_weights_and_bias(last_layer_activations, expected, learning_rate)
            layer.reset_all()

    def first_layer_reverse(self, inputs:list, learning_rate:float):
        layer = self.layers[0]
        last_layer_activations = inputs
        expected = layer.get_expected()
        layer.update_weights_and_bias(last_layer_activations, expected, learning_rate)
        layer.reset_all()

    def __repr__(self):
        return f"Network {self.layers}"

class Layer:
    def __init__(self, size:int, previous_layer_size:int, activation_function:callable,
                 activation_function_derivative:callable):
        self.neurons = []
        for _ in range(size):
            self.neurons.append(Neuron(previous_layer_size, activation_function,
                                       activation_function_derivative))
        self.activation_function = activation_function
        self.activation_function_erivative = activation_function_derivative
        self.previous_activations = None

    def activations(self, previous_layer_outputs:list):
        self.previous_activations = []
        for neuron in self.neurons:
            self.previous_activations.append(neuron.activation(previous_layer_outputs))
        return self.previous_activations

    def get_previous_activations(self):
        return self.previous_activations

    def get_expected(self):
        expected = []
        for neuron in self.neurons:
            expected.append(neuron.previous_activation - neuron.derivative_of_cost)
        return expected

    def set_derivatives_of_cost(self, derivatives:list):
        for neuron, derivative in zip(self.neurons, derivatives):
            neuron.set_derivative_of_cost(derivative)

    def previous_layer_derivatives(self, expected:list):
        derivatives_by_neuron = []
        for value, neuron in zip(expected, self.neurons):
            derivatives_by_neuron.append(neuron.previous_layer_derivatives(value))
        derivatives = []
        for i in range(len(derivatives_by_neuron[0])):
            derivatives.append(0)
            for j in range(len(derivatives_by_neuron)):
                derivatives[i] += derivatives_by_neuron[j][i]
        return derivatives

    def update_weights_and_bias(self, previous_layer_activations:list, expected:list, learning_rate:float):
        for neuron, value in zip(self.neurons, expected):
            weight_derivatives = neuron.weights_derivatives(previous_layer_activations, value)
            bias_derivative = neuron.bias_derivative(value)
            neuron.update_weights_and_bias(weight_derivatives, bias_derivative, learning_rate)

    def reset_all(self):
        for neuron in self.neurons:
            neuron.reset_all()

    def __repr__(self):
        return f"Layer: {self.neurons}"

class Neuron:
    def __init__(self, previous_layer_neurons, activation_function:callable,
                 activation_function_derivative:callable):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.weights = []
        for _ in range(previous_layer_neurons):
            self.weights.append(random() - 0.5)
        self.bias = random() - 0.5
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
            derivatives.append(- activation * part)
        return derivatives

    def bias_derivative(self, expected:float):
        part = self.activation_function_derivative(self.previous_weighted)
        part *= 2 * (self.previous_activation - expected)
        return - part

    def update_derivative_of_cost(self, new:float):
        self.derivative_of_cost = new

    def update_weights_and_bias(self, delta_weights:list, delta_bias:float, learning_rate:float):
        updated_weights = []
        for old, new in zip(self.weights, delta_weights):
            updated_weights.append(old + new * learning_rate)
        self.weights = updated_weights
        self.bias += delta_bias * learning_rate

    def reset_all(self):
        self.previous_activation = None
        self.derivative_of_cost = None
        self.previous_weighted = None

    def set_derivative_of_cost(self, derivative:float):
        self.derivative_of_cost = derivative

    def set_weights_and_bias(self, weights:list, bias:float):
        """Method for testing"""
        self.weights = weights
        self.bias = bias

    def __repr__(self):
        weights = [round(weight, 3) for weight in self.weights]
        bias = [round(self.bias, 3)]
        return f"Neuron {weights + bias}"
