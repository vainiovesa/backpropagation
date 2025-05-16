import json
from random import choice
from network import Network, CustomNetwork

def gradient_descent(net:Network, data:list, learning_rate:float, iterations:int, tolerance:float, info_packet_size:int):
    info = []
    errors = []
    i = 1

    min_error = float("inf")
    max_error = 0
    tolerance_error = tolerance + 1

    while i <= iterations and tolerance_error > tolerance:
        inputs, expected_outputs = choice(data)
        net.feedforward(inputs)
        error = net.loss(expected_outputs)
        errors.append(error)

        min_error = min(min_error, error)
        max_error = max(max_error, error)

        if i % info_packet_size == 0:
            error_average = sum(errors) / info_packet_size
            info.append((error_average, min_error, max_error))
            errors = []

            tolerance_error = max_error
            min_error = float("inf")
            max_error = 0

        net.reverse(inputs, expected_outputs, learning_rate)
        i += 1

    return info

def save_network(net:Network, json_filename:str):
    dictionary = net.as_dict()
    net_as_json = json.dumps(dictionary, indent=4)
    with open(json_filename, "w") as outfile:
        outfile.write(net_as_json)

def upload_network(json_filename:str):
    with open(json_filename, 'r') as file:
        network_as_json = json.load(file)
    net = CustomNetwork(network_as_json)
    return net
