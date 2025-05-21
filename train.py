import json
from random import choice
from network import Network, CustomNetwork

def gradient_descent(net:Network, data:list, learning_rate:float,
                     iterations:int, tolerance:float, info_packet_size:int):
    """
    Train a neural network with gradient descent.

    Parameters
    ----------
    net : Network
    data : list of tuples of lists
        [([inputs], [expected outputs]), ...]
    learning_rate : float
    iterations : int
        The maximum number of iterations before the training process finishes.
    tolerance : float
        If the maximum error of a group of iterations is smaller than the tolerance,
        the training process finishes.
    info packet size : int
        Size of the group of iterations.

    Returns
    -------
    list of tuples
        Information about the gradient descent process. List of error values for
        each group of iterations.\n
        [(average, minimum, maximum), ...].
    """
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
    """Stores a Network object to a json file"""
    net_as_dict = net.as_dict()
    net_as_json = json.dumps(net_as_dict, indent=4)
    with open(json_filename, "w") as file:
        file.write(net_as_json)

def upload_network(json_filename:str):
    """Creates a CustomNetwork object based on a network stored to a json file"""
    with open(json_filename, "r") as file:
        net_as_dict = json.load(file)
    return CustomNetwork(net_as_dict)
