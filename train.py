from random import choice
from network import Network

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
