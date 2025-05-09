from math import exp

def relu(x):
    return x if x > 0 else 0

def sigmoid(x):
    return 1 / (1 + exp(-x))
