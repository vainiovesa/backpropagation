from math import exp

def softmax(x:list):
    max_x = max(x)
    e_pow = [exp(x_i - max_x) for x_i in x]
    sum_e = sum(e_pow)
    return [e_pow_i / sum_e for e_pow_i in e_pow]
