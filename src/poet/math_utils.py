from math import factorial

def _binomial(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))