import math
decay = 0.19898883519701627


def count_factor(day):
    return math.exp(- max(((day - 7)/7, 0)) * decay)
