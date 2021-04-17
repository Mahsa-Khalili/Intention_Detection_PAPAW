import numpy as np


def l2norm(array):
    '''L2 norm of an array'''
    return np.linalg.norm(array, ord=2)


def rms(array):
    '''Root mean squared of an array'''
    return np.sqrt(np.mean(array ** 2))