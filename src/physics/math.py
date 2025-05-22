import numpy as np


def conv_discrete(f, g, x):
    dx = x[1] - x[0]  # Paso de muestreo
    return np.convolve(f, g, mode='same') * dx  # Escalar por dt para aproximar la integral