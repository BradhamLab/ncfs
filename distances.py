import numpy as np
import numba

_mock_ones = np.ones(2, dtype=np.float64)

@numba.njit()
def variance(x, w=_mock_ones):
    """Caluculate the sample variance of a vector."""
    x_bar = 0.0
    result = 0.0
    # calculate the average value
    for i in range(x.shape[0]):
        x_bar += x[i]
    x_bar *= 1 / x.shape[0]
    # calculate sample variance
    for i in range(x.shape[0]):
        result += w[i] * (x[i] - x_bar) ** 2
    result *= 1 / (x.shape[0] - 1)
    return result

@numba.njit()
def phi_s(x, y, w=_mock_ones):
    """Calculate the phi_s proportionaly metric between two vectors."""
    return variance(x - y, w) / variance(x + y, w)
