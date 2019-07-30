import numpy as np
import numba

_mock_ones = np.ones(2, dtype=np.float64)
# implementation/syntax inspired by UMAP.distances

@numba.njit()
def manhattan(x, y, w=_mock_ones):
    """Calculate the L1-distance between two vectors."""
    result = 0.0
    for i in range(x.shape[0]):
        result += w[i] * abs(x[i] - y[i])
    return result

@numba.njit()
def sqeuclidean(x, y, w=_mock_ones):
    """Calculate the L2-distance between two vectors."""
    result = 0.0
    for i in range(x.shape[0]):
        result += w[i] * (x[i] - y[i]) ** 2
    return result

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
    """Calculate the phi_s proportionality metric between two vectors."""
    return variance(x - y, w) / variance(x + y, w)

@numba.njit()
def rho_p(x, y, w=_mock_ones):
    """Calculate the rho_p proportionality metric between two vectors."""
    result = 
    return(variance(x - y, w) / variance(x, w) + variance(y, w))
  
supported_distances = {'l1': manhattan,
                       'cityblock': manhattan,
                       'taxicab': manhattan,
                       'manhattan': manhattan,
                       'l2': sqeuclidean,
                       'sqeuclidean': sqeuclidean,
                       'phi_s': phi_s,
                       'rho_p': rho_p}
