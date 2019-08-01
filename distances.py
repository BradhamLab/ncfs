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
def covariance(x, y, w=_mock_ones):
    """Calculate sample co-variance between two vectors."""
    result = 0.0
    # calculate average values over vectors
    x_bar = 0.0
    y_bar = 0.0
    for i in range(x.shape[0]):
        x_bar += x[i]
        y_bar += y[i]
    x_bar *= 1.0 / x.shape[0]
    y_bar *= 1.0 / x.shape[0]
    # calculate sum of weighted distance products
    for i in range(x.shape[0]):
        result += w[i] * (x[i] - x_bar) * (y[i] - y_bar)
    # dives by n - 1 for sample covariance
    result *= 1.0 / (x.shape[0] - 1)
    return result


def variance(x, w=_mock_ones):
    """Caluculate the sample variance of a vector."""
    return covariance(x, x, w=_mock_ones)


@numba.njit()
def phi_s(x, y, w=_mock_ones):
    """Calculate the phi_s proportionality metric between two vectors."""
    return variance(x - y, w) / variance(x + y, w)


@numba.njit()
def rho_p(x, y, w=_mock_ones):
    """Calculate the rho_p proportionality metric between two vectors."""
    return variance(x - y, w) / (variance(x, w) + variance(y, w))


supported_distances = {'l1': manhattan,
                       'cityblock': manhattan,
                       'taxicab': manhattan,
                       'manhattan': manhattan,
                       'l2': sqeuclidean,
                       'sqeuclidean': sqeuclidean,
                       'phi_s': phi_s,
                       'rho_p': rho_p}

class WeightedDistance(object):
    """
    Class for weighted, differentiable distance metrics.
    """

    def __init__(self, X, w):
        """
        Fit model to data for easy calculation of partial derivates

        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix.
        w : numpy.ndarray
            A weight vector for each feature in X.
        """
        if X.shape[1] != w.size():
            raise ValueError("Length of weight vector and the number of " 
                             "feautres in X do not align.")
        self.D_ = None
        self.weights_ = w
    
    def update_weights(self, value):
        """Set weights for each feature."""
        try:
            # when setting new weights, ensure expected shape is returned
            same_shape = np.all(value.shape == self.weights_.shape)
        except AttributeError:
            # weights hasn't be set yet, nothing to compare against
            same_shape = True
        if not same_shape:
            raise ValueError("Shape of weight vector differs from initialized.")
        self.weights_ = value


    def partials(self):
        """
        Calculate the partial derivative of the distance metric with respect to
        each feature weight.
        """
        pass 

class PhiS(WeightedDistance):
    """Calculate easy partial derivatives for Symmetric Phi"""

    def __init__(self, X, w):
        super(PhiS, self).__init__(X, w)
        X_bar = np.ones_like(X.T) * np.mean(X, axis=1)
        centered = X - X_bar.T
        cov = np.cov(X, ddof=1)
        # var = np.diag(cov).reshape(-1, 1)
        D = np.ones((X.shape[0], X.shape[0], X.shape[1]))
        # if you want to braodcast, do this, but it's slower
        # oi = cov[:, :, None] * (squared[None, :, :] +  squared[:, None, :])\
        #    - centered[None, :, :] * centered[:, None, :]\
        #    * (var[:, None] + var[None, :])
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                D[i, j, :] = 2.0 / (X.shape[1] - 1)\
                           * (cov[i, j]\
                           * (centered[i, :] ** 2 +  centered[j, :] ** 2)\
                           - (centered[i, :] * centered[j, :])\
                           * (cov[i, i] + cov[j, j]))
        self.D_ = D

    def partials(self):
        return self.D_ * self.weights_
    