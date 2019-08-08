import numpy as np
import numba

_mock_ones = np.ones(2, dtype=np.float64)

# implementation/syntax of distance functions inspired by UMAP.distances
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


@numba.njit()
def variance(x, w=_mock_ones):
    """Caluculate the sample variance of a vector."""
    return covariance(x, x, w=w)


@numba.njit()
def phi_s(x, y, w=_mock_ones):
    """Calculate the phi_s proportionality metric between two vectors."""
    return variance(x - y, w) / variance(x + y, w)


@numba.njit()
def rho_p(x, y, w=_mock_ones):
    """Calculate the rho_p proportionality metric between two vectors."""
    return 1 - variance(x - y, w) / (variance(x, w) + variance(y, w))

@numba.njit()
def symmetric_pdist(X, w, dist, func):
    """
    Find the pairwise distances between all rows in a data matrix.

    Find the pairwise distances between all rows in a (sample x feature)
    data matrix. Distance function is assumed to be symmetric.

    Parameters
    ----------
    X : numpy.ndarray
        A (sample x feature) data matrix.
    w : numpy.ndarray
        Feature weights for each feature in the data matrix.
    dist : numpy.ndarray
        A (sample x sample) matrix to fill with pairwise distances between rows.
    func : numba.njit function
        Numba compiled function to measure distance between rows. Assumed to be
        a symmetric measure.
    """
    for i in range(X.shape[0]):
        u = X[i, :]
        for j in range(i + 1, X.shape[0]):
            v = X[j, :]
            res = func(u, v, w)
            dist[i, j] = res
            dist[j, i] = res


class WeightedDistance(object):
    r"""
    Class for weighted, differentiable distance metrics.

    Attributes
    ----------
        D_ : numpy.ndarray
            A (sample x sample x feature) tensor of static values used to
            calculate partial deriviates of feature weights.
            
            For example, for the weighted L1 distance, the partial for the
            distance between sample :math:`i` and sample :math:`j` in feature
            :math:`l` is:

            .. math::

                \frac{\partial L1(x_i, x_j, w)}{\partial w_l} =
                      w_l | x_il - x_jl|

            D_ is then a (sample x sample x feature) tensor of
            :math:`|x_il - x_jl|` values.

        weights_ : numpy.ndarray
            A feature-length array of feature weights used when calculating
            partials.

            From the example above, the tensor `D_`
            contains :math:`|x_il - x_jl|` values. Therefore, to calculate
            partials each :math:`|x_il - x_jl|` value must be multiplied by its
            associated :math:`w_l` value.

    Methods
    -------
        update_weights(value : numpy.ndarray)
            Update feature weights.
        __fit(X : numpy.ndarray)
            Fit the tensor D to the data matrix X. Not implemented in base
            class.
        partials()
            Calculate and return a (sample x sample x feature) tensor, 
            :math:`A`, of partial derivatives. Where:
            
            .. math::
                A_{ijl} = \frac{\partial(Dist(X_i, X_j, w))}
                               {\partial w_l}
            
            Not implemented in base class.
    """

    def __init__(self, X, w):
        """
        Fit model to data for quick calculation of partial derivates.

        Fits a (sample x sample x feature) tensor to X that finds static 
        values used during partial derivate calculatons.

        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix.
        w : numpy.ndarray
            A weight vector for each feature in X.
        """
        if X.shape[1] != w.size:
            raise ValueError("Length of weight vector and the number of " 
                             "feautres in X do not align.")
        self.__fit(X)
        self.weights_ = w
    
    def __fit(self, X):
        self.D_ = None

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

# numba implementatin of axis means -- compliments to Joel Rich
# https://github.com/numba/numba/issues/1269#issuecomment-472574352

@numba.njit()
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@numba.njit()
def np_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)

@numba.njit()
def np_sum(array, axis):
    return np_apply_along_axis(np.sum, axis, array)
    


# specification of 
spec = [('D_', numba.float64[:, :, :])]
@numba.jitclass(spec)
class PhiS(object):
    r"""
    Calculate partial derivatives for Symmetric Phi.
    
    .. math::

        \phi_s(X, Y) = \frac{\Var(X - Y)}{\Var(X + Y)}
    """

    def __init__(self, X):
        r"""
        Class for quick calculuations of
        :math:`\frac{\partial \phi_s}{\partial w_l}`.
        
        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix.
        w : numpy.ndarray
            A feature-length vector 
        """
        self.D_ = np.ones((X.shape[0], X.shape[0], X.shape[1]),
                          dtype=np.float64)
        centered = X - (np.ones_like(X.T) * np_mean(X, 1)).T
        cov = np.cov(X, ddof=1)
        self.__fit(X, centered, cov)
        # remove centered, cov

    def __fit(self, X, centered, cov):
        """Fit tensor D_ to X."""
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.D_[i, j, :] = 4.0 / (X.shape[1] - 1)\
                                 * (cov[i, j]\
                                 * (centered[i, :] ** 2 + centered[j, :] ** 2)\
                                 - (centered[i, :] * centered[j, :])\
                                 * (cov[i, i] + cov[j, j]))\
                                 / ((cov[i, i] + cov[j, j]\
                                     + 2 * cov[i, j]) ** 2)

    def partials(self, weights):
        return self.D_ * weights


@numba.jitclass(spec)
class RhoP(object):
    r"""
    Calculate partial derivatives for Symmetric Phi.
    
    .. math::

        \rho_s(X, Y) = \frac{\Var(X - Y)}{\Var(X) + \Var(Y)}
    """

    def __init__(self, X):
        r"""
        Class for quick calculuations of
        :math:`\frac{\partial \rho_p}{\partial w_l}`.
        
        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix.
        w : numpy.ndarray
            A feature-length vector 
        """
        self.D_ = np.ones((X.shape[0], X.shape[0], X.shape[1]),
                            dtype=np.float64)
        centered = X - (np.ones_like(X.T) * np_mean(X, 1)).T
        cov = np.cov(X, ddof=1)
        self.__fit(X, centered, cov)

    def __fit(self, X, centered, cov):
        """Fit tensor D_ to X."""
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.D_[i, j, :] = 4.0 / (X.shape[1] - 1)\
                                 * (cov[i, j]\
                                 * (centered[i, :] ** 2 + centered[j, :] ** 2)\
                                 - (centered[i, :] * centered[j, :])\
                                 * (cov[i, i] + cov[j, j])) \
                                 / ((cov[i, i] + cov[j, j]) ** 2)

    def partials(self, weights):
        return self.D_ * -weights


@numba.jitclass(spec)
class Manhattan(WeightedDistance):
    r"""
    Calculate partial derivatives for weighted Manhattan Distance.
    
    .. math::

        L1(X, Y) = \sum \limits_{i = 1}^N w_l^2| x_i - y_i |
    """

    def __init__(self, X):
        r"""
        Class for quick calculuations of
        :math:`\frac{\partial L1}{\partial w_l}`.
        
        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix.
        w : numpy.ndarray
            A feature-length vector 
        """
        self.D_ = np.ones((X.shape[0], X.shape[0], X.shape[1]),
                          dtype=np.float64)
        self.__fit(X)

    def __fit(self, X):
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.D_[i, j, :] = 2 * np.abs(X[i, :] - X[j, :])

    def partials(self, weights):
        return self.D_ * weights

@numba.jitclass(spec)
class SqEuclidean(object):
    r"""
    Calculate partial derivatives for weighted squared euclidean distance.
    
    .. math::

        L2(X, Y) = \sum \limits_{i = 1}^N w_l^2 (x_i - y_i)^2 
    """

    def __init__(self, X):
        r"""
        Class for quick calculuations of
        :math:`\frac{\partial L2}{\partial w_l}`.
        
        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix.
        w : numpy.ndarray
            A feature-length vector 
        """
        self.D_ = np.ones((X.shape[0], X.shape[0], X.shape[1]),
                          dtype=np.float64)
        self.__fit(X)

    def __fit(self, X):
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.D_[i, j, :] = 2 * (X[i, :] - X[j, :]) ** 2 

    def partials(self, weights):
        return self.D_ * weights

supported_distances = {'l1': manhattan,
                       'cityblock': manhattan,
                       'taxicab': manhattan,
                       'manhattan': manhattan,
                       'l2': sqeuclidean,
                       'sqeuclidean': sqeuclidean,
                       'phi_s': phi_s,
                       'rho_p': rho_p}

partials = {'l1': Manhattan,
            'cityblock': Manhattan,
            'taxicab': Manhattan,
            'manhattan': Manhattan,
            'l2': SqEuclidean,
            'sqeuclidean': SqEuclidean,
            'phi_s': PhiS,
            'rho_p': RhoP}