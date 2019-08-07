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


class PhiS(WeightedDistance):
    r"""
    Calculate partial derivatives for Symmetric Phi.
    
    .. math::

        \phi_s(X, Y) = \frac{\Var(X - Y)}{\Var(X + y)}
    """

    def __init__(self, X, w):
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
        super(PhiS, self).__init__(X, w)
        self.__fit(X)

    def __fit(self, X):
        """Fit tensor D_ to X."""
        X_bar = np.ones_like(X.T) * np.mean(X, axis=1)
        X_bar = np.ones_like(X.T) * np.mean(X, axis=1)
        centered = X - X_bar.T
        # calculate covariance matrix
        cov = np.cov(X, ddof=1)
        # initialize (sample x sample x feature) tensor
        D = np.ones((X.shape[0], X.shape[0], X.shape[1]))
        # if you want to braodcast, do this, but it's slower
        # D = cov[:, :, None] * (squared[None, :, :] +  squared[:, None, :])\
        #    - centered[None, :, :] * centered[:, None, :]\
        #    * (var[:, None] + var[None, :])

        # numba vectorize?
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                D[i, j, :] = 4.0 / (X.shape[1] - 1)\
                           * (cov[i, j]\
                           * (centered[i, :] ** 2 +  centered[j, :] ** 2)\
                           - (centered[i, :] * centered[j, :])\
                           * (cov[i, i] + cov[j, j]))\
                           / ((cov[i, i] + cov[j, j] + 2 * cov[i, j]) ** 2)
        self.D_ = D

    def partials(self):
        return self.D_ * self.weights_[None, None, :]


class RhoP(WeightedDistance):
    r"""
    Calculate partial derivatives for Rho metric of proportionality.
    
    .. math::

        \Rho_p(X, Y) = \frac{\Var(X - Y)}{\Var(X) + \Var(y)}
    """

    def __init__(self, X, w):
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
        super(RhoP, self).__init__(X, w)
        # calculate a (sample x feature) matrix where sample values are
        # centered by the sample mean (i.e X_bar[i, ] = X[i, ] - mean(X[i, ]))
        self.__fit(X)

    def __fit(self, X):
        """Fit tensor D_ to X."""
        X_bar = np.ones_like(X.T) * np.mean(X, axis=1)
        centered = X - X_bar.T
        # calculate covariance matrix
        cov = np.cov(X, ddof=1)
        # initialize (sample x sample x feature) tensor
        D = np.ones((X.shape[0], X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                D[i, j, :] = 4.0 / (X.shape[1] - 1)\
                           * (cov[i, j]\
                           * (centered[i, :] ** 2 + centered[j, :] ** 2)\
                           - (centered[i, :] * centered[j, :])\
                           * (cov[i, i] + cov[j, j])) \
                           / ((cov[i, i] + cov[j, j]) ** 2)
        self.D_ = D

    def partials(self):
        return self.D_ * -self.weights_[None, None, :]


class Manhattan(WeightedDistance):
    r"""
    Calculate partial derivatives for weighted Manhattan Distance.
    
    .. math::

        L1(X, Y) = \sum \limits_{i = 1}^N w_l^2| x_i - y_i |
    """

    def __init__(self, X, w):
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
        super(Manhattan, self).__init__(X, w)
        self.__fit(X)

    def __fit(self, X):
        D = np.ones((X.shape[0], X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                D[i, j, :] = 2 * np.abs(X[i, :] - X[j, :])
        self.D_ = D

    def partials(self):
        return self.D_ * self.weights_[None, None, :]


class SqEuclidean(WeightedDistance):
    r"""
    Calculate partial derivatives for weighted squared euclidean distance.
    
    .. math::

        L2(X, Y) = \sum \limits_{i = 1}^N w_l^2 (x_i - y_i)^2 
    """

    def __init__(self, X, w):
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
        super(SqEuclidean, self).__init__(X, w)
        self.__fit(X)

    def __fit(self, X):
        D = np.ones((X.shape[0], X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                D[i, j, :] = 2 * (X[i, :] - X[j, :]) ** 2 
        self.D_ = D

    def partials(self):
        return self.D_ * self.weights_[None, None, :]

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