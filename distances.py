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


@numba.njit()
def variance(x, w=_mock_ones):
    """Calculate sample weighted variance"""
    mean = 0.0
    sum_of_weights = 0.0
    sum_of_squared_weights = 0.0
    result = 0
    for i in range(x.shape[0]):
        mean += x[i] * w[i]
        sum_of_weights += w[i]
        sum_of_squared_weights += w[i] ** 2
    mean = mean / sum_of_weights
    for i in range(x.shape[0]):
        result += w[i] * (x[i] - mean) ** 2
    return result / (sum_of_weights - sum_of_squared_weights / sum_of_weights)


@numba.njit()
def phi_s(x, y, w=_mock_ones):
    """Calculate the phi_s proportionality metric between two vectors."""
    if np.all(x == y):
        return 0.0
    return variance(x - y, w) / variance(x + y, w)


@numba.njit()
def rho_p(x, y, w=_mock_ones):
    """Calculate the rho_p proportionality metric between two vectors."""
    if np.all(x == y):
        return 1.0
    return 1.0 - variance(x - y, w) / (variance(x, w) + variance(y, w))


supported_distances = {'l1': manhattan,
                       'cityblock': manhattan,
                       'taxicab': manhattan,
                       'manhattan': manhattan,
                       'l2': sqeuclidean,
                       'sqeuclidean': sqeuclidean,
                       'phi_s': phi_s,
                       'rho_p': rho_p}

@numba.njit(parallel=True)
def symmetric_pdist(X, w, dist, func, symmetric = True):
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
    for i in numba.prange(X.shape[0]):
        u = X[i, :]
        for j in numba.prange(i + 1, X.shape[0]):
            v = X[j, :]
            res = func(u, v, w)
            dist[i, j] = res
            if symmetric:
                sym_res = res
            else:
                sym_res = func(v, u, w)
            dist[j, i] = sym_res

# TODO: Might need to transpose matrix
def log_ratio(X, ref=None):
    r"""
    Calculate the log-ratio matrix of a data matrix.
    
    Parameters
    ----------
    X : numpy.ndarray
        A (samples x features) data matrix. 
    ref : int, numpy.ndarray, optional
        Reference value for log-ratio calcuation. By default None, and the
        centered-log ratio transformation is used. Whereby

        .. math::
            clr(X_{ij}) = \log(\frac{X_ij} / G(X_i))
        
        and

        .. math::
           g(X_i) = \left ( \prod \limits_{j=1}^{M} X_{ij} \right ) ^ {(1/M)}
        
        If `ref` is an integer, it is assumed to be a column index in `X`, and 

        .. math::
            g(X_i) = X_{:, ref}
        
        Otherwise, if `ref` is a numpy array of feature length:

        .. math::
            g(X_i) = `ref`
        
        
    Returns
    -------
    numpy.ndarray
        Log-ratio transformed data matrix.
    """
    if np.any(X == 0) is None:
        print("Replacing zeros with next smallest values")
        X[X == 0] = np.min(X[X != 0])
    log_X = np.log(X)
    if ref is None:
        ref = np.mean(log_X, axis=1)
    elif isinstance(ref, int):
        if not 0 < ref < X.shape[1]:
            raise ValueError("Reference index should be between 0 and "
                             "{}".format(X.shape[1]))
        ref = log_X[:, ref]
    elif isinstance(ref, np.ndarray):
        if ref.size != X.shape[1]:
            raise ValueError("Expected reference array of size "
                             "{}".format(X.shape[1]))
    else:
        raise ValueError("Unexpected input type for `ref`: {}".format(
                         type(ref)))
    return log_X - (np.ones_like(X).T * ref).T 
    

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

@numba.njit(parallel=True)
def fit_phi_s(X, w, means, ss):
    """
    Fit means, sum of squared erros, and sum of weights for PhiS derivatives.
    
    Parameters
    ----------
    X : numpy.ndarray
        A (sample x feature) data matrix.
    w : numpy.ndarray
        A (feature) length vector of feature weights.
    means : numpy.ndarray
        A (2, sample, sample) tensor. Elements `means[0, i, j]` and
        `means[1, i, j]` represent the weighted mean values between
        `X[i, :] - x[j, :]` and `X[i, :] + X[j, :]`, respectively. 
    ss : [type]
        A (2, sample, sample) tensor. Elements `ss[0, i, j]` and `ss[1, i, j]`
        represent the weighted sum of squared values between
        `X[i, :] - x[j, :]` and `X[i, :] + X[j, :]`, respectively. 
    
    Returns
    -------
    [type]
        [description]
    """
    sum_of_weights = 0.0
    for x in w:
        sum_of_weights += x
    # once = True
    for i in numba.prange(X.shape[0]):
        for j in numba.prange(i, X.shape[0]):
            for l in numba.prange(X.shape[1]):
                if i == j:
                    # calculate x - x means
                    means[0, i, i] += 0
                    means[1, i, i] += 2 * w[l] * X[i, l]
                else:
                    # calculate x - y means
                    means[0, i, j] += w[l] * (X[i, l] - X[j, l])
                    means[0, j, i] += w[l] * (X[j, l] - X[i, l])
                    # calculate x + y means
                    means[1, i, j] += w[l] * (X[i, l] + X[j, l])
                    means[1, j, i] += w[l] * (X[j, l] + X[i, l])
    
            means[:, i, j] /= sum_of_weights
            means[:, j, i] /= sum_of_weights
    for i in numba.prange(X.shape[0]):
        for j in numba.prange(i, X.shape[0]):
            for l in numba.prange(X.shape[1]):
                if i == j:
                    # calculate x - x sum of squares
                    ss[0, i, i] += -w[l] * means[0, i, l]
                    ss[1, j, j] += (w[l] * (2 * X[i, l] - means[1, i, l]) ** 2)
                else:
                    # calculate x - y sum of squares
                    ss[0, i, j] += (w[l] * (X[i, l] - X[j, l]) - means[0, i, l])**2
                    ss[0, j, i] += (w[l] * (X[j, l] - X[i, l]) - means[0, j, l])**2
                    # calculate x + y sum of squares
                    ss[1, i, j] += (w[l] * (X[i, l] + X[j, l]) - means[1, i, l])**2
                    ss[1, j, i] += (w[l] * (X[j, l] + X[i, l]) - means[1, j, l])**2
    return sum_of_weights

# specification of 
spec = [('v1_', numba.float64),
        ('means_', numba.float64[:, :, :]),
        ('ss_', numba.float64[:, :, :]),
        ('w_', numba.float64[:])]
@numba.jitclass(spec)
class PhiS(object):
    r"""
    Calculate partial derivatives for Symmetric Phi.
    
    .. math::

        \phi_s(X, Y) = \frac{\Var(X - Y)}{\Var(X + Y)}
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
        self.means_ = np.zeros((2, X.shape[0], X.shape[0]))
        self.ss_ = np.zeros_like(self.means_)
        self.v1_ = 0.0
        self.__update_values(X, w)

    def __update_values(self, X, w):
        """
        Update calculated means, sum of squared means, and total feature weight.
        
        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix
        w : numpy.ndarray
            a (feature) length vector of feature weights. 
        """
        self.w_ = w
        self.means_ *= 0
        self.ss_ *= 0
        self.v1_ = fit_phi_s(X, self.w_, self.means_, self.ss_)
       

    def sample_feature_partial(self, X, i, j, l):
        """
        Calculate `d Phi(x_i, X_j) / d w_l`.

        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feautre) data matrix.
        i : int
            The ith sample in `X`.
        j : int
            The jth sample in `X`.
        l : int
            The lth feature in `X`.
        
        Returns
        -------
        float
            Partial derivative of `Phi_s(x, y)` for feature `l`. 
        """
        x = X[i, l] - X[j, l]
        y = X[i, l] + X[j, l]
        return((2 * self.w_[l] * ((self.w_[l] - self.v1_) / self.v1_ ** 2) \
                            * (self.ss_[1, i, j] * (x ** 2\
                                                   -x * self.means_[0, i, j])\
                              -self.ss_[0, i, j] * (y ** 2\
                                                   -y * self.means_[1, i, j])))\
                            / self.ss_[1, i, j] ** 2)

    def partials(self, X, D, l):
        """
        Caculate `d Phi(x_i, X_j) / d w_l` for all samples `i` and `j`. 
        
        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix.
        D : numpy.ndarray
            A (sample x sample) matrix for storing derivatives.
        l : int
            The `lth` feature in `X`. 
        
        Returns
        -------
        numpy.ndarray
            Matrix of partial derivative of `Phi_s(x, y)` for feature `l` over
            all samples `x` and `y`. 
        """
        for i in numba.prange(X.shape[0]):
            for j in numba.prange(i, X.shape[0]):
                D[i, j] = self.sample_feature_partial(X, i, j, l)
                D[j, l] = self.sample_feature_partial(X, j, i, l)
        return D



@numba.jitclass(spec)
class RhoP(object):
    r"""
    Calculate partial derivatives for Symmetric Phi.
    
    .. math::

        \rho_s(X, Y) = \frac{\Var(X - Y)}{\Var(X) + \Var(Y)}
    """

    def __init__(self, X, D):
        r"""
        Class for quick calculuations of
        :math:`\frac{\partial \rho_p}{\partial w_l}`.
        
        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix.
        D : numpy.memmap
            A (sample x sample x feature) distance matrix 
        """
        self.D_ = D
        centered = X - (np.ones_like(X.T) * np_mean(X, 1)).T
        cov = np.cov(X, ddof=1)
        self.__fit(X, centered, cov)

    def __fit(self, X, centered, cov):
        """Fit tensor D_ to X."""
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.D_[i, j, :] = 4.0 / (X.shape[1] - 1)\
                                 * (cov[i, j]\
                                 * (np.power(centered[i, :], 2)\
                                    + np.power(centered[j, :], 2))\
                                 - (centered[i, :] * centered[j, :])\
                                 * (cov[i, i] + cov[j, j])) \
                                 / (np.power(cov[i, i] + cov[j, j], 2))

    def partials(self, weights):
        return self.D_ * -weights


@numba.jitclass(spec)
class Manhattan(WeightedDistance):
    r"""
    Calculate partial derivatives for weighted Manhattan Distance.
    
    .. math::

        L1(X, Y) = \sum \limits_{i = 1}^N w_l^2| x_i - y_i |
    """

    def __init__(self, X, D):
        r"""
        Class for quick calculuations of
        :math:`\frac{\partial L1}{\partial w_l}`.
        
        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix.
        D : numpy.memmap
            A (sample x sample x feature) distance matrix
        """
        self.D_ = D
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

    def __init__(self, X, D):
        r"""
        Class for quick calculuations of
        :math:`\frac{\partial L2}{\partial w_l}`.
        
        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix.
        D : numpy.memmap
            A (sample x sample x feature) distance matrix
        """
        self.D_ = D
        self.__fit(X)

    def __fit(self, X):
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.D_[i, j, :] = 2 * np.power((X[i, :] - X[j, :]), 2)

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