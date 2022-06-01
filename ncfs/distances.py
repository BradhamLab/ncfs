import numpy as np
import numba

_mock_ones = np.ones(2, dtype=np.float64)
# implementation/syntax inspired by UMAP.distances


@numba.njit(fastmath=True)
def manhattan(x, y, w=_mock_ones):
    """Calculate the L1-distance between two vectors."""
    result = 0.0
    for i in numba.prange(x.shape[0]):
        result += w[i] * abs(x[i] - y[i])
    return result


@numba.njit(fastmath=True)
def manhattan_grad(X, i, j, l, w=_mock_ones):
    """Calculate manhattan distance gradient with respect to feature weight l."""
    return 2 * w[l] * abs(X[i, l] - X[j, l])


@numba.njit(fastmath=True)
def euclidean(x, y, w):
    """Calculate weighted Euclidean distnace between two vectors."""
    result = 0.0
    for i in numba.prange(x.shape[0]):
        result += w[i] * (x[i] - y[i]) ** 2
    return np.sqrt(result)


@numba.njit(fastmath=True)
def euclidean_grad(X, i, j, l, w=_mock_ones):
    """Calculate manhattan distance gradient with respect to feature weight l."""
    dist = euclidean(X[i, :], X[j, :], w)
    return 2 * w[l] * (X[i, l] - X[j, l]) ** 2 / dist


@numba.njit(fastmath=True)
def sqeuclidean(x, y, w=_mock_ones):
    """Calculate the L2-distance between two vectors."""
    result = 0.0
    for i in numba.prange(x.shape[0]):
        result += w[i] * (x[i] - y[i]) ** 2
    return result


@numba.njit(fastmath=True)
def sqeuclidean_grad(X, i, j, l, w=_mock_ones):
    return 2 * w[l] * (X[i, l] - X[j, l]) ** 2


@numba.njit(fastmath=True)
def covariance(x, y, w=_mock_ones):
    """Calculate sample co-variance between two vectors."""
    result = 0.0
    # calculate average values over vectors
    x_bar = 0.0
    y_bar = 0.0
    for i in numba.prange(x.shape[0]):
        x_bar += x[i]
        y_bar += y[i]
    x_bar *= 1.0 / x.shape[0]
    y_bar *= 1.0 / x.shape[0]
    # calculate sum of weighted distance products
    for i in numba.prange(x.shape[0]):
        result += w[i] * (x[i] - x_bar) * (y[i] - y_bar)
    # dives by n - 1 for sample covariance
    result *= 1.0 / (x.shape[0] - 1)
    return result


@numba.njit(fastmath=True)
def variance(x, w=_mock_ones):
    """Calculate sample weighted variance"""
    mean = 0.0
    sum_of_weights = 0.0
    sum_of_squared_weights = 0.0
    result = 0
    for i in numba.prange(x.shape[0]):
        mean += x[i] * w[i]
        sum_of_weights += w[i]
        sum_of_squared_weights += w[i] ** 2
    if sum_of_weights == 0:
        return np.nan
    mean = mean / sum_of_weights
    for i in numba.prange(x.shape[0]):
        result += w[i] * (x[i] - mean) ** 2
    return result / (sum_of_weights - sum_of_squared_weights / sum_of_weights)


# should we 1 - this?
@numba.njit()
def phi_s(x, y, w=_mock_ones):
    """Calculate the phi_s proportionality metric between two vectors."""
    if np.all(x == y):
        return 0.0
    nume = variance(x - y, w)
    denom = variance(x + y, w)
    if denom == 0:
        return np.inf
    return nume / denom


@numba.njit()
def rho_p(x, y, w=_mock_ones):
    """Calculate the rho_p proportionality metric between two vectors."""
    if np.all(x == y):
        return 1.0
    return 1.0 - variance(x - y, w) / (variance(x, w) + variance(y, w))


supported_distances = {
    "l1": manhattan,
    "cityblock": manhattan,
    "taxicab": manhattan,
    "manhattan": manhattan,
    "l2": sqeuclidean,
    "sqeuclidean": sqeuclidean,
    "phi_s": phi_s,
    "rho_p": rho_p,
}


@numba.njit(parallel=True)
def pdist(X, w, dist, func, symmetric=True):
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
            raise ValueError(
                "Reference index should be between 0 and " "{}".format(X.shape[1])
            )
        ref = log_X[:, ref]
    elif isinstance(ref, np.ndarray):
        if ref.size != X.shape[1]:
            raise ValueError(
                "Expected reference array of size " "{}".format(X.shape[1])
            )
    else:
        raise ValueError("Unexpected input type for `ref`: {}".format(type(ref)))
    return log_X - (np.ones_like(X).T * ref).T


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
        `X[i, :] - x[j, :]` and `X[i, :] + X[j, :]`, respectively. Should be
        initialized to a zero tensor before fitting.
    ss : [type]
        A (2, sample, sample) tensor. Elements `ss[0, i, j]` and `ss[1, i, j]`
        represent the weighted sum of squared values between
        `X[i, :] - x[j, :]` and `X[i, :] + X[j, :]`, respectively. Should be
        initialized to a zero tensor before fitting.

    Returns
    -------
    float
        Sum of weights in `w` vector.
    """
    sum_of_weights = 0.0
    for x in w:
        sum_of_weights += x
    # once = True
    for i in numba.prange(X.shape[0]):
        for j in numba.prange(i, X.shape[0]):
            for l in numba.prange(X.shape[1]):
                # calculate x - y means
                means[0, i, j] += w[l] * (X[i, l] - X[j, l])
                means[0, j, i] += w[l] * (X[j, l] - X[i, l])
                # calculate x + y means, addition is communicative
                res = w[l] * (X[i, l] + X[j, l])
                means[1, i, j] += res
                means[1, j, i] += res

            means[:, i, j] /= sum_of_weights
            means[:, j, i] /= sum_of_weights
    for i in numba.prange(X.shape[0]):
        for j in numba.prange(i, X.shape[0]):
            for l in numba.prange(X.shape[1]):
                # calculate x - y sum of squares
                ss[0, i, j] += w[l] * ((X[i, l] - X[j, l]) - means[0, i, l]) ** 2
                ss[0, j, i] += w[l] * ((X[j, l] - X[i, l]) - means[0, j, l]) ** 2
                # calculate x + y sum of squares, addition is communicative
                res = w[l] * ((X[i, l] + X[j, l]) - means[1, i, l]) ** 2
                ss[1, i, j] += res
                ss[1, j, i] += res
    return sum_of_weights


supported_distances = {
    "l1": manhattan,
    "cityblock": manhattan,
    "taxicab": manhattan,
    "manhattan": manhattan,
    "l2": euclidean,
    "euclidean": euclidean,
    "sqeuclidean": sqeuclidean,
    "phi_s": phi_s,
}

metrics = {
    "manhattan": (manhattan, manhattan_grad),
    "cityblock": (manhattan, manhattan_grad),
    "taxicab": (manhattan, manhattan_grad),
    "l1": (manhattan, manhattan_grad),
    "l2": (euclidean, euclidean_grad),
    "euclidean": (euclidean, euclidean_grad),
    "sqeuclidean": (sqeuclidean, sqeuclidean_grad),
}
#    'phi_s': (phi_s, phi_s_grad)}
