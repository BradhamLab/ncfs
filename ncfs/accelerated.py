import numpy as np
import numba

from ncfs import distances

# TODO not sure we're calculating w_l ** 2 * dist(x, y)


@numba.njit(parallel=True, fastmath=True)
def exponential_transform(D, sigma):
    """Transform distances to kernel distances."""
    return np.exp(-1 * D / sigma)


@numba.njit(parallel=True, fastmath=True)
def probability_matrix(X, w, D, distance, transform, sigma):
    """
    Numba accelerated calculation of reference probability matrix.

    Calculates the matrix `P`. Element `p_ij` is the probability of sample `j`
    being the reference point during 1-NN classification step

    Parameters
    ----------
    X : numpy.ndarray
        A (sample x feature) data matrix.
    w : numpy.ndarray
        A (feature) length vector of feature weights.
    D : numpy.ndarray
        A (sample x sample) matrix to be filled with distances between samples.
        Assumed to be a zero matrix.
    distance : Numba.njit
        A Numba.njit() accelerated distance function to measure the weighted
        distance between two vectors.
    transform : Numba.njit
        A Numba.njit() compiled function that transforms the distance matrix
        into a kernel space.

    Returns
    -------
    numpy.ndarray
        The matrix `P`.
    """
    # calculate D_w(x_i, x_j): sum(w_l^2 * |x_il - x_jl], l) for all i,j
    distances.pdist(X, w ** 2, D, distance, True)
    # calculate K(D_w(x_i, x_j)) for all i, j pairs
    p_reference = transform(D, sigma)
    for i in range(p_reference.shape[0]):
        row_sum = 0.0
        for j in range(p_reference.shape[0]):
            if i == j:
                # set p_ii as 0, can't reference self
                p_reference[i, j] = 0.0
            row_sum += p_reference[i, j]
        # check pseudo counts? Not doing anything?
        if row_sum == 0:
            p_reference[i, :] = np.exp(-20)
            row_sum = np.exp(-20) * p_reference.shape[0]
            row_sum += np.exp(-20)
        p_reference[i, :] /= row_sum
    return p_reference


@numba.njit(parallel=True, fastmath=True)
def feature_gradient(
    X,
    class_matrix,
    coefs,
    sample_weights,
    p_reference,
    p_correct,
    l,
    distance_grad,
    sigma,
    reg,
):
    value = 0.0
    for i in numba.prange(p_reference.shape[0]):
        all_term = 0.0
        in_class_term = 0.0
        for j in numba.prange(p_reference.shape[1]):
            # change this to grad function (X, w, i, j, l)
            partial = distance_grad(X, i, j, l, coefs)
            all_term += partial * p_reference[i, j]
            in_class_term += partial * p_reference[i, j] * class_matrix[i, j]
        value += sample_weights[i] * (all_term * p_correct[i] - in_class_term)

    # calculate delta following gradient ascent
    return 1.0 / sigma * value - 2 * coefs[l] * reg


@numba.njit(parallel=True, fastmath=True)
def objective(p_reference, class_matrix, sample_weights, coef, reg):
    score = 0
    for i in numba.prange(p_reference.shape[0]):
        row_score = 0
        for j in numba.prange(p_reference.shape[1]):
            row_score += p_reference[i, j] * class_matrix[i, j]
        row_score *= sample_weights[i]
        score += row_score
    reg_term = 0
    for i in numba.prange(coef.size):
        reg_term += coef[i] ** 2
    return score - reg * reg_term


@numba.njit(parallel=True, fastmath=True)
def score(
    X,
    class_matrix,
    sample_weights,
    coefs,
    distance_matrix,
    distance,
    transform,
    sigma,
    reg,
):
    p_reference = probability_matrix(
        X, coefs, distance_matrix, distance, transform, sigma
    )
    return objective(p_reference, class_matrix, sample_weights, coefs, reg)


@numba.njit(fastmath=True)
def ncfs_update_weights(coefs, feature_gradients, alpha, loss):
    for i in numba.prange(coefs.size):
        coefs[i] += feature_gradients[i] * alpha
    if loss > 0:
        alpha *= 1.01
    else:
        alpha *= 0.4


@numba.njit(parallel=True, fastmath=True)
def partial_fit(
    X,
    class_matrix,
    sample_weights,
    coefs,
    distance,
    distance_grad,
    transform,
    sigma,
    reg,
    alpha,
    distance_matrix,
    partials_vec,
    score,
):
    # Matrix P, where P_ij is the probability of j referencing i
    p_reference = probability_matrix(
        X, coefs, distance_matrix, distance, transform, sigma
    )
    # a sample length vector -- probability of correctly assigning sample i
    p_correct = (p_reference * class_matrix).sum(axis=1)
    for l in numba.prange(X.shape[1]):
        # (sample x sample) matrix to store gradients where element
        # (i, j) is the gradient for w_l between samples i and
        partials_vec[l] = feature_gradient(
            X,
            class_matrix,
            coefs,
            sample_weights,
            p_reference,
            p_correct,
            l,
            distance_grad,
            sigma,
            reg,
        )
    new_score = objective(p_reference, class_matrix, sample_weights, coefs, reg)
    loss = new_score - score
    ncfs_update_weights(coefs, partials_vec, alpha, loss)
    return new_score


@numba.njit(cache=True)
def fit(
    X,
    class_matrix,
    sample_weights,
    coefs,
    distance,
    distance_grad,
    transform,
    sigma,
    reg,
    alpha,
    eta,
    max_iter,
    distance_matrix,
    partials_vec,
):
    loss = np.inf
    i = 0
    score = 0
    while abs(loss) > eta and i < max_iter:
        new_score = partial_fit(
            X,
            class_matrix,
            sample_weights,
            coefs,
            distance,
            distance_grad,
            transform,
            sigma,
            reg,
            alpha,
            distance_matrix,
            partials_vec,
            score,
        )
        loss = score - new_score
        score = new_score
        i += 1
    return (score, i)
