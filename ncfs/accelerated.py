import numpy as np
import numba

from . import distances

@numba.njit(parallel=True)
def exponential_transform(D, sigma):
    return np.exp(-1 * D / sigma)

@numba.njit(parallel=True)
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
    # dmat = spatial.distance.squareform(spatial.distance.pdist(X, metric='cityblock', w=self.metric.w_ ** 2))
    # calculate K(D_w(x_i, x_j)) for all i, j pairs
    p_reference = transform(D, sigma)
    # set p_ii = 0, can't select self as reference sample
    np.fill_diagonal(p_reference, 0.0)
    # add pseudocount if necessary to avoid dividing by zero
    row_sums = p_reference.sum(axis=1)
    n_zeros = np.sum(row_sums == 0)
    if n_zeros > 0:
        pseudocount = np.exp(-20)
        row_sums += pseudocount
    scale_factors = 1 / (row_sums)
    p_reference = (p_reference.T * scale_factors).T
    return p_reference

@numba.njit()
def feature_gradient(X, class_matrix, sample_weights, p_reference, p_correct,
                     l, gradient_matrix, metric, sigma, reg):
    """[summary]
    
    Parameters
    ----------
    X : [type]
        [description]
    class_matrix : [type]
        [description]
    sample_weights : [type]
        [description]
    p_reference : [type]
        [description]
    p_correct : [type]
        [description]
    l : [type]
        [description]
    gradient_matrix : [type]
        [description]
    metric : [type]
        [description]
    sigma : [type]
        [description]
    reg : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    metric.partials(X, gradient_matrix, l)
    # weighted gradient matrix D_ij = d_ij * p_ij, p_ii = 0
    gradient_matrix *= p_reference
    # calculate p_i * sum(D_ij), j from 0 to N
    all_term = p_correct * gradient_matrix.sum(axis=1)
    # weighted in-class distances using adjacency matrix,
    in_class_term = np.sum(gradient_matrix * class_matrix, axis=1)
    sample_terms = sample_weights * (all_term - in_class_term)
    # calculate delta following gradient ascent 
    return 1 / sigma * sample_terms.sum() - 2 * metric.w_[l] * reg

@numba.jit(parallel=True)
def gradients(X, class_matrix, sample_weights, metric,
              distance, transform, sigma, reg):
    # sample by sample matrix to store distance over all features, weighted
    D = np.zeros((X.shape[0], X.shape[0]))
    # vector to store partial derivatives for all feature weights
    partials = np.zeros(X.shape[1])
    # Matrix P, where P_ij is the probability of j referencing i
    p_reference = probability_matrix(X, metric.w_, D, distance, transform,
                                     sigma)
    # a sample length vector of the correctly assigning sample i
    p_correct = (p_reference * class_matrix).sum(axis=1)
    for l in numba.prange(X.shape[1]):
        # (sample x sample) matrix to store gradients where element
        # (i, j) is the gradient for w_l between samples i and 
        gradient_matrix = np.zeros((X.shape[0], X.shape[0]))
        partials[l] = feature_gradient(X, class_matrix, sample_weights,
                                       p_reference, p_correct, l,
                                       gradient_matrix, metric,
                                       sigma, reg)
            
    return p_reference, partials