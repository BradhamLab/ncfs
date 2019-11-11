"""
Python implementation of Neighborhood Component Feature Selection

Yang, W., Wang, K., & Zuo, W. (2012). Neighborhood Component Feature Selection
for High-Dimensional Data. Journal of Computers, 7(1).
https://doi.org/10.4304/jcp.7.1.161-168

Author : Dakota Hawkins
"""
from tempfile import mkdtemp
import os

import numba
import numpy as np
from scipy import spatial
from sklearn import base, model_selection

from ncfs_expanded import distances
# import distances


class NCFSOptimizer(object):
    """Gradient ascent/descent optimization following NCFS protocol."""

    def __init__(self, alpha=0.01):
        """
        Gradient ascent/descent optimization following NCFS protocol.
        
        Parameters
        ----------
        alpha : float, optional
            Initial step size. The default is 0.01.

        Attributes
        ----------
        alpha

        Methods
        -------
        get_steps(gradients) : get gradient deltas for each feature gradient.
        """
        self.alpha = alpha

    def get_steps(self, gradients, loss=None, *args):
        """
        Calculate gradient deltas for each feature gradient.
        
        Parameters
        ----------
        gradients : numpy.ndarray
            Calculated gradient delta of objective function with respect to
            given feature.
        loss : float
            Difference between objective function at t and t - 1.
        
        Returns
        -------
        numpy.ndarray
            Gradient steps for each feature gradient.
        """
        steps = self.alpha * gradients
        if loss is not None:
            if loss > 0:
                self.alpha *= 1.01
            else:
                self.alpha *= 0.4
        return steps


@numba.njit(parallel=True)
def exp_gradient(p_reference, p_correct, partial, class_matrix,
                 weight, reg, sigma):
    # weighted partial matrix D_ij = d_ij * p_ij, p_ii = 0
    fuzzy_partials = partial * p_reference
    # calculate p_i * sum(D_ij), j from 0 to N
    all_term = p_correct * fuzzy_partials.sum(axis=1)
    # weighted in-class distances using adjacency matrix,
    in_class_term = np.sum(fuzzy_partials * class_matrix, axis=1)
    val =  0.0
    for i in numba.prange(all_term.size):
        val += (all_term[i] - in_class_term[i])
    out = ((1.0 / sigma) * val - 2.0 * reg * weight)
    return out

@numba.njit(parallel=True)
def feature_gradients(p_reference, p_correct, partials, weights,
                      class_matrix, reg, sigma, gradients):
    for l in numba.prange(gradients.size):
        gradients[l] = exp_gradient(p_reference, p_correct, partials[:, :, l],
                                    class_matrix, weights[l], reg, sigma)
    return gradients

 
class ExponentialKernel(object):
    """Base class for kernel functions."""

    def __init__(self, sigma, reg, n_features):
        """
        Base class for kernel functions.

        Parameters
        ----------
        sigma : float
            Scaling parameter sigma as mentioned in original paper.
        reg : float
            Regularization parameter.
        n_features : float
            Number of features.

        Methods
        -------
        transform():
            Apply the kernel tranformation to a distance matrix.

        gradients():
            Get current gradients for each feature
        """
        self.sigma = sigma
        self.reg = reg
        self.gradients_ = np.zeros(n_features, np.float64)

    def transform(self, distance_matrix):
        """Apply a kernel transformation to a distance matrix."""
        return np.exp(-1 * distance_matrix / self.sigma)

    def gradients(self, p_reference, partials, weights, class_matrix):
        r"""
        Calculate feature gradients of objective function.

        Calculates the gradient vector of the objective function with respect
        to feature weights. Objective function is the same outlined in the
        original NCFS paper.

        **Objective Function**:

        .. math::
            E(\vec w) = \sum \limits_i^N \sum \limits_i^N y_{ij} * p_{ij}
                      - \lambda \sum \limits_l^M w_l^2
        
        Parameters
        ----------
        p_reference : numpy.ndarray
            A (sample x sample) probability matrix with :math:`P_{ij}`
            represents the probability of selecting sample :math:`j` as a
            reference for sample :math:`i`.
        partials : numpy.ndarray
            A (sample x sample x feature) tensor holding partial derivative
            values, such that

            .. math::
                A[i, j, l] = \frac{\partial D(x_i, x_j, w)}{\partial w_l}

            Where :math:`D(x_i, x_j, w)` is some distance function with feature
            weights.
        weights : numpy.ndarray
            A feature-length array feature weights.
        class_matrix : numpy.ndarray
            A one-hot (sample x sample) matrix where :math:`Y_{ij} = 1`
            indicates sample :math:`i` and sample :math:`j` are in the same
            class, and :math:`Y_{ij} = 0` otherwise. It is assumed
            :math:`Y_{ii} = 0`. 
        
        Returns
        -------
        numpy.ndarray
            Gradient vector for each feature with respect to the objective
            function.
        """
        p_correct = np.sum(p_reference * class_matrix, axis=1)
        self.gradients_ = feature_gradients(p_reference, p_correct, partials,
                                            weights, class_matrix, self.reg,
                                            self.sigma, self.gradients_)
        return self.gradients_


class NCFS(base.BaseEstimator, base.TransformerMixin):
    """
    Class to perform Neighborhood Component Feature Selection 

    References
    ----------

    Yang, W., Wang, K., & Zuo, W. (2012). Neighborhood Component Feature
    Selection for High-Dimensional Data. Journal of Computers, 7(1).
    https://doi.org/10.4304/jcp.7.1.161-168
    """

    def __init__(self, alpha=0.1, sigma=1, reg=1, eta=0.001,
                 metric='cityblock', kernel='exponential', solver='ncfs'):
        """
        Class to perform Neighborhood Component Feature Selection 

        Parameters
        ----------
        alpha : float, optional
            Initial step length for gradient ascent. Should be between 0 and 1.
            Default is 0.1.
        sigma : float, optional
            Kernel width. Default is 1.
        reg : float, optional
            Regularization constant. Lambda in the original paper. Default is 1.
        eta : float, optional
            Stopping criteria for iteration. Threshold for difference between
            objective function scores after each iteration. Default is 0.001.
        metric : str, optional
            Metric to calculate distances between samples. Must be a scipy
            implemented distance and accept a parameter 'w' for a weighted
            distance. Default is the euclidean distance.
        kernel : str, optional
            Method to calculate kernel distance between samples. Possible values
            are 'exponential'.
        solver : str, optional
            Method to perform gradient ascent. Possible values are 'ncfs'.

        Attributes:
        ----------
        alpha : float
            Step length for gradient ascent. Varies during training.
        sigma : float
            Kernel width.
        reg : float
            Regularization constant. Lambda in the original paper.
        eta : float
            Stopping criteria for iteration. Threshold for difference between
            objective function scores after each iteration.
        metric : str
            Distance metric to use.
        kernel : str
            Kernel function to use to calculate kernel distance.
        solver : str
            Method to use to perform gradient ascent.
        coef_ : numpy.array
            Feature weights. Unimportant features tend toward zero.
        score_ : float
            Objective function score at the end of fitting.

        Methods
        -------

        fit : Fit feature weights given a particular data matrix and sample
            labels.

        References
        ----------
        Yang, W., Wang, K., & Zuo, W. (2012). Neighborhood Component Feature
        Selection for High-Dimensional Data. Journal of Computers, 7(1).
        https://doi.org/10.4304/jcp.7.1.161-168
        """
        self.alpha = alpha
        self.sigma = sigma
        self.reg = reg
        self.eta = eta 
        self.metric = metric
        self.kernel = kernel
        self.solver = solver
        self.coef_ = None
        self.score_ = None

    @staticmethod
    def __check_X(X):
        mins = np.min(X, axis=0)
        maxes = np.max(X, axis=0)
        if any(mins < 0) or any(maxes > 1):
            print('Best to have values in X between 0 and 1.')
        return X.astype(np.float64)

    def fit(self, X, y):
        """
        Fit feature weights using Neighborhood Component Feature Selection.

        Fit feature weights using Neighborhood Component Feature Selection.
        Weights features in `X` by their ability to distinguish classes in `y`.
        Coefficients are set to the instance variable `self.coef_`. 

        Parameters
        ----------
        X : numpy.ndarray
            An n x p data matrix where n is the number of samples, and p is the
            number of features.
        y : numpy.array
            List of pre-defined classes for each sample in `X`.

        Returns
        -------
        NCFS
            Fitted NCFS object with weights stored in the `.coef_` instance
            variable.
        """
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha value should be between 0 and 1.")
        if not isinstance(X, np.ndarray):
            raise ValueError('`X` must be two-dimensional numpy array. Got ' + 
                             '{}.'.format(type(X)))
        if len(X.shape) != 2:
            raise ValueError('`X` must be two-dimensional numpy array. Got ' + 
                             '{} dimensional.'.format(len(X.shape)))
        if not isinstance(y, np.ndarray):
            raise ValueError('`y` must be a numpy array. ' + 
                             'Got {}.'.format(type(y)))
        if y.shape[0] != X.shape[0]:
            raise ValueError('`X` and `y` must have the same row numbers.')
        n_samples, n_features = X.shape
        # intialize kernel function
        if self.kernel == 'exponential':
            X = NCFS.__check_X(X)
            self.kernel_ = ExponentialKernel(self.sigma, self.reg, n_features)
        else:
            raise ValueError('Unsupported kernel function: {}'\
                             .format(self.kernel))
        # initialize gradient ascent solver
        if self.solver == 'ncfs':
            self.solver_ = NCFSOptimizer(alpha=self.alpha)
        else:
            raise ValueError('Unsupported gradient ascent method{}'.\
                             format(self.solver))
        # initialize distance metric
        if self.metric.lower() not in distances.supported_distances:
            raise ValueError("Unsupported distance metric {}".\
                             format(self.metric))
        else:
            self.metric_ = distances.supported_distances[self.metric]
            # initialize WeightedDistance object for partial calculations.
            weighted_dist = distances.partials[self.metric]
            # create distance memmap file
            memfile = os.path.join(mkdtemp(), 'distance.dat')
            self.distance_ = weighted_dist(X, np.memmap(memfile,
                                                        dtype='float64',
                                                        mode='w+',
                                                        shape=(X.shape[0],
                                                               X.shape[0],
                                                               X.shape[1])))
        # initialize all weights as 1
        self.coef_ = np.ones(n_features, dtype=np.float64)
        # construct adjacency matrix of class membership for matrix mult. 
        class_matrix = np.zeros((n_samples, n_samples), np.float64)
        for i in range(n_samples):
            for j in range(n_samples):
                if y[i] == y[j] and i != j:
                    class_matrix[i, j] = 1

        objective, loss = 0, np.inf
        while abs(loss) > self.eta:
            new_objective = self.__fit(X, class_matrix, objective)
            loss = objective - new_objective
            objective = new_objective
        self.score_ = objective
        # delete stored temporary file
        fn = self.distance_.D_.filename
        print(fn)
        os.remove(fn)
        os.removedirs(os.path.split(fn)[0])
        return self

    def transform(self, X):
        """
        Transform features according to their learned weights.
        
        Parameters
        ----------
        X : numpy.ndarray
            An `(n x p)` data matrix where `n` is the number of samples, and `p`
            is the number of features. Features number and order should be the
            same as those used to fit the model.  
        
        Raises
        ------
        RuntimeError
            Raised if the NCFS object has not been fit yet.
        ValueError
            Raided if the number of feature dimensions does not match the
            number of learned weights.
        
        Returns
        -------
        numpy.ndarray
            Transformed data matrix calculated by multiplying each feature by
            its learnt weight.
        """

        if self.coef_ is None:
            raise RuntimeError('NCFS is not fit. Please fit the ' +
                               'estimator by calling `.fit()`.')
        if X.shape[1] != len(self.coef_):
            raise ValueError('Expected data matrix `X` to contain the same' + 
                             'number of features as learnt feature weights.')
        if self.kernel == 'exponential':
            X = NCFS.__check_X(X)
        return X * self.coef_ ** 2

    def __fit(self, X, class_matrix, objective):
        """
        Underlying method to fit NCFS model.
        
        Parameters
        ----------
        X : numpy.ndarray
            A sample by feature data matrix.
        class_matrix : numpy.ndarray
            A sample by sample one-hot matrix marking samples in the same class.
        objective : float
            Objective score reached from past iteration
        
        Returns
        -------
        float
            Objective reached by current iteration.
        """
        # organize as distance matrix by summing along feature index
        # calculate D_w(x_i, x_j): sum(w_l^2 * |x_il - x_jl], l) for all i,j
        dmat = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
        distances.symmetric_pdist(X, self.coef_, dmat, self.metric_)
        # calculate K(D_w(x_i, x_j)) for all i, j pairs
        p_reference = self.kernel_.transform(dmat)
        # set p_ii = 0, can't select self as reference sample
        np.fill_diagonal(p_reference, 0.0)
        # add pseudocount if necessary to avoid dividing by zero
        row_sums = p_reference.sum(axis=1)
        n_zeros = sum(row_sums == 0)
        if n_zeros > 0:
            print('Adding pseudocounts to distance matrix to avoid ' +
                    'dividing by zero.')
            pseudocount = np.exp(-20)
            row_sums += pseudocount
        scale_factors = 1 / (row_sums)
        p_reference = (p_reference.T * scale_factors).T
        # calculate partial derivates for each sample x sample x feature trip
        partials = self.distance_.partials(self.coef_)
        # caclulate weight adjustments
        gradients = self.kernel_.gradients(p_reference, partials, self.coef_,
                                           class_matrix)
        # calculate objective function
        new_objective = (np.sum(p_reference * class_matrix) \
                      - self.reg * np.dot(self.coef_, self.coef_))
        # calculate loss from previous objective function
        loss = new_objective - objective
        # update weights
        deltas = self.solver_.get_steps(gradients, loss)
        self.coef_ = self.coef_ + deltas
        # return objective score for new iteration
        return new_objective


def toy_dataset(n_features=1000):
    """
    Generate a toy dataset with features from the original NCFS paper.
    
    Generate a toy dataset with features from the original NCFS paper. Signal
    features are in the first index, and the 10th percent index (e.g.
    :math:`0.1 * N`). See original paper for specific parameter values for
    signal/noise features.
    
    Parameters
    ----------
    n_features : int, optional
        Number of total features. Two of these features will feature signal,
        the other N - 2 will be noise. The default is 1000.
    
    Returns
    -------
    tuple (X, y)
        X : numpy.array
            Simulated dataset with 200 samples (rows) and N features. Features
            are scaled between 0 and 1.
        y : numpy.array
            Class membership for each sample in X.
    """

    class_1 = np.zeros((100, 2))
    class_2 = np.zeros((100, 2))
    cov = np.identity(2)
    for i in range(100):
        r1, r2 = np.random.rand(2)
        if r1 > 0.5:
            class_1[i, :] = np.random.multivariate_normal([-0.75, -3], cov)
        else:
            class_1[i, :] = np.random.multivariate_normal([0.75, 3], cov)
        if r2 > 0.5:
            class_2[i, :] = np.random.multivariate_normal([3, -3], cov)
        else:
            class_2[i, :] = np.random.multivariate_normal([-3, 3], cov)
    class_data = np.vstack((class_1, class_2))
    n_irrelevant = n_features - 2
    second_idx = int(0.1*(n_features)) - 1
    bad_features = np.random.normal(loc=0, scale=np.sqrt(20),
                                    size=(200, n_irrelevant))
    data = np.hstack((class_data[:, 0].reshape(-1, 1),
                      bad_features[:, :second_idx],
                      class_data[:, 1].reshape(-1, 1),
                      bad_features[:, second_idx:]))
    classes = np.array([0]*100 + [1]*100)
    # scale between 0 - 1
    x_std = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return x_std, classes

def main():
    X, y = toy_dataset(n_features=1000)
    f_select = NCFS(alpha=0.001, sigma=1, reg=1, eta=0.001, metric='cityblock',
                    kernel='exponential', solver='ncfs')
    from timeit import default_timer as timer
    times = np.zeros(1)
    # previous 181.82286000379972
    # not parallel: 116
    # parallel, jobs=1 = 124
    # expanding dims = 150s
    # vectorized 104
    for i in range(times.size):            
        start = timer()
        f_select.fit(X, y)
        end = timer()
        times[i] = end - start
    print("Average execution time in seconds: {}".format(np.mean(times)))
    sorted_coef = np.argsort(-1 * f_select.coef_)
    print(sorted_coef[:10])
    print(f_select.coef_[0], f_select.coef_[100], f_select.coef_[sorted_coef[2]])

if __name__ == '__main__':
    main()
