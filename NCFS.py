"""
Python implementation of Neighborhood Component Feature Selection

Yang, W., Wang, K., & Zuo, W. (2012). Neighborhood Component Feature Selection
for High-Dimensional Data. Journal of Computers, 7(1).
https://doi.org/10.4304/jcp.7.1.161-168

Author : Dakota Hawkins
"""

import numpy as np
from scipy import spatial
from sklearn import base

def pairwise_distance(data_matrix, metric='euclidean'):
    """
    Calculate the pairwise distance between each sample in each feature.
    
    Parameters
    ----------
    data_matrix : numpy.ndarray
        An (N x M) data matrix where N is the number of samples and M is the
        number of features.
    metric : str, optional
        Distance metric to use. The default is 'euclidean'.
    
    Returns
    -------
    numpy.ndarray
        An (M x N X N) numpy array where array[0] is the distance matrix
        representing pairwise distances between samples in feature 0.
    """
    # matrix to hold pairwise distances between samples in each feature
    dists = np.zeros((data_matrix.shape[1],
                        data_matrix.shape[0], data_matrix.shape[0]))
    for j in range(data_matrix.shape[1]):
        dists[j] = spatial.distance.squareform(
                        spatial.distance.pdist(X[:, j].reshape(-1, 1),
                                                metric=metric))
    return dists

class KernelMixin(object):

    def __init__(self, sigma, reg, metric, weights):
        self.sigma = sigma
        self.reg = reg
        self.metric = metric
        self.weights = weights

    def transform(self, distance_matrix):
        return distance_matrix

    def gradient_deltas(self, p_reference, data_matrix, class_matrix,
                        metric='cityblock'):
        return np.zeros(data_matrix.shape[1])

    def update_weights(self, new_weights):
        self.weights = new_weights


class ExponentialKernel(KernelMixin):

    def __init__(self, sigma, reg, metric, weights):
        super(ExponentialKernel, self).__init__(sigma, reg, metric, weights)

    def transform(self, distance_matrix):
        return np.exp(-1 * distance_matrix / self.sigma, dtype=np.float64)

    def gradient_deltas(self, p_reference, data_matrix, class_matrix,
                        metric='cityblock'):
        deltas = np.zeros(data_matrix.shape[1], dtype=np.float64)
        # calculate probability of correct classification
        # check class mat one hot shit
        p_correct = np.sum(p_reference * class_matrix, axis=0)

        # calculate unweighted, pairwise distance between all samples
        # in all features
        feature_distances = pairwise_distance(data_matrix, metric=self.metric)

        # caclulate weight adjustments
        for l in range(data_matrix.shape[1]):
            # distance matrix between all samples in feature l
            d_mat = feature_distances[l]
            # weighted distance matrix D_ij = d_ij * p_ij, p_ii = 0
            d_mat *= p_reference
            # calculate p_i * sum(D_ij), j from 0 to N
            all_term = p_correct * d_mat.sum(axis=0)
            # weighted in-class distances using adjacency matrix,
            in_class_term = np.sum(d_mat * class_matrix, axis=0)
            sample_terms = all_term - in_class_term
            # calculate delta following gradient ascent 
            deltas[l] = 2 * self.weights[l] \
                        * ((1 / self.sigma) * sample_terms.sum() - self.reg)
        return deltas


class GaussianKernel(KernelMixin):

    def __init__(self, sigma, reg, metric, weights):
        super(GaussianKernel, self).__init__(sigma, reg, metric, weights)

    def transform(self, distance_matrix):
        centered = self.__scale_distances(distance_matrix)
        return np.exp(-1 * centered / self.sigma, dtype=np.float64)

    def gradient_deltas(self, p_reference, data_matrix, class_matrix,
                        metric='cityblock'):
        deltas = np.zeros(data_matrix.shape[1], dtype=np.float64)
        # calculate probability of correct classification
        # check class mat one hot shit
        p_correct = np.sum(p_reference * class_matrix, axis=0)
        # caclulate weight adjustments
        for l in range(data_matrix.shape[1]):
            # values for feature l starting with sample 0 to N
            feature_vec = data_matrix[:, l].reshape(-1, 1)
            # distance in feature l for all samples, d_ij
            d_mat = spatial.distance.pdist(feature_vec, metric=metric) # only need to calculate this once
            d_mat = spatial.distance.squareform(d_mat)
            # weighted distance matrix D_ij = d_ij * p_ij, p_ii = 0
            d_mat *= p_reference
            # calculate p_i * sum(D_ij), j from 0 to N
            all_term = p_correct * d_mat.sum(axis=0)
            # weighted in-class distances using adjacency matrix,
            in_class_term = np.sum(d_mat * class_matrix, axis=0)
            sample_terms = all_term - in_class_term
            # calculate delta following gradient ascent 
            deltas[l] = 2 * self.weights[l] \
                        * ((1 / self.sigma) * sample_terms.sum() - self.reg)

    def __scale_distances(self, distance_matrix):
        mean_dists = np.sum(distance_matrix, axis=0)\
                   / (distance_matrix.shape[0] - 1)
        center_mat = np.ones((mean_dists.size, mean_dists.size)) * mean_dists
        return distance_matrix - center_mat.T



class NCFS(base.BaseEstimator, base.TransformerMixin): 

    def __init__(self, alpha=0.1, sigma=1, reg=1, eta=0.001,
                 metric='cityblock', kernel='exponential'):
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
            distance. Default is 'cityblock', as used in the original paper.
        kernel : str, optional
            Method to calculate kerel distance between samples. Default is 
            'exponential', as used in the original NCFS paper.

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
        self.coef_ = None
        self.score_ = None

    @staticmethod
    def __check_X(X):
        mins = np.min(X, axis=0)
        maxes = np.max(X, axis=0)
        if any(mins < 0):
            raise ValueError('Values in X should be between 0 and 1.')
        if any(maxes > 1):
            raise ValueError('Values in X should be between 0 and 1.')
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
        X= NCFS.__check_X(X)
        n_samples, n_features = X.shape
        # initialize all weights as 1
        self.coef_ = np.ones(n_features, dtype=np.float64)

        # intialize kernel function
        if self.kernel == 'exponential':
            kernel = ExponentialKernel(self.sigma, self.reg, self.metric,
                                       self.coef_)
        elif self.kernel == 'gaussian':
            kernel = GaussianKernel(self.sigma, self.reg, self.metric,
                                    self.coef_)
        else:
            raise ValueError('Unsupported kernel ' +
                             'function: {}'.format(self.kernel))

        # get initial step size
        step_size = self.alpha 
        # construct adjacency matrix of class membership for matrix mult. 
        class_mat = np.zeros((n_samples, n_samples), np.float64)
        for i in range(n_samples):
            for j in range(n_samples):
                if y[i] == y[j]:
                    class_mat[i, j] = 1

        past_objective, loss = 0, np.inf
        diag_idx = np.diag_indices(n_samples, 2)
        while abs(loss) > self.eta:
            # calculate D_w(x_i, x_j): w^2 * |x_i - x_j] for all i,j
            distances = spatial.distance.pdist(X, metric=self.metric,
                                               w=np.power(self.coef_, 2))
            # organize as distance matrix
            distances = spatial.distance.squareform(distances)
            # calculate K(D_w(x_i, x_j)) for all i, j pairs
            p_reference = kernel.transform(distances)
            # set p_ii = 0, can't select self in leave-one-out
            p_reference[diag_idx] = 0

            # add pseudocount if necessary to avoid dividing by zero
            p_i = p_reference.sum(axis=0)
            n_zeros = sum(p_i == 0)
            if n_zeros > 0:
                print('Adding pseudocounts to distance matrix to avoid ' +
                      'dividing by zero.')
                if n_zeros == len(p_i):
                    pseudocount = np.exp(-20)
                else:
                    pseudocount = np.min(p_i)
                p_i += pseudocount
            scale_factors = 1 / (p_i)
            p_reference = p_reference * scale_factors

            # caclulate weight adjustments
            deltas = kernel.gradient_deltas(p_reference, X, class_mat,
                                            metric=self.metric)
                
            # calculate objective function
            new_objective = (np.sum(p_reference * class_mat) \
                          - self.reg * np.dot(self.coef_, self.coef_))
            # calculate loss from previous objective function
            loss = new_objective - past_objective
            # update weights
            self.coef_ = self.coef_ + step_size * deltas
            kernel.update_weights(self.coef_)
            # reset objective score for new iteration
            past_objective = new_objective
            if loss > 0:
                step_size *= 1.01
            else:
                step_size *= 0.4
        self.score_ = past_objective
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
        NCFS.__check_X(X)
        return X*self.coef_


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

if __name__ == '__main__':
    X, y = toy_dataset()
    f_select = NCFS(alpha=0.01, sigma=1, reg=1, eta=0.001)
    f_select.fit(X, y)
    print(np.argsort(-f_select.coef_)[:10])
    print(f_select.coef_[0], f_select.coef_[100])