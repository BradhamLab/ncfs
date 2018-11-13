"""
Python implementation of Neighborhood Component Feature Selection

Yang, W., Wang, K., & Zuo, W. (2012). Neighborhood Component Feature Selection
for High-Dimensional Data. Journal of Computers, 7(1).
https://doi.org/10.4304/jcp.7.1.161-168

Author : Dakota Hawkins
"""

import numpy as np
from scipy import spatial

class NCFS(object): 

    def __init__(self, alpha, sigma, reg, nu):
        """
        Class to perform Neighborhood Component Feature Selection 

        Parameters
        ----------
        alpha : float
            Initial step length for gradient ascent. Should be between 0 and 1.
        sigma : float
            Kernel width.
        reg : float
            Regularization constant. Lambda in the original paper.
        nu : float
            Stopping criteria for iteration. Threshold for difference between
            objective function scores after each iteration. 
        """
        if not 0 < alpha < 1:
            raise ValueError("Alpha value should be between 0 and 1.")
        self.alpha = alpha
        self.sigma = sigma
        self.reg = reg
        self.nu = nu 
        self.coef_ = None
        self.objective_ = np.inf

    @staticmethod
    def __check_X(X):
        mins = np.min(X, axis=0)
        maxes = np.max(X, axis=0)
        if any(mins < 0):
            raise ValueError('Values in X should be between 0 and 1.')
        if any(maxes > 1):
            raise ValueError('Values in X should be between 0 and 1.')

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
        None
        """
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
        NCFS.__check_X(X)
        n_samples, n_features = X.shape
        self.coef_ = np.ones(n_features)
        p_reference = np.zeros((n_samples, n_samples))
        p_correct = np.zeros(n_samples)
        deltas = np.zeros(n_features)
        # construct adjacency matrix of class membership for matrix mult. 
        class_mat = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if y[i] == y[j]:
                    class_mat[i, j] = 1

        current_objective = 0
        while abs(self.objective_ - current_objective) > self.nu:
            # calculate K(D_w(xi, xj)) for all i, j pairs
            for i in range(n_samples):
                for j in range(n_samples):
                    if i == j:
                        p_reference[i, j] = 0
                    else:
                        p_reference[i, j] = self.kernel_distance(X[i, :],
                                                                 X[j, :])
            # scale p_reference by row sums 
            scale_factors = 1 / p_reference.sum(axis = 1)
            p_reference = p_reference * scale_factors

            # calculate probability of correct classification
            p_correct = np.sum(p_reference * class_mat, axis=1)

            # caclulate weight adjustments
            for l in range(n_features):
                feature_vec = X[:, l].reshape(-1, 1)
                # weighted sample distances
                d_mat = spatial.distance_matrix(feature_vec, feature_vec)
                d_mat = np.abs(d_mat) * p_reference
                # weighted in-class distances
                in_class = d_mat*class_mat
                sample_terms = np.sum(p_correct * d_mat, axis=1) \
                             - np.sum(in_class, axis=1)
                # calculate delta following gradient ascent 
                deltas[l] = 2 * ((1 / self.sigma) * sample_terms.sum() \
                          - self.reg) * self.coef_[l]
                
            # update weights and other parameters
            self.coef_ += self.alpha * deltas
            self.objective_ = current_objective
            current_objective = np.sum(p_reference * class_mat) \
                              - self.reg * np.dot(self.coef_, self.coef_)
            if current_objective > self.objective_:
                self.alpha *= 1.01
            else:
                self.alpha *= 0.4

    def kernel_distance(self, x_i, x_j):
        abs_diff = np.abs(x_i - x_j)
        weighted_dist = np.sum(self.coef_**2 * abs_diff)
        return np.exp(-1 * weighted_dist / self.sigma)

def toy_dataset():
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
    bad_features = np.random.normal(loc=0, scale=np.sqrt(20), size=(200, 1000))
    data = np.hstack((class_data[:, 0].reshape(-1, 1), bad_features[:, :99],
                      class_data[:, 1].reshape(-1, 1), bad_features[:, 99:]))
    classes = np.array([0]*100 + [1]*100)
    # scale between 0 - 1
    x_std = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return x_std, classes

if __name__ == '__main__':
    X, y = toy_dataset()
    f_select = NCFS(alpha=0.01, sigma=1, reg=1, nu=0.01)
    f_select.fit(X, y)