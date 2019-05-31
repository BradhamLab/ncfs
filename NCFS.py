"""
Python implementation of Neighborhood Component Feature Selection

Yang, W., Wang, K., & Zuo, W. (2012). Neighborhood Component Feature Selection
for High-Dimensional Data. Journal of Computers, 7(1).
https://doi.org/10.4304/jcp.7.1.161-168

Author : Dakota Hawkins
"""

import numpy as np
from scipy import spatial
from sklearn import base, model_selection

def pairwise_feature_distance(data_matrix, metric='euclidean'):
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
                        spatial.distance.pdist(data_matrix[:, j].reshape(-1, 1),
                                               metric=metric))
    return dists

def center_distances(distance_matrix):
    """
    Mean-center a distance matrix per row.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        A square distance matrix.
    
    Returns
    -------
    np.ndarray
        A centered distance matrix where values are centered by row means. 
    """
    # subtract row means from each d_ij value
    mean_dists = np.sum(distance_matrix, axis=1)\
                / (distance_matrix.shape[0] - 1)
    center_mat = np.ones((mean_dists.size, mean_dists.size)) * mean_dists
    return distance_matrix - center_mat.T

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


class AdamOptimizer(object):
    """
    Gradient ascent/descent optimization via the Adam algorithm.
    
    References
    ----------
        Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic
        Optimization. Retrieved from http://arxiv.org/abs/1412.6980
    """
    
    def __init__(self, n_features, alpha=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=10**(-8)):
        """
        Gradient ascent/descent optimization via Adam algorithm
        
        Parameters
        ----------
        n_features : int
            Number of features to optimize.
        alpha : float, optional
            Initial step size. The default is 0.001.
        beta_1 : float, optional
            Exponential decay rate for first moment estimate. The default is
            0.9.
        beta_2 : float, optional
            Exponential decay rate for second moment estimate. The default is
            0.999.
        epsilon : float, optional
            Small value to act as pseudocount to avoid division by zeor. The
            default is 10**(-8).

        Attributes
        ----------
        t : int
            Current time step.
        first_moments : numpy.ndarray
            Vector of estimated first moments.
        second_moments : numpy.ndarray
            Vector of estimated second moments.
        alpha : float
        beta_1 : float
        beta_2 : float
        epsilon : float

        Methods
        -------
        get_steps(gradients) : get gradient deltas for each feature gradient.

        References
        ----------
        Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic
        Optimization. Retrieved from http://arxiv.org/abs/1412.6980
        """
        self.t = 0
        self.first_moments = np.zeros(n_features)
        self.second_moments = np.zeros(n_features)
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def get_steps(self, gradients, *args):
        """
        Calculate gradient deltas for each feature gradient.
        
        Parameters
        ----------
        gradients : numpy.ndarray
            Calculated gradient delta of objective function with respect to
            given feature.

        Raises
        ------
        ValueError
            Checks if `gradient` fits expected dimensions.
        
        Returns
        -------
        numpy.ndarray
            Feature specific gradient steps for either ascent or descent.
        """        
        if self.first_moments.shape[0] != gradients.shape[0]:
            raise ValueError("Expected gradient vector with length" +
                             "{}. Got {}".format(self.first_moments.shape[0],
                                                 gradients.shape[0]))
        self.t += 1
        self.first_moments = self.beta_1 * self.first_moments \
                           + (1 - self.beta_1) * gradients
        self.second_moments = self.beta_2 * self.second_moments \
                            + (1 - self.beta_2) * gradients**2
        alpha_t = self.alpha * np.sqrt(1 - self.beta_2**self.t) \
                / (1 - self.beta_1**self.t)
        steps = alpha_t * self.first_moments \
              / (np.sqrt(self.second_moments)  + self.epsilon)
        return steps 

    
class KernelMixin(object):
    """Base class for kernel functions."""

    def __init__(self, sigma, reg, weights):
        """
        Base class for kernel functions.

        Parameters
        ----------
        sigma : float
            Scaling parameter sigma as mentioned in original paper.
        reg : float
            Regularization parameter.
        weights : numpy.ndarray
            Initial vector of feature weights.

        Methods
        -------
        transform():
            Apply the kernel tranformation to a distance matrix.

        gradients():
            Get current gradients for each feature
        update_weights():
            Update feature weights to a new value.
        """
        self.sigma = sigma
        self.reg = reg
        self.weights = weights

    def transform(self, distance_matrix):
        """Apply a kernel transformation to a distance matrix."""
        return distance_matrix

    def gradients(self, p_reference, feature_distances, class_matrix):
        """
        Calculate gradients with respect to feature weights.

        Calculates the gradient vector of the objective function with respect
        to feature weights. Objective function is the same outlined in the
        original NCFS paper.
        
        Parameters
        ----------
        p_reference : numpy.ndarray
            A (sample x sample) probability matrix with p_ij represents the
            probability of selecting sample j as a reference for sample i.
        feature_distances : numpy.ndarray
            A (feature x sample x sample) tensor, holding per feature sample
            distances, such that [l, i, j] indexes the distance between
            sample i and j in feature l.
        class_matrix : numpy.ndarray
            A one-hot (sample x sample) matrix where c_ij = 1 indicates sample
            i and sample j are in the same class, and c_ij = 0 otherwise.
        
        Returns
        -------
        numpy.ndarray
            Gradient vector for each feature with respect to the objective
            function.
        """
        deltas = np.zeros(self.weights.size, dtype=np.float64)
        # calculate probability of correct classification
        # check class mat one hot shit
        p_correct = np.sum(p_reference * class_matrix, axis=1)
        # caclulate weight adjustments for each feature
        for l in range(self.weights.size):
            # weighted distance matrix D_ij = d_ij * p_ij, p_ii = 0
            d_mat = feature_distances[l] * p_reference
            # calculate p_i * sum(D_ij), j from 0 to N
            all_term = p_correct * d_mat.sum(axis=1)
            # weighted in-class distances using adjacency matrix,
            in_class_term = np.sum(d_mat * class_matrix, axis=1)
            sample_terms = all_term - in_class_term
            # calculate delta following gradient ascent 
            deltas[l] = 2 * self.weights[l] \
                      * ((1 / self.sigma) * sample_terms.sum() - self.reg)
        return deltas

    def update_weights(self, new_weights):
        """
        Update feature weights.
        
        Parameters
        ----------
        new_weights : numpy.ndarray
            New feature weights.
        """
        self.weights = new_weights


class ExponentialKernel(KernelMixin):
    """Class for the exponential kernel function used in original NCFS paper."""

    def __init__(self, sigma, reg, weights):
        """
        Class for the exponential kernel function used in original NCFS paper.

        Extends KernelMixin.

        Parameters
        ----------
        sigma : float
            Scaling parameter sigma as mentioned in original paper.
        reg : float
            Regularization parameter.
        weights : numpy.ndarray
            Initial vector of feature weights.

        Methods
        -------
        transform():
            Apply the kernel tranformation to a distance matrix.

        gradients():
            Get current gradients for each feature
        update_weights():
            Update feature weights to a new value.
        """
        super(ExponentialKernel, self).__init__(sigma, reg, weights)

    def transform(self, distance_matrix):
        """
        Apply the kernel function to each sample distance in a distance matrix.
        
        Parameters
        ----------
        distance_matrix : numpy.ndarray
            A (sample x sample) distance matrix.
        
        Returns
        -------
        numpy.ndarray
            A (sampel x sample) kernel distance matrix.
        """
        return np.exp(-1 * distance_matrix / self.sigma, dtype=np.float64)


class GaussianKernel(KernelMixin):

    def __init__(self, sigma, reg, weights):
        """
        Class for sample-centered distance kernel function.

        The distance between samples i and j, d_ij, is centered by subtracting
        the average distance between sample i and all other samples.

        Extends KernelMixin.

        Parameters
        ----------
        sigma : float
            Scaling parameter sigma as mentioned in original paper.
        reg : float
            Regularization parameter.
        weights : numpy.ndarray
            Initial vector of feature weights.

        Methods
        -------
        transform():
            Apply the kernel tranformation to a distance matrix.

        gradients():
            Get current gradients for each feature
        update_weights():
            Update feature weights to a new value.
        """
        super(GaussianKernel, self).__init__(sigma, reg, weights)

    def transform(self, distance_matrix):
        """
        Apply the kernel function to each sample distance in a distance matrix.
        
        Parameters
        ----------
        distance_matrix : numpy.ndarray
            A (sample x sample) distance matrix.
        
        Returns
        -------
        numpy.ndarray
            A (sampel x sample) kernel distance matrix.
        """
        centered = center_distances(distance_matrix)
        return np.exp(-1 * centered / self.sigma, dtype=np.float64)


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
                 metric='euclidean', kernel='gaussian', solver='ncfs',
                 stochastic=False):
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
            are 'exponential' and 'gaussian'. The default is 'gaussian', whereas
            'exponential' follows the implementation in the original NCFS paper.
        solver : str, optional
            Method to perform gradient ascent. Possible values are 'ncfs' and
            'adam'. The defautl in 'ncfs', as used in the original NCFS paper.
        stochastic : boolean, optional
            Whether gradient ascent should be stochastic. Stochastic gradient
            ascent implies breaking the data set into 3 stratified partions and
            using each partition to update weights independently. Default is 
            False, and weight adjustment is done on the entire data matrix.

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
        stochastic : boolean
            Whether gradient ascent should be stochastic or determinent.
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
        self.stochastic = stochastic
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
        # initialize all weights as 1
        self.coef_ = np.ones(n_features, dtype=np.float64)

        # intialize kernel function, and assign expected feature distances 
        feature_distances = None
        if self.kernel == 'exponential':
            X = NCFS.__check_X(X)
            self.kernel_ = ExponentialKernel(self.sigma, self.reg, self.coef_)
            feature_distances = pairwise_feature_distance(X, metric=self.metric)
        elif self.kernel == 'gaussian':
            X = X.astype(np.float64)
            self.kernel_ = GaussianKernel(self.sigma, self.reg, self.coef_)
            feature_distances = pairwise_feature_distance(X, metric=self.metric)
            for l in range(feature_distances.shape[0]):
                feature_distances[l] = center_distances(feature_distances[l])
        else:
            raise ValueError('Unsupported kernel ' +
                             'function: {}'.format(self.kernel))

        if self.solver == 'ncfs':
            if self.stochastic:
                Warning("Converge with stochastic gradient ascent via NCFS "
                        "line search not guaranteed. It's recommended to use "
                        "'Adam' instead.")
            self.solver_ = NCFSOptimizer(alpha=self.alpha)
        elif self.solver == 'adam':
            self.solver_ = AdamOptimizer(n_features)
        else:
            raise ValueError('Unsupported gradient ascent method ' +
                             '{}'.format(self.solver))
        # construct adjacency matrix of class membership for matrix mult. 
        class_matrix = np.zeros((n_samples, n_samples), np.float64)
        for i in range(n_samples):
            for j in range(n_samples):
                if y[i] == y[j] and i != j:
                    class_matrix[i, j] = 1

        objective, loss = 0, np.inf
        while abs(loss) > self.eta:
            if self.stochastic:
                folds = model_selection.StratifiedKFold(n_splits=3,
                                                        shuffle=True)
                new_objective = objective
                for train_idxs, __ in folds.split(X, y):
                    random_X = X[train_idxs, :]
                    random_C = class_matrix[train_idxs, :][:, train_idxs]
                    random_F = feature_distances[:, train_idxs, :][:, :,
                                                                   train_idxs]
                    new_objective = self.__fit(random_X, random_C,
                                               new_objective, random_F)
            else:
                new_objective = self.__fit(X, class_matrix, objective,
                                           feature_distances)
            loss = objective - new_objective
            objective = new_objective

        self.score_ = objective
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
        return X * self.coef_

    def __fit(self, X, class_matrix, objective, feature_distances):
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
        # calculate D_w(x_i, x_j): w^2 * |x_i - x_j] for all i,j
        distances = spatial.distance.pdist(X, metric=self.metric,
                                           w=np.power(self.coef_, 2))
        # organize as distance matrix
        distances = spatial.distance.squareform(distances)
        # calculate K(D_w(x_i, x_j)) for all i, j pairs
        p_reference = self.kernel_.transform(distances)
        # set p_ii = 0, can't select self in leave-one-out
        for i in range(p_reference.shape[0]):
            p_reference[i, i] = 0

        # add pseudocount if necessary to avoid dividing by zero
        row_sums = p_reference.sum(axis=1)
        n_zeros = sum(row_sums == 0)
        if n_zeros > 0:
            print('Adding pseudocounts to distance matrix to avoid ' +
                    'dividing by zero.')
            if n_zeros == len(row_sums):
                pseudocount = np.exp(-20)
            else:
                pseudocount = np.min(row_sums)
            row_sums += pseudocount
        scale_factors = 1 / (row_sums)
        p_reference = (p_reference.T * scale_factors).T

        # caclulate weight adjustments
        gradients = self.kernel_.gradients(p_reference, feature_distances,
                                           class_matrix)
            
        # calculate objective function
        new_objective = (np.sum(p_reference * class_matrix) \
                        - self.reg * np.dot(self.coef_, self.coef_))
        # calculate loss from previous objective function
        loss = new_objective - objective
        # update weights
        deltas = self.solver_.get_steps(gradients, loss)
        self.coef_ = self.coef_ + deltas
        self.kernel_.update_weights(self.coef_)
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
    X, y = toy_dataset()
    f_select = NCFS(alpha=0.001, sigma=1, reg=1, eta=0.001, metric='cityblock',
                    kernel='exponential', solver='adam', stochastic=True)
    from timeit import default_timer as timer
    start = timer()
    f_select.fit(X, y)
    end = timer()
    print("Execution time in seconds: {}".format(end - start))
    print(np.argsort(-1 * f_select.coef_)[:10])
    print(f_select.coef_[0], f_select.coef_[100])

if __name__ == '__main__':
    main()
