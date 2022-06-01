"""
Python implementation of Neighborhood Component Feature Selection

Yang, W., Wang, K., & Zuo, W. (2012). Neighborhood Component Feature Selection
for High-Dimensional Data. Journal of Computers, 7(1).
https://doi.org/10.4304/jcp.7.1.161-168

Author : Dakota Hawkins
"""
import warnings
import sys

import numba
import numpy as np
from scipy import spatial
from sklearn import base, model_selection, exceptions

from ncfs import distances
from ncfs import accelerated

# import distances
# import accelerated


class NCFS(base.BaseEstimator, base.TransformerMixin):
    """
    Class to perform Neighborhood Component Feature Selection

    References
    ----------

    Yang, W., Wang, K., & Zuo, W. (2012). Neighborhood Component Feature
    Selection for High-Dimensional Data. Journal of Computers, 7(1).
    https://doi.org/10.4304/jcp.7.1.161-168
    """

    def __init__(
        self,
        alpha=0.1,
        sigma=1,
        reg=1,
        eta=0.001,
        metric="cityblock",
        kernel="exponential",
        max_iter=1000,
    ):
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
        max_iter : int, None, optional
            Number of max iterations during fitting. Default is 1000. If None,
            fitting will continue until convergence.

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
        self.max_iter = max_iter
        self.warm_start_ = False
        # self._debug = True
        # if debug:
        #     import logging
        #     logging.basicConfig(filename='ncfs.log')

    @staticmethod
    def __check_X(X):
        if not np.isclose(X.min(), 0) or not np.isclose(X.max(), 1):
            warnings.warn(
                "Data matrix contains values outside of the [0, 1] "
                "interval. May be numerical unstable and lead to "
                "pseudocount additions during fitting."
            )
        return X.astype(np.float64)

    def __check_params(self, X, y):
        """
        Check params for NCFS fitting

        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix.
        y : numpy.ndarray
            A sample length vector of class labels.
        """
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha value should be between 0 and 1.")
        if not isinstance(X, np.ndarray):
            raise ValueError(
                "`X` must be two-dimensional numpy array. Got " + "{}.".format(type(X))
            )
        if len(X.shape) != 2:
            raise ValueError(
                "`X` must be two-dimensional numpy array. Got "
                + "{} dimensional.".format(len(X.shape))
            )
        if isinstance(y, list):
            y = np.array(y)
        if not isinstance(y, np.ndarray):
            if "pandas" in sys.modules:
                if isinstance(y, sys.modules["pandas"].Series):
                    y = y.values
            else:
                raise ValueError(
                    "`y` must be a numpy array. " + "Got {}.".format(type(y))
                )
        if y.shape[0] != X.shape[0]:
            raise ValueError("`X` and `y` must have the same row numbers.")
        if self.max_iter is None:
            self.max_iter = np.inf

        self.n_features_ = X.shape[1]
        # check if model has been previously fit
        if not self.warm_start_:
            # initialize objective score as zero
            self.score_ = 0
            # initialize all weights as 1
            self.coef_ = np.ones(self.n_features_, dtype=np.float64)
            self.warm_start_ = True
        else:
            # if not warm start, ensure expected .coef_ structure
            if not isinstance(self.coef_, np.ndarray):
                raise ValueError(
                    "Expected numpy array for self.coef_ "
                    "during warm start. Got {}.".format(type(self.coef_))
                )
            if self.coef_.size != self.n_features_:
                raise ValueError(
                    "Expected {} ".format(self.coef_.size)
                    + "features on warm start. Got {}.".format(self.n_features_)
                )
        # check distance metric
        if self.metric not in distances.supported_distances:
            raise ValueError(
                f"Unsupported distance metric {self.metric}"
                "Supported metrics are {}".format(distances.metrics.keys())
            )
        # initialize distance and gradient functions
        self.distance_, self.distance_grad_ = distances.metrics[self.metric]
        # intialize kernel function
        if self.kernel == "exponential":
            X = NCFS.__check_X(X)
            self.kernel_ = accelerated.exponential_transform
        else:
            raise ValueError("Unsupported kernel function: {}".format(self.kernel))
        return (X, y)

    @staticmethod
    def __check_sample_weights(y, sample_weights):
        if sample_weights is None:
            sample_weights = np.ones(y.size)
        elif sample_weights == "balanced":
            sample_weights = NCFS.calculate_sample_weights(y)
        else:
            if isinstance(sample_weights, list):
                sample_weights = np.array(sample_weights)
            if isinstance(sample_weights, np.ndarray):
                if sample_weights.size != y.size:
                    raise ValueError(
                        "Size of sample weight array does not " "match size of samples."
                    )
            else:
                raise TypeError(
                    "Unsupported type for sample weights: "
                    "{}".format(type(sample_weights))
                )
        return sample_weights

    @staticmethod
    def calculate_sample_weights(y):
        """
        Calculate balanced sample weights.

        Calculates balanced sample weights per scikit-learn documentation, where

        class_weight = (n_samples) / (n_classes * class_size)

        Parameters
        ----------
        y : numpy.ndarray
            A sample-length vector of class labels.

        Returns
        -------
        numpy.ndarray
            A sample-length vector of class-balanced sample weights.

        References
        ----------
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

        Logistic Regression in Rare Events Data, King, Zen, 2001
        """
        labels, counts = np.unique(y, return_counts=True)
        n_labels = len(labels)
        sample_weights = np.zeros(y.size)
        # max_count = counts.max()
        for label, count in zip(labels, counts):
            # calculate w = (n_samples) / (n_classes * |c_i|)
            weight = y.size / (n_labels * count)
            sample_weights[np.where(y == label)[0]] = weight
        return sample_weights

    @staticmethod
    def calculate_class_matrix(y):
        """
        Construct adjacency matrix of class membership.

        Construct matrix of class membership such that a_ij == 1 if sample
        i and j share the class label.

        Parameters
        ----------
        y : numpy.ndarray
            A sample-length vector of class labels.

        Returns
        -------
        numpy.ndarray
            Adjacency matrix for class membership.
        """
        # construct adjacency matrix of class membership for matrix mult.
        class_matrix = np.zeros((y.size, y.size), np.float64)
        for i in range(y.size):
            for j in range(y.size):
                if y[i] == y[j] and i != j:
                    class_matrix[i, j] = 1
        return class_matrix

    def partial_fit(self, X, y, sample_weights=None):
        warnings.warn(
            "Calling `partial_fit()` directly will lead "
            "to a performance loss due to continually copying data "
            "between Pythong and Numba jit-compiled C++ code. Calling "
            "`fit()` is recommended if possible."
        )
        if sample_weights == "balanced":
            raise ValueError(
                "Balanced sample weights is unsupported during "
                "partial fit. If balanced weights are desired, "
                "call `NCFS.calculate_sample_weights()` on "
                "either a representative sample or the complete "
                "set of labels for the given dataset."
            )
        X, y = self.__check_params(X, y)
        class_matrix = NCFS.calculate_class_matrix(y)
        sample_weights = NCFS.__check_sample_weights(y, sample_weights)
        # this should be a jitted function
        distance_matrix = np.zeros_like(class_matrix)
        partials_vec = np.zeros_like(self.coef_)
        self.score_ = accelerated.partial_fit(
            X,
            class_matrix,
            sample_weights,
            self.coef_,
            self.distance_,
            self.distance_grad_,
            self.kernel_,
            self.sigma,
            self.reg,
            self.alpha,
            distance_matrix,
            partials_vec,
            self.score_,
        )

        return self

    def fit(self, X, y, sample_weights=None):
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
        X, y = self.__check_params(X, y)
        class_matrix = NCFS.calculate_class_matrix(y)
        sample_weights = NCFS.__check_sample_weights(y, sample_weights)
        distance_matrix = np.zeros_like(class_matrix)
        partial_vec = np.zeros_like(self.coef_)
        self.score_, i = accelerated.fit(
            X,
            class_matrix,
            sample_weights,
            self.coef_,
            self.distance_,
            self.distance_grad_,
            self.kernel_,
            self.sigma,
            self.reg,
            self.alpha,
            self.eta,
            self.max_iter,
            distance_matrix,
            partial_vec,
        )
        if i >= self.max_iter:
            warnings.warn(
                "Number of max iterations reached before convergence."
                "Fit may be poor. Consider increasing the number of "
                "max number of iterations."
            )
        if any([np.isnan(x) for x in self.coef_]):
            raise exceptions.FitFailedWarning(
                "NaN values in coefficients. " "Fit failed. Try scaling data."
            )
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

        try:
            self.coef_
        except NameError:
            raise RuntimeError(
                "NCFS is not fit. Please fit the " + "estimator by calling `.fit()`."
            )
        if X.shape[1] != len(self.coef_):
            raise ValueError(
                "Expected data matrix `X` to contain the same"
                + "number of features as learnt feature weights."
            )
        if self.kernel == "exponential":
            X = NCFS.__check_X(X)
        # do not square weights when transforming -- break expected behavior
        return X * self.coef_

    def score(self, X, y, sample_weights=None):
        """
        Score the current fit of an NCFS object.

        Parameters
        ----------
        X : numpy.ndarray
            A (sample x feature) data matrix.
        y : numpy.ndarray
            A sample-length vector of class labels
        sample_weights : numpy.ndarray, string, None, optional
            Weights for each sample. Default is none and each sample will have
            a weight of 1. If 'balanced' is passed, weights proportional to
            class frequency are calculated.

        Returns
        -------
        float
            Regularized accuracy in a KNN classifier during leave-one-out
            validation.
        """
        self.__check_X(X)
        X, y = self.__check_params(X, y)
        sample_weights = self.__check_sample_weights(y, sample_weights)
        class_matrix = self.calculate_class_matrix(y)
        distance_matrix = np.zeros_like(class_matrix)
        return accelerated.score(
            X,
            class_matrix,
            sample_weights,
            self.coef_,
            distance_matrix,
            self.distance_,
            self.kernel_,
            self.sigma,
            self.reg,
        )


def log_system_info():
    import os
    import logging
    import psutil

    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info().rss / 2 ** 30
    logging.info("Memory usage during NCFS: {:0.03} GB".format(memory_use))


def toy_dataset(n_features=1000, n1=100, n2=100):
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
    n1 : int, optional
        Number of samples in population 1. Default is 100.
    n2 : int, optional
        Number of samples in population 2. Default is 100.

    Returns
    -------
    tuple (X, y)
        X : numpy.array
            Simulated dataset with 200 samples (rows) and N features. Features
            are scaled between 0 and 1.
        y : numpy.array
            Class membership for each sample in X.
    """

    class_1 = np.zeros((n1, 2))
    class_2 = np.zeros((n2, 2))
    cov = np.identity(2)
    for i in range(n1):
        r1 = np.random.rand()
        if r1 > 0.5:
            class_1[i, :] = np.random.multivariate_normal([-0.75, -3], cov)
        else:
            class_1[i, :] = np.random.multivariate_normal([0.75, 3], cov)
    for i in range(n2):
        r2 = np.random.rand()
        if r2 > 0.5:
            class_2[i, :] = np.random.multivariate_normal([3, -3], cov)
        else:
            class_2[i, :] = np.random.multivariate_normal([-3, 3], cov)
    class_data = np.vstack((class_1, class_2))
    n_irrelevant = n_features - 2
    second_idx = int(0.1 * (n_features)) - 1
    bad_features = np.random.normal(
        loc=0, scale=np.sqrt(20), size=(n1 + n2, n_irrelevant)
    )
    data = np.hstack(
        (
            class_data[:, 0].reshape(-1, 1),
            bad_features[:, :second_idx],
            class_data[:, 1].reshape(-1, 1),
            bad_features[:, second_idx:],
        )
    )
    classes = np.array([0] * n1 + [1] * n2)
    # scale between 0 - 1
    x_std = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return x_std, classes


def main():
    X, y = toy_dataset(n_features=1000, n1=150, n2=50)
    f_select = NCFS(
        alpha=0.01, sigma=1, reg=1, eta=0.001, metric="manhattan", kernel="exponential"
    )
    from timeit import default_timer as timer

    times = np.zeros(10)
    # previous 181.82286000379972
    # not parallel: 116
    # parallel, jobs=1 = 124
    # expanding dims = 150s
    # vectorized 104
    # NUMBA -- 80s w/ manhattan, euclidean doesnt' converge, sq does -- 80s
    # NUMBA -- 40S w/ manhattan + accelerated gradients
    # NUMBA -- 703 ms w/ manhattan + for loops
    for i in range(times.size):
        start = timer()
        f_select.fit(X, y, sample_weights="balanced")
        end = timer()
        times[i] = end - start
    print("Average execution time in seconds: {}".format(np.mean(times)))
    sorted_coef = np.argsort(-1 * f_select.coef_)
    print(sorted_coef[:10])
    print(f_select.coef_[0], f_select.coef_[100], f_select.coef_[sorted_coef[2]])
    print(f_select.score(X, y))


if __name__ == "__main__":
    main()
