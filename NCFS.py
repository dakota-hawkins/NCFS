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
import numba

class NCFS(base.BaseEstimator, base.TransformerMixin): 

    def __init__(self, alpha=0.1, sigma=1, reg=1, eta=0.001,
                 metric='cityblock'):
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
        if self.metric in ['cityblock', 'manhattan']:
            distance = manhattan
        elif self.metric == 'euclidean':
            distance = euclidean
        else:
            raise ValueError('Unsupported distance metric: {}'.\
                              format(self.metric))
        X= NCFS.__check_X(X)
        n_samples, n_features = X.shape
        # initialize all weights as 1
        self.coef_ = np.ones(n_features, dtype=np.float64)
        # instantiate feature deltas to zero
        deltas = np.zeros(n_features, dtype=np.float64)
        # get initial step size
        step_size = self.alpha 
        # construct adjacency matrix of class membership for matrix mult. 
        class_matrix = np.zeros((n_samples, n_samples), np.float64)
        for i in range(n_samples):
            for j in range(n_samples):
                if y[i] == y[j]:
                    class_matrix[i, j] = 1

        past_score, loss = 0, np.inf
        while abs(loss) > self.eta:
            # calculate probability of reference
            p_reference = reference_probabilities(X, self.coef_, self.sigma,
                                                  distance)
            # calculate probability of correct classification
            p_correct = p_reference * class_matrix
            # caclulate weight adjustments
            for l in range(X.shape[1]):
                # values for feature l starting with sample 0 to N
                # feature_vec = X[:, l].reshape(-1, 1)
                # distance in feature l for all samples, d_ij
                d_mat = spatial.distance.pdist(X[:, l].reshape(-1, 1),
                                               metric=self.metric)
                d_mat = spatial.distance.squareform(d_mat)
                # weighted distance matrix D_ij = d_ij * p_ij, p_ii = 0
                d_mat *= p_reference
                # calculate p_i * sum(D_ij), j from 0 to N
                all_term = p_correct * d_mat.sum(axis=0)
                # weighted in-class distances using adjacency matrix,
                in_class_term = np.sum(d_mat*class_matrix, axis=0)
                sample_terms = all_term - in_class_term
                # calculate delta following gradient ascent 
                deltas[l] = 2 * self.coef_[l] \
                            * ((1 / self.sigma) * sample_terms.sum() - self.reg)
            # # calculate objective function
            score = (np.sum(p_reference * class_matrix) \
                  - self.reg * np.dot(self.coef_, self.coef_))
            # calculate loss from previous objective function
            loss = score - past_score
            # update weights
            self.coef_ = self.coef_ + step_size * deltas
            # reset objective score for new iteration
            past_score = score
            if loss > 0:
                step_size *= 1.01
            else:
                step_size *= 0.4
        self.score_ = score
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

@numba.jit(nopython=True, parallel=True)
def manhattan(x, y, w):
    """Calculate a weighted manhattan distance."""
    value = 0
    for i in range(x.shape[0]):
        value += w[i] * np.abs(x[i] - y[i]) 
    return value

@numba.jit(nopython=True, parallel=True)
def euclidean(x, y, w):
    """Calculate a weighted euclidean distance."""
    value = 0
    for i in range(x.shape[0]):
        value += w[i] * (x[i] - y[i])**2
    return np.sqrt(value)


@numba.jit(nopython=True, parallel=True)
def distance_matrix(X, weights, distance):
    """Construct a weighted distance matrix."""
    dist_mat = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            dist_mat[i, j] = distance(X[i, :], X[j, :], weights)
    return dist_mat


@numba.jit(nopython=True, parallel=True)
def reference_probabilities(X, weights, sigma, distance):
    """
    Calculate reference probability matrix.
    
    Parameters
    ----------
    X : np.ndarray
        An (n x p) data matrix, where n is the number of samples and p is the
        number of features.
    weights : np.ndarray
        Feature weight vector. 
    sigma : float
        Kernel width
    distance : callable
        Distance function, either NCFS.manhattan or NCFS.euclidean.
    
    Returns
    -------
    np.ndarray
        Matrix P, where p_ij is the probability of selecting j as a reference
        for i.
    """
    # calculate D_w(x_i, x_j): w^2 * |x_i - x_j] for all i,j
    distances = distance_matrix(X, weights, distance)
    # calculate K(D_w(x_i, x_j)) for all i, j pairs
    p_reference = np.exp(-1 * distances / sigma)
    # set p_ii = 0, can't select self in leave-one-out
    for i in range(p_reference.shape[0]):
        p_reference[i, i] = 0
    # add pseudocount if necessary to avoid dividing by zero
    scale_factors = p_reference.sum(axis=0)
    n_zeros = (scale_factors == 0).sum()
    if n_zeros > 0:
        if n_zeros == scale_factors.shape[0]:
            pseudocount = np.exp(-20)
        else:
            pseudocount = np.min(scale_factors[scale_factors != 0])
        scale_factors += pseudocount
    for i in range(p_reference.shape[0]):
        # denom = scale_factors[i]
        for j in range(p_reference.shape[1]):
            p_reference[i, j] = p_reference[i, j] / scale_factors[j]
    return p_reference

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
    X, y = toy_dataset(n_features=1000)
    f_select = NCFS(alpha=0.01, sigma=1, reg=1, eta=10**(-3),
                    metric='cityblock')
    f_select.fit(X, y)